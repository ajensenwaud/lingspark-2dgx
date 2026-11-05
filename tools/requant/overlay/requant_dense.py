"""
Dense-only requantization of an existing EXL3 pack (exllamav3 v1.6.6 port of Muspark's patches-exl3 0005).

Requantizes selected non-expert linears of the decoder blocks at new bitrates and keeps every other tensor of the base
pack byte for byte: routed and shared experts, dense MLPs, other attention kinds, embedding, norms, router, head, MTP.
Built for packs whose expert matrices dominate the conversion time, where only some dense bits change. LingSpark use:
the KDA (linear attention) projections qkv/f/g/o of Ling-3.1-flash from 5 to 4 bits:

  python3 -m exllamav3.conversion.requant_dense -b /base -s /src -w /work --bits 4 \
      --include "*.attention.qkv_proj,*.attention.f_proj,*.attention.g_proj,*.attention.o_proj" \
      --layer_types linear_attention [--ids_ref <converter ckpt>/original_input_ids.safetensors] [-r]

Per decoder block, in order, the converter's own steps restricted to the target linears (convert_model.py main loop):

  1. Load the block with the base pack as the model dir: the target linears' EXL3 tensors are hidden and only their
     source tensors (`.weight`, fp8 `.weight_scale_inv`, ...; alt keys such as q/k/v for a fused qkv) are added from
     --dense_src (header filtered, so no other base tensor is overridden). Every other Linear's qmap is cleared, so
     only the targets capture. Normal top-k routing (no all-experts pass: the attention Hessians do not depend on it).
  2. Capture pass over the calibration state (the converter's params, batched --batch rows a forward).
  3. Quantize through convert_model.quantize_linears_parallel (the converter's grouping, seed = module index, codebook,
     out scales, Hessian regularisation) and save <work>/qtensors/<block key>.safetensors (targets only).
  4. Optionally the base pack's tensors of the same linears on the reference rows (rfn_base).
  5. Reload with the new tensors and advance the calibration state: the next block is calibrated on the output of the
     requantized blocks before it, as in the converter. Blocks without targets are advanced with the base pack.

Calibration: the converter's get_default_calibration() at the base pack's rows/cols (or a converter checkpoint via
--seed); --ids_ref requires identical rows; the rows' sha256 is printed and kept in report/done files. Checkpoints use
the converter's job format; -r resumes. The pack is assembled from the base pack by a separate byte-copy step
(tools/requant/dense_pack.py).
"""

import argparse
import fnmatch
import glob
import hashlib
import json
import os
import re
import shutil
import sys
import time

col_default = "\u001b[0m"
col_red = "\u001b[31;1m"

EXPERT_RE = re.compile(r"\.experts\.\d+\.")
EXL3_SUBKEYS = ("su", "sv", "suh", "svh", "trellis", "bias", "mcg", "mul1")
NUM_REF_ROWS = 5

parser = argparse.ArgumentParser(allow_abbrev = False, description = "Dense-only requantization of an EXL3 pack")
parser.add_argument("-b", "--base", type = str, default = None, help = "Base EXL3 pack (config, tokenizer, and every tensor that is not requantized)")
parser.add_argument("-s", "--dense_src", type = str, default = None, help = "Unquantized source of the target linears: directories or .safetensors files (comma separated); only the targets' tensors are read from them")
parser.add_argument("-w", "--work_dir", type = str, required = True, help = "Working directory (job state, qtensors, report)")
parser.add_argument("--bits", type = float, default = None, help = "Bitrate of the target linears (1-8, or 1.5/2.5/3.5 with the mul1 codebook)")
parser.add_argument("--override", type = str, default = "", help = "Per-tensor bitrates, comma separated pattern=bits, fnmatch on the full key, last match wins")
parser.add_argument("--include", type = str, default = "", help = "Target linears: comma separated fnmatch patterns on the full key (default: every non-expert Linear with an EXL3 tensor in the base)")
parser.add_argument("--layer_types", type = str, default = "", help = "Only blocks whose config.layer_types entry is one of these (comma separated), e.g. linear_attention")
parser.add_argument("--layers", type = str, default = None, help = "Decoder layers to requantize, a-b (inclusive), default all")
parser.add_argument("--seed", type = str, default = None, help = "Start from this checkpoint dir (job.json, state.safetensors, original_input_ids.safetensors)")
parser.add_argument("--advance_base", action = "store_true", help = "Run the decoder layers before --layers with the base pack's tensors (unchanged) to reach the first layer")
parser.add_argument("--ids_ref", type = str, default = None, help = "original_input_ids.safetensors of the base conversion: require identical calibration rows")
parser.add_argument("-cr", "--cal_rows", type = int, default = None, help = "Calibration rows (default: the base pack's)")
parser.add_argument("-cc", "--cal_cols", type = int, default = None, help = "Calibration columns (default: the base pack's)")
parser.add_argument("-cb", "--codebook", type = str, default = None, help = "Codebook (default: the base pack's)")
parser.add_argument("--batch", type = int, default = 10, help = "Calibration rows per forward pass")
parser.add_argument("--verify_batch", action = "store_true", help = "First requantized layer: also run the first batch's rows one at a time and report the difference")
parser.add_argument("--no_ref_base", action = "store_true", help = "Skip the base pack's tensors on the reference rows (rfn_base)")
parser.add_argument("--max_layers", type = int, default = None, help = "Stop after this many requantized layers (timing trials); the job stays resumable")
parser.add_argument("-cpi", "--checkpoint_interval", type = int, default = 600, help = "Minimum seconds between checkpoints; < 0: only at the end of a range")
parser.add_argument("-d", "--device", type = int, default = 0, help = "CUDA device")
parser.add_argument("-r", "--resume", action = "store_true", help = "Resume the job in --work_dir")
parser.add_argument("--plan", action = "store_true", help = "Print the plan (tensors, bitrates, sizes, sources found) and exit; no GPU work")


def parse_overrides(s):
    out = []
    for part in [p.strip() for p in s.split(",") if p.strip()]:
        assert "=" in part, f"--override entry without '=': {part}"
        pat, bits = part.rsplit("=", 1)
        out.append((pat.strip(), float(bits)))
    return out


def parse_layers(s, num_layers):
    if not s:
        return 0, num_layers - 1
    a, b = (s.split("-") + [None])[:2]
    a = int(a)
    b = int(b) if b not in (None, "") else num_layers - 1
    assert 0 <= a <= b < num_layers, f"--layers {s} outside 0-{num_layers - 1}"
    return a, b


def valid_bits(v, codebook):
    from .allocation import HALF_RATES
    if float(v).is_integer() and 1 <= v <= 8:
        return True
    return codebook == "mul1" and float(v) in HALF_RATES


def norm_bits(v):
    return int(v) if float(v).is_integer() else float(v)


def dense_strategy(linears, bits, overrides):
    strategy = {}
    for lin in linears:
        k = bits
        for pat, b in overrides:
            if fnmatch.fnmatchcase(lin.key, pat):
                k = b
        strategy[lin.key] = norm_bits(k)
    return strategy


def block_linears(block, include):
    """(targets, others): targets = non-expert Linears matching --include (all non-expert ones without it)"""
    from ..modules import Linear
    targets, others = [], []
    for m in block:
        if not isinstance(m, Linear):
            continue
        is_target = not EXPERT_RE.search(m.key + ".") and (not include or any(fnmatch.fnmatchcase(m.key, p) for p in include))
        (targets if is_target else others).append(m)
    return targets, others


def source_names(lin):
    names = {lin.key}
    alt = getattr(lin, "alt_key", None)
    if isinstance(alt, str):
        names.add(alt)
    elif alt:
        names.update(alt)
    return names


def sha_rows(rows):
    import torch
    h = hashlib.sha256()
    for t in rows:
        h.update(t.contiguous().numpy().tobytes() if t.dtype == torch.int64 else t.contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def save_list(tensors, path):
    from ..loader.safetensors_alt import save_file
    save_file({f"tensor.{i}": t for i, t in enumerate(tensors)}, path)


def load_list(path):
    from safetensors import safe_open
    with safe_open(path, framework = "pt", device = "cpu") as f:
        keys = set(f.keys())
        if "tensor" in keys:
            return [f.get_tensor("tensor")]
        out, i = [], 0
        while f"tensor.{i}" in keys:
            out.append(f.get_tensor(f"tensor.{i}"))
            i += 1
        return out


def write_checkpoint(work, job, state, ids):
    """ckpt_new -> ckpt (previous ckpt -> ckpt_old, removed first): at most two state copies, always one complete"""
    ck, ck_new, ck_old = (os.path.join(work, d) for d in ("ckpt", "ckpt_new", "ckpt_old"))
    if os.path.exists(ck_old) and os.path.exists(os.path.join(ck, "job.json")):
        shutil.rmtree(ck_old)
    if os.path.exists(ck_new):
        shutil.rmtree(ck_new)
    os.makedirs(ck_new)
    with open(os.path.join(ck_new, "job.json"), "w") as f:
        json.dump(job, f, indent = 4)
    save_list(state, os.path.join(ck_new, "state.safetensors"))
    save_list(ids, os.path.join(ck_new, "original_input_ids.safetensors"))
    if os.path.exists(ck):
        if os.path.exists(ck_old):
            shutil.rmtree(ck_old)
        os.rename(ck, ck_old)
    os.rename(ck_new, ck)


def add_filtered_tensor_files(stc, sources, wanted_names):
    """Add the source keys whose module name (key minus its last part) is in wanted_names (headers only).
    Returns the set of keys added."""
    from ..loader.safetensors import read_header
    files = []
    for s in sources:
        if s.endswith(".safetensors") and os.path.isfile(s):
            files.append(s)
        else:
            files += sorted(glob.glob(os.path.join(s, "*.safetensors")))
    found = set()
    for fn in files:
        header = read_header(fn, stc.tensor_name_fixes)
        keys = [k for k in header if k not in ("__metadata__", "_header_offset") and k.rpartition(".")[0] in wanted_names]
        if not keys:
            continue
        assert not any(k in stc.tensor_file_map for k in keys), f"{fn}: would override base pack tensors"
        if fn not in stc.file_headers:
            stc.tensor_files.append(fn)
            stc.handles[fn] = None
        stc.file_headers[fn] = header
        for k in keys:
            stc.tensor_file_map[k] = fn
            found.add(k)
    return found


class Batches:
    def __init__(self, n, bad_rows, batch):
        rows = [i for i in range(n) if i not in bad_rows]
        self.chunks = [rows[c : c + batch] for c in range(0, len(rows), batch)]


def run_forward(module, state, ids, chunk, extra = None):
    import torch
    params = {"attn_mode": "flash_attn_nc", "input_ids": torch.cat([ids[i] for i in chunk], dim = 0),
              "quant_preserve": {}}
    if extra:
        params.update(extra)
    x = module.prepare_for_device(torch.cat([state[i] for i in chunk], dim = 0), params)
    return module.forward(x, params)


def state_error(x, ref):
    import torch
    from ..util.measures import cosine_error, sqnr
    x = x.view(-1, x.shape[-1]).float()
    ref = ref.view(-1, ref.shape[-1]).float().to(x.device)
    err = torch.linalg.norm(x - ref, "fro") / torch.linalg.norm(ref, "fro")
    return err.item(), cosine_error(x, ref), sqnr(x, ref)


def load_module(module, device, stc, new_tensors = None):
    import torch
    stc.set_new_tensors(new_tensors)
    try:
        defer = module.can_defer_load()
        if defer:
            stc.begin_deferred_load()
        try:
            dev = torch.device("cpu") if module.caps.get("prefer_cpu") else device
            if new_tensors is not None:
                module.load(dev, source = new_tensors, keep_source_weights = True)
            else:
                module.load(dev, keep_source_weights = True)
        finally:
            if defer:
                stc.end_deferred_load()
    finally:
        stc.set_new_tensors(None)
    stc.close()


def human_time(s):
    s = int(s)
    return f"{s // 3600} h {s % 3600 // 60:02d} m" if s >= 3600 else f"{s // 60} m {s % 60:02d} s"


def prepare(args):
    work = args.work_dir
    os.makedirs(os.path.join(work, "qtensors"), exist_ok = True)
    args_path = os.path.join(work, "args.json")
    if args.resume and os.path.exists(args_path):
        in_args = json.load(open(args_path))
        for k in ("base", "bits", "override", "include", "layer_types", "layers", "cal_rows", "cal_cols", "codebook"):
            v = getattr(args, k)
            if v is not None and v != "" and v != in_args.get(k):
                raise SystemExit(f" ## Resuming with {k} = {in_args.get(k)}, cannot change it to {v}")
        if args.dense_src:
            in_args["dense_src"] = args.dense_src
        for k in ("batch", "checkpoint_interval", "device", "no_ref_base", "max_layers"):
            in_args[k] = getattr(args, k)
        ck_job = os.path.join(work, "ckpt", "job.json")
        if os.path.exists(ck_job):
            return in_args, json.load(open(ck_job)), True
        return in_args, {"next_module_idx": 0, "q_strategy": None, "bad_rows": [], "mode": "requant_dense"}, False

    assert args.base and args.dense_src and args.bits is not None, "--base, --dense_src and --bits are required"
    assert not os.path.exists(os.path.join(work, "ckpt", "job.json")), f"{work} holds a job already (use -r)"
    base_cfg = json.load(open(os.path.join(args.base, "config.json")))
    qcfg = base_cfg.get("quantization_config", {})
    assert qcfg.get("quant_method") == "exl3", f"{args.base} is not an EXL3 pack"
    cal = qcfg.get("calibration", {})
    in_args = {
        "base": args.base, "dense_src": args.dense_src, "bits": args.bits, "override": args.override,
        "include": args.include, "layer_types": args.layer_types, "layers": args.layers, "seed": args.seed,
        "advance_base": args.advance_base,
        "cal_rows": args.cal_rows or cal.get("rows", 250),
        "cal_cols": args.cal_cols or cal.get("cols", 2048),
        "codebook": args.codebook or qcfg.get("codebook", "mul1"),
        "out_scales": qcfg.get("out_scales", "auto"),
        "batch": args.batch, "checkpoint_interval": args.checkpoint_interval, "device": args.device,
        "no_ref_base": args.no_ref_base, "ids_ref": args.ids_ref, "max_layers": args.max_layers,
    }
    job = {"next_module_idx": 0, "q_strategy": None, "bad_rows": [], "mode": "requant_dense"}
    return in_args, job, False


def main(args):
    import torch
    from .. import Config, Model, Tokenizer
    from ..loader.safetensors import SafetensorsCollection
    from ..modules.quant import LinearFP16, LinearEXL3
    from ..modules.quant.exl3_lib import quantize as exl3_quantize
    from ..modules.quant.exl3_lib.quantize import get_temp_buffers
    from ..util.memory import free_mem, malloc_trim, is_integrated_device
    from . import convert_model as cm
    from .calibration_data import get_default_calibration

    torch.set_grad_enabled(False)
    work = args.work_dir
    if os.path.exists(os.path.join(work, "requant-done.json")):
        print(f" -- {work}: requantization already done (requant-done.json)")
        return
    in_args, job, from_ckpt = prepare(args)
    codebook = in_args["codebook"]
    bits = in_args["bits"]
    overrides = parse_overrides(in_args["override"])
    include = [p.strip() for p in (in_args.get("include") or "").split(",") if p.strip()]
    layer_types = [p.strip() for p in (in_args.get("layer_types") or "").split(",") if p.strip()]
    assert valid_bits(bits, codebook), f"--bits {bits} is not a supported rate"
    for pat, b in overrides:
        assert valid_bits(b, codebook), f"--override {pat}={b}: not a supported rate"

    config = Config.from_directory(in_args["base"])
    # As the converter (get_base_model): block-scaled fp8 source tensors stay packed and dequantize per use
    config.infer_params.packed_fp16_source = True
    model = Model.from_config(config)
    stc = config.stc
    num_layers = config.num_hidden_layers
    first = model.first_block_idx
    la, lb = parse_layers(in_args["layers"], num_layers)
    blocks = model.modules[first : first + num_layers]
    cfg_types = getattr(config, "layer_types", None)

    plan = {}
    for li, block in enumerate(blocks):
        targets, others = block_linears(block, include)
        if layer_types and cfg_types[li] not in layer_types:
            others, targets = others + targets, []
        targets = [t for t in targets if stc.has_tensor(f"{t.key}.trellis")]
        for m in others:
            m.qmap = None                     # only the targets capture
        if la <= li <= lb and targets:
            plan[li] = (targets, dense_strategy(targets, bits, overrides))
    assert plan, "no target linears in the range"
    la, lb = min(plan), max(plan)             # the state is not needed past the last target block
    wanted = {}
    hidden = {}
    by_name = {}
    for k in stc.tensor_file_map:
        by_name.setdefault(k.rpartition(".")[0], []).append(k)
    for li, (targets, _) in plan.items():
        for lin in targets:
            for n in source_names(lin):
                wanted[n] = lin
            for k in by_name.get(lin.key, []):
                hidden[k] = stc.tensor_file_map.pop(k)
    found = add_filtered_tensor_files(stc, in_args["dense_src"].split(","), set(wanted))
    have = {k.rpartition(".")[0] for k in found}
    missing = sorted(lin.key for _, (t, _) in plan.items() for lin in t if not (source_names(lin) & have))

    numel_new = {}
    for li, (targets, strat) in plan.items():
        for lin in targets:
            kind = lin.key.rsplit(".", 1)[-1]
            numel_new.setdefault((kind, strat[lin.key]), 0)
            numel_new[(kind, strat[lin.key])] += lin.weights_numel()
    n_t = sum(len(t) for t, _ in plan.values())
    print(f" -- Dense requantization of {in_args['base']}")
    print(f"    Layers {la}-{lb} of {num_layers}: {len(plan)} blocks with targets, {n_t} target linears, "
          f"{len(hidden)} base tensors hidden; every other tensor from the base pack")
    print(f"    Target blocks: {sorted(plan)}")
    for (kind, k), n in sorted(numel_new.items()):
        print(f"    {kind:20s} {k:>4} bpw  {n / 1e9:7.3f} G weights")
    print(f"    Source: {in_args['dense_src']} ({len(found)} tensors for {len(have)} names)")
    print(f"    Calibration: {in_args['cal_rows']} rows x {in_args['cal_cols']}, batch {in_args['batch']}, "
          f"codebook {codebook}, out scales {in_args['out_scales']}")
    if missing:
        print(f" !! {col_red}Missing source tensors{col_default}: {missing[:5]}{' ...' if len(missing) > 5 else ''}")
        raise SystemExit(1)
    if args.plan:
        return

    json.dump(in_args, open(os.path.join(work, "args.json"), "w"), indent = 4)
    device = torch.device(in_args["device"])
    exl3_quantize.NO_SWAP = is_integrated_device(in_args["device"])        # as convert_model.main (GB10: True)
    qargs = {
        "work_dir": work, "apply_out_scales": {"always": True, "never": False}.get(in_args["out_scales"]),
        "out_scales_hessian": False, "hessians": "", "hessians_reg": 0.025, "scaled_hessian": False,
        "codebook": codebook, "image_dump": False, "verbose": False,
    }
    base_stc = SafetensorsCollection(in_args["base"]) if not in_args["no_ref_base"] else None

    qlog = {}
    orig_print = cm.print_quantized_linear
    def print_quantized_linear(config_, linear, quant_args, proxy_err, time_str = ""):
        qlog[linear.key] = {"bits": quant_args["K"], "proxy_err": float(proxy_err)}
        orig_print(config_, linear, quant_args, proxy_err, time_str)
    cm.print_quantized_linear = print_quantized_linear

    if from_ckpt:
        ck = os.path.join(work, "ckpt")
        state = load_list(os.path.join(ck, "state.safetensors"))
        ids = load_list(os.path.join(ck, "original_input_ids.safetensors"))
        print(f" -- Resuming at module {job['next_module_idx']}")
    elif in_args["seed"]:
        sd = in_args["seed"]
        j = json.load(open(os.path.join(sd, "job.json")))
        job = {"next_module_idx": j["next_module_idx"], "q_strategy": None, "bad_rows": j.get("bad_rows") or [],
               "mode": "requant_dense"}
        state = load_list(os.path.join(sd, "state.safetensors"))
        ids = load_list(os.path.join(sd, "original_input_ids.safetensors"))
        print(f" -- Seeded from {sd} at module {job['next_module_idx']}")
    else:
        tokenizer = Tokenizer.from_config(config)
        rows = get_default_calibration({"cal_rows": in_args["cal_rows"], "cal_cols": in_args["cal_cols"]}, tokenizer)
        ids = [r.clone() for r in rows]
        state = rows
        print(f" -- Calibration data: {len(rows)} rows")
    ids_sha = sha_rows(ids)
    print(f" -- Calibration ids sha256: {ids_sha}")
    if in_args.get("ids_ref"):
        ref_ids = load_list(in_args["ids_ref"])
        same = len(ref_ids) == len(ids) and all(a.shape == b.shape and torch.equal(a, b) for a, b in zip(ref_ids, ids))
        assert same, f"calibration rows differ from {in_args['ids_ref']}"
        print(f" -- Calibration rows identical to {in_args['ids_ref']} ({len(ids)} rows)")
    bad_rows = set(job.get("bad_rows") or [])
    nxt = job["next_module_idx"]
    end_idx = first + lb + 1
    if not from_ckpt and nxt > first + la:
        raise SystemExit(f" ## The seed is at module {nxt}, past layer {la} (module {first + la})")

    report_path = os.path.join(work, "report.jsonl")
    last_ckpt = time.time()
    batch = in_args["batch"]
    t_job = time.time()
    done_layers = 0

    def advance(module, measure = None):
        errs = []
        for chunk in Batches(len(state), bad_rows, batch).chunks:
            rs = run_forward(module, state, ids, chunk).cpu()
            for j, i in enumerate(chunk):
                r = rs[j : j + 1].clone()
                if not torch.isfinite(r).all().item():
                    bad_rows.add(i)
                    print(f" !! Non-finite hidden state in calibration row {i}, excluding row")
                elif measure is not None and i in measure:
                    errs.append(state_error(r, measure[i]))
                state[i] = r
        cm.check_bad_rows(bad_rows, len(state))
        return errs

    for idx in range(nxt, end_idx):
        module = model.modules[idx]
        get_temp_buffers.cache_clear()
        free_mem()
        t0 = time.time()
        li = idx - first

        if li not in plan:
            # Embedding, or a block without targets (or before the range): advanced with the base pack's tensors.
            # The embedding runs as the converter's module 0 does (row by row)
            load_module(module, device, stc)
            if idx < first:
                for i in range(len(state)):
                    p = {"attn_mode": "flash_attn_nc", "input_ids": ids[i], "quant_preserve": {}}
                    state[i] = module.forward(module.prepare_for_device(state[i], p), p).cpu()
            else:
                advance(module)
            module.unload()
            print(f" -- Advanced (base pack): {module.key}  [{time.time() - t0:.1f} s]", flush = True)
        else:
            targets, strategy = plan[li]
            secs = {}

            # 1. Targets from the source, everything else from the base pack
            load_module(module, device, stc)
            for lin in targets:
                assert isinstance(lin.inner, LinearFP16), f"{lin.key}: expected the unquantized source weight"
            secs["load"] = time.time() - t0

            # 2. Capture (only the targets have qmaps)
            t1 = time.time()
            capture_H, refs = {}, {}
            single = None
            for c, chunk in enumerate(Batches(len(state), bad_rows, batch).chunks):
                if c == 0 and args.verify_batch and done_layers == 0 and len(chunk) > 1:
                    single = [run_forward(module, state, ids, [i]).float().cpu() for i in chunk]
                rs = run_forward(module, state, ids, chunk, {"capture": capture_H})
                for j, i in enumerate(chunk):
                    if i < NUM_REF_ROWS:
                        r = rs[j : j + 1]
                        if torch.isfinite(r).all().item():
                            refs[i] = r.cpu().clone()
                        else:
                            bad_rows.add(i)
                            print(f" !! Non-finite reference state in calibration row {i}, excluding row")
                if single is not None and c == 0:
                    d = max(float((s - rs[j : j + 1].float().cpu()).norm() / s.norm()) for j, s in enumerate(single))
                    print(f" -- Batch of {len(chunk)} vs one row at a time: max rel l2 {d:.3e}", flush = True)
                    secs["batch_vs_single_rel_l2"] = d
                rs = None
            want_q = {lin.qmap for lin in targets}
            assert set(capture_H) == want_q, f"captured {sorted(capture_H)}, expected {sorted(want_q)}"
            print(f" -- Captured: {module.key}  Hessians: " + ", ".join(sorted(capture_H)), flush = True)
            for k, v in capture_H.items():
                infs, nans = v["inf_nan"][0].item(), v["inf_nan"][1].item()
                if infs or nans:
                    print(f" !! Warning: {k} state has {infs:,} inf values and {nans:,} NaN values")
                v["H_swap_device"] = v["H"].device
                if not exl3_quantize.NO_SWAP:
                    v["H"] = v["H"].cpu()
            secs["capture"] = time.time() - t1

            # 3. Quantize (the converter's path and seeds)
            t1 = time.time()
            if not exl3_quantize.NO_SWAP:
                for lin in targets:
                    lin.inner.swap_cpu()
            qlog.clear()
            cm.quantize_linears_parallel(qargs, targets, config, strategy, idx, [in_args["device"]], None,
                                         capture_H, state, None)
            capture_H = None
            q_tensors = {}
            for lin in targets:
                q_tensors.update(lin.get_tensors())
            q_path = os.path.join(work, "qtensors", f"{module.key}.safetensors")
            cm.save_tensor(q_tensors, os.path.relpath(q_path, work), {"work_dir": work})
            module.unload()
            get_temp_buffers.cache_clear()
            free_mem()
            secs["quantize"] = time.time() - t1

            # 4. The base pack's tensors of the same linears on the reference rows
            rfn_base = None
            if base_stc is not None and refs:
                t1 = time.time()
                base_t = {k: base_stc.get_tensor(k, "cpu", allow_bf16 = True, no_defer = True) for k in hidden
                          if k.rpartition(".")[0] in {lin.key for lin in targets}}
                base_stc.close()
                load_module(module, device, stc, base_t)
                ref_rows = sorted(refs)
                rs = run_forward(module, state, ids, ref_rows).cpu()
                e = [state_error(rs[j : j + 1], refs[i]) for j, i in enumerate(ref_rows)]
                rfn_base = [sum(x[n] for x in e) / len(e) for n in range(3)]
                module.unload()
                del base_t, rs
                free_mem()
                secs["ref_base"] = time.time() - t1

            # 5. Advance with the new tensors (the last target block only needs the reference rows)
            t1 = time.time()
            load_module(module, device, stc, q_tensors)
            for lin in targets:
                assert isinstance(lin.inner, LinearEXL3) and lin.inner.K == strategy[lin.key], f"{lin.key}: reload"
            if li == lb:
                ref_rows = sorted(refs)
                rs = run_forward(module, state, ids, ref_rows).cpu()
                errs = [state_error(rs[j : j + 1], refs[i]) for j, i in enumerate(ref_rows)]
            else:
                errs = advance(module, refs)
            module.unload()
            del q_tensors
            secs["advance"] = time.time() - t1
            rfn = [sum(x[n] for x in errs) / max(len(errs), 1) for n in range(3)]

            malloc_trim()
            secs["total"] = time.time() - t0
            rec = {
                "layer": li, "module_idx": idx, "key": module.key,
                "rfn": rfn[0], "cos": rfn[1], "sqnr": rfn[2],
                "rfn_base": rfn_base[0] if rfn_base else None, "sqnr_base": rfn_base[2] if rfn_base else None,
                "tensors": dict(qlog),
                "seconds": {k: (v if k == "batch_vs_single_rel_l2" else round(v, 2)) for k, v in secs.items()},
                "bad_rows": sorted(bad_rows), "ids_sha256": ids_sha,
            }
            with open(report_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
            bpw = sorted(set(strategy.values()))
            print(
                f" -- Requantized: {module.key}  bpw: {','.join(str(x) for x in bpw)}  rfn: {rfn[0]:.6f}"
                f"  sqnr: {rfn[2]:.3f}" +
                (f"  (base: rfn {rfn_base[0]:.6f} sqnr {rfn_base[2]:.3f})" if rfn_base else "") +
                f"  [{secs['total']:.1f} s: load {secs['load']:.1f}, capture {secs['capture']:.1f}, "
                f"quantize {secs['quantize']:.1f}, " + (f"ref_base {secs['ref_base']:.1f}, " if "ref_base" in secs else "") +
                f"advance {secs['advance']:.1f}]" + cm.host_rss_str(),
                flush = True
            )
            done_layers += 1
            left = sum(1 for x in plan if x > li)
            if left:
                print(f" -- Estimated remaining time: {human_time((time.time() - t_job) / done_layers * left)} "
                      f"({left} target blocks)", flush = True)

        job["next_module_idx"] = idx + 1
        job["bad_rows"] = sorted(bad_rows)
        stop = in_args.get("max_layers") and done_layers >= in_args["max_layers"] and li in plan
        cpi = in_args["checkpoint_interval"]
        if idx + 1 < end_idx and (stop or (cpi >= 0 and time.time() > last_ckpt + cpi)):
            t1 = time.time()
            write_checkpoint(work, job, state, ids)
            last_ckpt = time.time()
            print(f" -- Saved checkpoint: next module {idx + 1}  [{last_ckpt - t1:.1f} s]", flush = True)
        if stop and idx + 1 < end_idx:
            print(f" -- --max_layers {in_args['max_layers']} reached after {module.key} "
                  f"({human_time(time.time() - t_job)}); resume with -r")
            return

    qfiles = [f"{model.modules[first + li].key}.safetensors" for li in sorted(plan)]
    absent = [f for f in qfiles if not os.path.exists(os.path.join(work, "qtensors", f))]
    assert not absent, f"qtensors missing after the run: {absent[:3]}"
    done = {"layers": [la, lb], "blocks": sorted(plan), "next_module_idx": end_idx, "bad_rows": sorted(bad_rows),
            "qtensors": len(qfiles), "ids_sha256": ids_sha, "seconds": round(time.time() - t_job)}
    json.dump(done, open(os.path.join(work, "requant-done.json"), "w"), indent = 4)
    print(f" -- Dense requantization done: {len(qfiles)} blocks ({la}-{lb}) in {human_time(time.time() - t_job)}")


if __name__ == "__main__":
    _args = parser.parse_args()
    # NVIDIA's CuTe DSL re-parses sys.argv at its first kernel compile (see tools/exl3/convert.py)
    sys.argv = sys.argv[:1]
    if os.environ.get("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE") is not None:
        sys.exit("unset TORCH_ALLOW_TF32_CUBLAS_OVERRIDE before requantizing (env -u ...)")
    main(_args)
