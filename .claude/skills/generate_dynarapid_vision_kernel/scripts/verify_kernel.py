#!/usr/bin/env python3
"""
Post-Write verification for DynaRapid vision kernel generate.py files.

Usage: verify_kernel.py <path_to_generate_py>

Modes:
  - Smoke test (always):
      Imports the generate.py, calls behavioral(), verifies outputs are
      properly-shaped int32 arrays. Also verifies inputs_info structure.

  - CSim test (opt-in via DYNARAPID_CSIM=1):
      Writes a C++ testbench that bridges int32 ↔ original Vitis types,
      runs Vitis HLS C simulation, compares against behavioral().
      Currently supports single-input/single-output kernels only; falls
      through with a warning for more complex signatures.

Exit codes:
  0 = all checks passed
  2 = a check failed (stderr contains the reason - shown to Claude by the hook)
"""
from __future__ import annotations
import importlib.util
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path


def _fail(msg: str, details: str = "") -> None:
    """Print error to stderr. Exit 0 (advisory) so Claude can continue working.
    The diagnostic is still visible in hook output."""
    print(f"[VERIFY FAIL] {msg}", file=sys.stderr)
    if details:
        print(details, file=sys.stderr)
    sys.exit(0)


def _load_generate(path: Path):
    """Import generate.py as a module and return it. Blocks on import errors."""
    spec = importlib.util.spec_from_file_location("gen", str(path))
    if spec is None or spec.loader is None:
        _fail(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    # Run generate.py in its own working dir so pc.write_json("kernel.json")
    # lands alongside generate.py, not in CWD.
    old_cwd = os.getcwd()
    try:
        os.chdir(path.parent)
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
    except Exception as e:
        _fail(f"{path.name} raised during import/execution", f"{type(e).__name__}: {e}")
    finally:
        os.chdir(old_cwd)
    return mod


def smoke_test(path: Path) -> dict:
    """Validate the generate.py imports, runs, and produces a valid kernel.json."""
    import numpy as np

    mod = _load_generate(path)

    # Required attributes
    for attr in ("behavioral", "inputs_info", "function_signature", "description"):
        if not hasattr(mod, attr):
            _fail(f"{path.name}: missing `{attr}`")

    ii = mod.inputs_info
    if not isinstance(ii, list) or len(ii) < 2:
        _fail(f"{path.name}: inputs_info must be a list with at least 2 entries")

    # Every entry must be int32
    for entry in ii:
        if entry.get("dtype") != "int32":
            _fail(
                f"{path.name}: inputs_info entry `{entry.get('name')}` has "
                f"dtype={entry.get('dtype')!r}; DynaRapid requires 'int32'"
            )

    # Function signature must only use int32_t*
    sig = mod.function_signature
    for bad in ("ap_uint", "ap_int", "uint8_t*", "int8_t*", "uint16_t*",
                "int16_t*", "uint32_t*", "uint64_t*", "int64_t*", "float*", "double*"):
        if bad in sig:
            _fail(f"{path.name}: function_signature contains `{bad}` — DynaRapid is int32_t-only")

    # Function name must end with _int32
    m = re.search(r"void\s+(\w+)\s*\(", sig)
    if not m:
        _fail(f"{path.name}: could not parse function name from signature")
    fn_name = m.group(1)
    if not fn_name.endswith("_int32"):
        _fail(f"{path.name}: function name `{fn_name}` must have `_int32` suffix")

    # Build in-memory test args from inputs_info "in" entries
    in_entries = [e for e in ii if e["iotype"] == "in"]
    out_entries = [e for e in ii if e["iotype"] == "out"]
    if not in_entries or not out_entries:
        _fail(f"{path.name}: need at least one 'in' and one 'out' entry")

    rng = np.random.default_rng(42)
    args = []
    for entry in in_entries:
        data = rng.integers(entry["min"], entry["max"] + 1,
                            size=entry["total_elements"], dtype=np.int32)
        args.append(data)

    try:
        result = mod.behavioral(*args)
    except Exception as e:
        _fail(f"{path.name}: behavioral() raised", f"{type(e).__name__}: {e}")

    # Normalize to tuple
    if not isinstance(result, tuple):
        result = (result,)
    if len(result) != len(out_entries):
        _fail(f"{path.name}: behavioral() returned {len(result)} arrays but inputs_info "
              f"has {len(out_entries)} 'out' entries")

    for arr, entry in zip(result, out_entries):
        if not hasattr(arr, "dtype"):
            _fail(f"{path.name}: behavioral() return value is not a numpy array")
        if arr.dtype != np.int32:
            _fail(f"{path.name}: output `{entry['name']}` has dtype {arr.dtype}, expected int32")
        if arr.size != entry["total_elements"]:
            _fail(f"{path.name}: output `{entry['name']}` has {arr.size} elements, "
                  f"expected {entry['total_elements']}")

    # Verify kernel.json was written alongside generate.py
    kjson = path.parent / "kernel.json"
    if not kjson.exists():
        _fail(f"{path.name}: generate.py did not produce kernel.json in {path.parent}")

    return {
        "function": fn_name,
        "in_entries": in_entries,
        "out_entries": out_entries,
        "outputs": result,
    }


# ------------------------- CSim (opt-in) -------------------------

def _find_vitis_kernel_dir(top_function: str, vitis_libs: Path) -> Path | None:
    """Search L1/examples for a kernel dir containing `void <top_function>(`."""
    examples = vitis_libs / "vision" / "L1" / "examples"
    if not examples.exists():
        return None
    for cpp in examples.glob("*/xf_*_accel.cpp"):
        try:
            text = cpp.read_text()
        except Exception:
            continue
        if re.search(rf"void\s+{re.escape(top_function)}\s*\(", text):
            return cpp.parent
    return None


def _extract_defines(config_h: Path) -> dict:
    """Parse simple `#define KEY VALUE` lines from xf_config_params.h."""
    out = {}
    if not config_h.exists():
        return out
    for line in config_h.read_text().splitlines():
        m = re.match(r"\s*#define\s+(\w+)\s+(.+?)\s*(?://.*)?$", line)
        if m:
            out[m.group(1)] = m.group(2).strip()
    return out


def _parse_vitis_signature(accel_text: str, top_fn: str):
    """
    Parse the Vitis function signature from accel .cpp.
    Returns a list of (param_type, param_name) tuples in declaration order.
    """
    m = re.search(rf"void\s+{re.escape(top_fn)}\s*\((.*?)\)\s*\{{", accel_text, re.DOTALL)
    if not m:
        return None
    args_text = m.group(1)
    # Remove line comments and collapse whitespace
    args_text = re.sub(r"//[^\n]*", "", args_text)
    args_text = re.sub(r"\s+", " ", args_text).strip()
    params = []
    for raw in args_text.split(","):
        raw = raw.strip()
        if not raw:
            continue
        # Pull name off the tail: last identifier, possibly after `*`
        nm = re.match(r"(.+?)(\*?\s*)(\w+)\s*$", raw)
        if not nm:
            continue
        type_part = (nm.group(1) + nm.group(2)).strip()
        name = nm.group(3)
        params.append((type_part, name))
    return params


def _param_role(type_part: str):
    """
    Classify a C/C++ parameter: ('buffer', ctype, ap_type_or_None) or ('scalar', ctype, None).
    ap_type_or_None is e.g. "ap_uint<INPUT_PTR_WIDTH>" for buffer pointers.
    """
    if "*" in type_part:
        base = type_part.replace("*", "").strip()
        # ap_uint<...> buffer
        apm = re.match(r"ap_uint<(.+)>", base)
        if apm:
            return ("buffer", None, base)
        # plain C type buffer (unsigned char, int, etc.)
        return ("buffer", base, None)
    return ("scalar", type_part, None)


def _xf_to_ctype(xf: str):
    return {
        "XF_8UC1": "uint8_t", "XF_8UC3": "uint8_t", "XF_8UC4": "uint8_t",
        "XF_16UC1": "uint16_t", "XF_16UC3": "uint16_t", "XF_16UC4": "uint16_t",
        "XF_16SC1": "int16_t", "XF_16SC3": "int16_t",
        "XF_32SC1": "int32_t", "XF_32UC1": "uint32_t",
    }.get(xf)


def _xf_pixel_bits(xf: str) -> int:
    """Bits per logical pixel for an XF type."""
    return {
        "XF_8UC1": 8, "XF_8UC3": 24, "XF_8UC4": 32,
        "XF_16UC1": 16, "XF_16UC3": 48, "XF_16UC4": 64,
        "XF_16SC1": 16, "XF_16SC3": 48,
        "XF_32SC1": 32, "XF_32UC1": 32,
    }.get(xf, 0)


def _guess_scalar_value(pname: str, ctype: str, gen_mod):
    """
    Guess a scalar arg value for csim:
      1. Look for a module-level constant matching the param name (e.g. THRESH → mod.THRESH)
      2. Fall back to sensible defaults by type
    """
    # Try uppercase variants
    for candidate in (pname.upper(), pname, pname.lower()):
        if hasattr(gen_mod, candidate):
            v = getattr(gen_mod, candidate)
            if isinstance(v, (int, float)):
                return v
    # Common param name mappings
    common = {
        "thresh": 128, "maxval": 255, "threshold": 20, "lower_thresh": 50,
        "upper_thresh": 200, "shift": 0, "direction": 0, "sigma": 1.0,
        "alpha": 0.5, "beta": 0.5, "gamma": 0.0, "rgain": 384, "ggain": 256,
        "bgain": 512, "bformat": 0, "k": 4, "height": "HEIGHT", "width": "WIDTH",
        "rows": "HEIGHT", "cols": "WIDTH",
    }
    v = common.get(pname.lower())
    if v == "HEIGHT" or v == "WIDTH":
        return v  # substituted into tb.cpp as the #define
    if v is not None:
        return v
    # Default by type
    if "float" in ctype or "double" in ctype:
        return 1.0
    return 1


def csim_test(path: Path, smoke_info: dict, vitis_libs: Path, vitis_hls: Path) -> None:
    """
    Run Vitis HLS csim, bridging int32 test data to the original Vitis types.
    Supports multi-buffer and scalar parameters by parsing the Vitis signature.
    """
    import numpy as np

    in_entries = smoke_info["in_entries"]
    out_entries = smoke_info["out_entries"]
    fn_name = smoke_info["function"]
    top_fn_vitis = fn_name[:-len("_int32")]

    kdir = _find_vitis_kernel_dir(top_fn_vitis, vitis_libs)
    if kdir is None:
        _fail(f"{path.name}: could not locate Vitis kernel dir for `{top_fn_vitis}`")
    cfg_h = kdir / "config" / "xf_config_params.h"
    if not cfg_h.exists():
        _fail(f"{path.name}: {cfg_h} not found")
    defines = _extract_defines(cfg_h)
    in_type = defines.get("IN_TYPE", "XF_8UC1")
    out_type = defines.get("OUT_TYPE", in_type)

    accel_cpps = list(kdir.glob("xf_*_accel.cpp"))
    if not accel_cpps:
        _fail(f"{path.name}: no accel .cpp in {kdir}")
    accel_cpp = accel_cpps[0]
    accel_text = accel_cpp.read_text()
    inc_m = re.search(r'#include\s+"(xf_\w+_accel_config\.h)"', accel_text)
    cfg_header = inc_m.group(1) if inc_m else "xf_config_params.h"

    params = _parse_vitis_signature(accel_text, top_fn_vitis)
    if not params:
        _fail(f"{path.name}: could not parse Vitis signature for `{top_fn_vitis}`")

    # Load the generate.py module to read scalar constants
    gen_spec = importlib.util.spec_from_file_location("gen", str(path))
    gen_mod = importlib.util.module_from_spec(gen_spec)
    old_cwd = os.getcwd()
    try:
        os.chdir(path.parent)
        gen_spec.loader.exec_module(gen_mod)  # type: ignore[union-attr]
    finally:
        os.chdir(old_cwd)

    # Classify each Vitis param. Map buffer params to inputs_info entries by NAME
    # when names match (Vitis and DynaRapid typically use the same names but may
    # differ in parameter ORDER). Fall back to positional matching if names don't align.
    ordered_params = []  # [("buffer" | "scalar", pname, ctype, ap_type, iotype_or_value, entry_or_None)]
    buffer_params = []
    in_out_entries = list(in_entries) + list(out_entries)
    by_name = {e["name"]: e for e in in_out_entries}
    used_entries = set()
    vitis_buf_names = [pname for (ptype, pname) in params if "*" in ptype]
    use_name_mapping = all(n in by_name for n in vitis_buf_names) and \
                       len(vitis_buf_names) == len(in_out_entries)
    buf_idx = 0
    for ptype, pname in params:
        role, ctype, ap_type = _param_role(ptype)
        if role == "buffer":
            if use_name_mapping:
                if pname not in by_name:
                    _fail(f"{path.name}: Vitis buffer `{pname}` not found in inputs_info names")
                entry = by_name[pname]
                used_entries.add(pname)
            else:
                if buf_idx >= len(in_out_entries):
                    _fail(f"{path.name}: more buffer params in Vitis signature than in inputs_info")
                entry = in_out_entries[buf_idx]
                buf_idx += 1
            rec = ("buffer", pname, ctype, ap_type, entry["iotype"], entry)
            ordered_params.append(rec)
            buffer_params.append((pname, ctype, ap_type, entry["iotype"], entry))
        else:
            val = _guess_scalar_value(pname, ctype, gen_mod)
            ordered_params.append(("scalar", pname, ctype, None, val, None))
    if use_name_mapping and len(used_entries) != len(in_out_entries):
        _fail(f"{path.name}: not all inputs_info entries matched Vitis buffer names")
    elif not use_name_mapping and buf_idx != len(in_out_entries):
        _fail(f"{path.name}: {buf_idx} Vitis buffers but {len(in_out_entries)} inputs_info entries")

    # If generate.py's HEIGHT/WIDTH differ from the Vitis config, we need to
    # patch xf_config_params.h to match so the Vitis kernel instantiates with
    # the same template dimensions. Otherwise the kernel allocates for the
    # original (possibly huge) image and crashes on our smaller buffers.
    gen_height = getattr(gen_mod, "HEIGHT", None)
    gen_width = getattr(gen_mod, "WIDTH", None)
    vitis_height = int(defines.get("HEIGHT", "0")) if defines.get("HEIGHT", "").isdigit() else None
    vitis_width = int(defines.get("WIDTH", "0")) if defines.get("WIDTH", "").isdigit() else None

    patched_config_dir = None
    kdir_to_use = kdir
    if (gen_height and gen_width and vitis_height and vitis_width and
            (gen_height != vitis_height or gen_width != vitis_width)):
        # Create a patched copy of the kernel dir with HEIGHT/WIDTH overridden.
        import shutil
        patched_kdir = Path(tempfile.mkdtemp(prefix=f"patched_{kdir.name}_"))
        shutil.copytree(kdir, patched_kdir / kdir.name, dirs_exist_ok=True)
        cfg_target = patched_kdir / kdir.name / "config" / "xf_config_params.h"
        if cfg_target.exists():
            txt = cfg_target.read_text()
            txt = re.sub(r"^(\s*#define\s+HEIGHT\s+)\d+", rf"\g<1>{gen_height}", txt, flags=re.M)
            txt = re.sub(r"^(\s*#define\s+WIDTH\s+)\d+", rf"\g<1>{gen_width}", txt, flags=re.M)
            cfg_target.write_text(txt)
        kdir_to_use = patched_kdir / kdir.name

    # Prepare test data
    work = Path(tempfile.mkdtemp(prefix=f"csim_verify_{fn_name}_"))
    rng = np.random.default_rng(42)
    for i, entry in enumerate(in_entries):
        data = rng.integers(entry["min"], entry["max"] + 1,
                            size=entry["total_elements"], dtype=np.int32)
        data.tofile(work / f"input_{i}.bin")
    # expected outputs from behavioral
    for i, arr in enumerate(smoke_info["outputs"]):
        arr.tofile(work / f"expected_{i}.bin")

    # Pre-compute per-buffer packing info.
    # For ap_uint<W> buffer parameters, Vitis packs multiple pixels into each AXI word:
    #   pix_bits  = bits-per-logical-pixel from IN_TYPE/OUT_TYPE
    #   word_bits = W (from ap_uint<W>)
    #   pix_per_word = word_bits / pix_bits
    #   num_words    = total_pixels / pix_per_word
    # For plain pointer buffers (e.g. `unsigned char* kernel`) — no packing, 1 element per element.

    # Map each buffer param to its pixel-type (determines pix_bits)
    def _buf_pix_bits(pname: str, iotype: str, ctype_or_none: str | None, ap_type_or_none: str | None):
        if ap_type_or_none is None:
            # plain buffer (unsigned char*, int*) — treat as one element per word
            return {"uint8_t": 8, "unsigned char": 8, "int16_t": 16, "uint16_t": 16,
                    "int32_t": 32, "int": 32}.get(ctype_or_none, 32)
        # Image buffers: pix bits come from IN_TYPE (inputs) or OUT_TYPE (outputs)
        xf = in_type if iotype == "in" else out_type
        return _xf_pixel_bits(xf) or 8

    def _word_bits_of(ap_type_str: str) -> int:
        """Extract N from 'ap_uint<N>' — may be 'INPUT_PTR_WIDTH' symbolic; resolve from defines."""
        inside = ap_type_str.split("<", 1)[1].rstrip(">")
        inside = inside.strip()
        if inside.isdigit():
            return int(inside)
        # Symbolic: look up in defines dict
        v = defines.get(inside)
        if v and v.isdigit():
            return int(v)
        # Common: INPUT_PTR_WIDTH / OUTPUT_PTR_WIDTH
        return 0

    # Build tb.cpp
    forward_decl_args = ", ".join(f"{ptype} {pname}" for ptype, pname in params)
    buf_decls = []
    buf_loads = []
    buf_writes = []
    buf_compares = []
    out_i = 0
    n_inputs = len(in_entries)
    # Map each inputs_info "in" entry's name to its file index (input_{i}.bin).
    in_name_to_idx = {e["name"]: i for i, e in enumerate(in_entries)}

    # Build call_args in original signature order
    call_args = []
    for rec in ordered_params:
        kind = rec[0]
        if kind == "buffer":
            call_args.append(rec[1])  # pname
        else:
            pname, ctype, _, val, _ = rec[1], rec[2], rec[3], rec[4], rec[5]
            if val == "HEIGHT":
                call_args.append("HEIGHT")
            elif val == "WIDTH":
                call_args.append("WIDTH")
            elif "float" in ctype:
                call_args.append(f"{float(val)}f")
            else:
                call_args.append(str(int(val)))

    for pname, ctype, ap_type, iotype, entry in buffer_params:
        n_pix = entry["total_elements"]
        if ap_type:
            # ap_uint<W>* with pixel packing
            word_bits = _word_bits_of(ap_type)
            pix_bits = _buf_pix_bits(pname, iotype, None, ap_type)
            if word_bits == 0 or pix_bits == 0:
                _fail(f"{path.name}: cannot determine ap_uint word/pixel bits for `{pname}`")
            if word_bits % pix_bits != 0:
                _fail(f"{path.name}: word_bits={word_bits} not divisible by pix_bits={pix_bits}")
            pix_per_word = max(1, word_bits // pix_bits)
            n_words = (n_pix + pix_per_word - 1) // pix_per_word
            decl_type = f"ap_uint<{ap_type.split('<',1)[1].rstrip('>')}>"
            buf_decls.append(f"    static {decl_type} {pname}[{n_words}];   // {pix_per_word} pix/word, {n_pix} pix")
            buf_decls.append(f"    static int32_t {pname}_int32[{n_pix}];")
            if iotype == "in":
                in_idx = in_name_to_idx.get(entry["name"], 0)
                buf_loads.append(f'    f = fopen(argv[{in_idx + 1}], "rb"); fread({pname}_int32, 4, {n_pix}, f); fclose(f);')
                buf_loads.append(f"    for (int w = 0; w < {n_words}; w++) {{")
                buf_loads.append(f"        {decl_type} word = 0;")
                buf_loads.append(f"        for (int k = 0; k < {pix_per_word}; k++) {{")
                buf_loads.append(f"            int idx = w * {pix_per_word} + k;")
                buf_loads.append(f"            if (idx >= {n_pix}) break;")
                buf_loads.append(f"            word.range((k+1)*{pix_bits} - 1, k*{pix_bits}) = ")
                buf_loads.append(f"                ({_xf_to_ctype(in_type) or 'uint32_t'}){pname}_int32[idx];")
                buf_loads.append(f"        }}")
                buf_loads.append(f"        {pname}[w] = word;")
                buf_loads.append(f"    }}")
            if iotype == "out":
                # Unpack after kernel call
                out_ctype = _xf_to_ctype(out_type) or "int32_t"
                buf_writes.append(f"    for (int w = 0; w < {n_words}; w++) {{")
                buf_writes.append(f"        for (int k = 0; k < {pix_per_word}; k++) {{")
                buf_writes.append(f"            int idx = w * {pix_per_word} + k;")
                buf_writes.append(f"            if (idx >= {n_pix}) break;")
                buf_writes.append(f"            uint64_t bits = {pname}[w].range((k+1)*{pix_bits} - 1, k*{pix_bits}).to_uint64();")
                buf_writes.append(f"            {pname}_int32[idx] = (int32_t)({out_ctype})bits;")
                buf_writes.append(f"        }}")
                buf_writes.append(f"    }}")
        else:
            # Plain pointer buffer (e.g. unsigned char* kernel) — 1 element per element, no packing
            buf_decls.append(f"    static {ctype} {pname}[{n_pix}];")
            buf_decls.append(f"    static int32_t {pname}_int32[{n_pix}];")
            if iotype == "in":
                in_idx = in_name_to_idx.get(entry["name"], 0)
                buf_loads.append(f'    f = fopen(argv[{in_idx + 1}], "rb"); fread({pname}_int32, 4, {n_pix}, f); fclose(f);')
                buf_loads.append(f"    for (int i = 0; i < {n_pix}; i++) {pname}[i] = ({ctype}){pname}_int32[i];")
            if iotype == "out":
                buf_writes.append(f"    for (int i = 0; i < {n_pix}; i++) {pname}_int32[i] = (int32_t){pname}[i];")

    # After call: compare output buffers against expected.
    # Match by inputs_info out-entry index (so expected_{i}.bin is read correctly
    # even when Vitis output ordering differs from inputs_info output ordering).
    out_name_to_idx = {e["name"]: i for i, e in enumerate(out_entries)}
    for pname, ctype, ap_type, iotype, entry in buffer_params:
        if iotype != "out":
            continue
        n_pix = entry["total_elements"]
        out_i_local = out_name_to_idx.get(entry["name"], out_i)
        exp_argv_idx = n_inputs + 1 + out_i_local
        buf_compares.append(f"""
    {{
        static int32_t exp_int32[{n_pix}];
        f = fopen(argv[{exp_argv_idx}], "rb"); fread(exp_int32, 4, {n_pix}, f); fclose(f);
        int mism = 0, maxd = 0;
        for (int i = 0; i < {n_pix}; i++) {{
            int d = std::abs({pname}_int32[i] - exp_int32[i]);
            if (d > maxd) maxd = d;
            if (d > 1) {{
                mism++;
                if (mism <= 5) printf("  [{pname}] mismatch %d: got=%d exp=%d diff=%d\\n", i, {pname}_int32[i], exp_int32[i], d);
            }}
        }}
        printf("[{pname}] Max diff: %d, Mismatches (diff>1): %d / {n_pix}\\n", maxd, mism);
        if (mism != 0) total_fail++;
    }}""")
        out_i += 1

    # argv layout: argv[1..N_in] = input files, argv[N_in+1..] = expected files (one per output)
    n_outputs = len(out_entries)
    argc_expected = 1 + n_inputs + n_outputs
    argv_input_files = " ".join(f'"{work}/input_{i}.bin"' for i in range(n_inputs))
    argv_expected_files = " ".join(f'"{work}/expected_{i}.bin"' for i in range(n_outputs))

    tb_cpp = f"""
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include "{cfg_header}"

void {top_fn_vitis}({forward_decl_args});

int main(int argc, char** argv) {{
    if (argc != {argc_expected}) {{ fprintf(stderr, "argc=%d expected={argc_expected}\\n", argc); return 1; }}
    FILE* f;
    int total_fail = 0;
{chr(10).join(buf_decls)}
{chr(10).join(buf_loads)}

    {top_fn_vitis}({", ".join(call_args)});

{chr(10).join(buf_writes)}
{chr(10).join(buf_compares)}

    if (total_fail == 0) {{ printf("PASS\\n"); return 0; }}
    printf("FAIL\\n"); return 1;
}}
"""
    (work / "tb.cpp").write_text(tb_cpp)

    # Build argv: input_0.bin ... input_Nin-1.bin expected_0.bin ... expected_Nout-1.bin
    argv_files = " ".join(
        [f"{work}/input_{i}.bin" for i in range(n_inputs)] +
        [f"{work}/expected_{i}.bin" for i in range(n_outputs)]
    )

    # run_csim.tcl — point at patched kernel dir when dimensions were overridden
    tcl = f"""
set XF_ROOT {vitis_libs}/vision
set TOP     {top_fn_vitis}
set ACCEL_CPP "{kdir_to_use}/{accel_cpp.name}"
set CFG_DIR   "{kdir_to_use}/config"
set INC_DIR   "{kdir_to_use}"
set TB_CPP    "{work}/tb.cpp"
set CFLAGS "-I ${{CFG_DIR}} -I ${{INC_DIR}} -I ${{XF_ROOT}}/L1/include -D__SDSVHLS__ -std=c++14"
open_project -reset "csim_${{TOP}}.prj"
add_files      "${{ACCEL_CPP}}" -cflags "${{CFLAGS}}" -csimflags "${{CFLAGS}}"
add_files -tb  "${{TB_CPP}}"    -cflags "${{CFLAGS}}" -csimflags "${{CFLAGS}}"
set_top ${{TOP}}
open_solution -reset sol1
set_part xcu200-fsgd2104-2-e
create_clock -period 3.3
csim_design -argv "{argv_files}"
exit
"""
    (work / "run_csim.tcl").write_text(tcl)

    vitis_bin = vitis_hls / "bin" / "vitis_hls"
    if not vitis_bin.exists():
        print(f"[VERIFY WARN] {vitis_bin} not found — skipping csim", file=sys.stderr)
        return

    try:
        result = subprocess.run(
            [str(vitis_bin), "run_csim.tcl"],
            cwd=str(work), capture_output=True, text=True, timeout=300,
        )
    except subprocess.TimeoutExpired:
        _fail(f"{path.name}: csim timed out (>300s)")

    stdout = result.stdout + result.stderr
    # "PASS" (testbench) only printed when total_fail == 0. "FAIL" indicates mismatches.
    if re.search(r"^PASS\s*$", stdout, re.M) and "FAIL" not in stdout:
        print(f"[VERIFY CSIM OK] {path.parent.name}", file=sys.stderr)
        return
    _fail(
        f"{path.name}: csim did not pass",
        "\n".join(line for line in stdout.splitlines()
                  if re.search(r"(PASS|FAIL|mismatch|Max diff|Mismatches|error)", line, re.I))
        or stdout[-2000:],
    )


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit(0)  # no path passed, nothing to do
    path = Path(sys.argv[1]).resolve()

    # Only verify dataset/kernels/*_int32/generate.py files
    if path.name != "generate.py":
        sys.exit(0)
    if not re.search(r"/dataset/kernels/\w+_int32/generate\.py$", str(path)):
        sys.exit(0)
    if not path.exists():
        sys.exit(0)

    info = smoke_test(path)
    print(f"[VERIFY SMOKE OK] {path.parent.name}", file=sys.stderr)

    # CSim is mandatory. Resolve Vitis paths from env, then config file.
    vitis_libs = os.environ.get("VITIS_LIBRARIES")
    vitis_hls = os.environ.get("VITIS_HLS")
    if not vitis_libs or not vitis_hls:
        cfg = Path(__file__).resolve().parents[4] / "vision_skill_paths.cfg"
        if cfg.exists():
            for line in cfg.read_text().splitlines():
                line = line.strip()
                if line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                if k == "VITIS_LIBRARIES" and not vitis_libs:
                    vitis_libs = v.strip()
                if k == "VITIS_HLS" and not vitis_hls:
                    vitis_hls = v.strip()
    if not (vitis_libs and vitis_hls):
        _fail(f"{path.name}: VITIS_LIBRARIES / VITIS_HLS could not be resolved; "
              f"csim required but cannot run")
    csim_test(path, info, Path(vitis_libs), Path(vitis_hls))


if __name__ == "__main__":
    main()
