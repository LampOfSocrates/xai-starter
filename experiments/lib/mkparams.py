"""Turn `key=value` CLI params into a typed YAML param file for papermill.

papermill's ``-p name value`` coerces everything to a string (so ``SEEDS=[0]``
arrives as the literal "[0]"). Each value here is parsed as a YAML scalar
instead, so ints, floats, bools, and lists keep their real types, and the file
is handed to papermill via ``-f``.

    python experiments/lib/mkparams.py <out.yaml> N_TRAIN=5000 'SEEDS=[0,1,2]' PLM_NAME=facebook/esm2_t6_8M_UR50D
"""
import sys

import yaml

out_path = sys.argv[1]
params = {}
for kv in sys.argv[2:]:
    if not kv.strip():
        continue
    key, _, raw = kv.partition("=")
    try:
        value = yaml.safe_load(raw)        # "5000"->int, "[0,1]"->list, "true"->bool
    except yaml.YAMLError:
        value = raw                         # fall back to the raw string
    params[key] = value

with open(out_path, "w", encoding="utf-8") as fh:
    yaml.safe_dump(params, fh, sort_keys=True, default_flow_style=False)

print(f"[mkparams] {params}")
