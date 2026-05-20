#!/usr/bin/env bash
# Try the symmetry_motif_axis_position potential with the phi=0 convention.
#
# Convention used here:
#   target_theta: 0.0   -> same level as the symmetry center, in the equatorial plane
#   target_phi:   0.0   -> middle of each symmetry instance/wedge
#
# Usage, from any directory:
#   bash run_symmetry_axis_position_phi0.sh [ckpt_path]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
foundry="$(realpath "$SCRIPT_DIR/../../../../")"

export PYTHONPATH="$foundry/src:$foundry/models/rfd3/src/"

ckpt_path="${1:-rfd3_latest.ckpt}"
outdir="$SCRIPT_DIR/symmetry_axis_position_phi0_outputs"
input_json="$outdir/symmetry_axis_position_phi0_input.json"

mkdir -p "$outdir"

python3 - "$SCRIPT_DIR/symmetry.json" "$input_json" <<'PY'
import json
import sys
from pathlib import Path

source_path = Path(sys.argv[1])
output_path = Path(sys.argv[2])

with source_path.open() as handle:
    source = json.load(handle)

case = source["unsym_C3_6t8h"]
output_path.parent.mkdir(parents=True, exist_ok=True)
with output_path.open("w") as handle:
    json.dump({"axis_position_phi0_C3": case}, handle, indent=4)
    handle.write("\n")
PY

uv run python "$foundry/models/rfd3/src/rfd3/run_inference.py" \
    ckpt_path="$ckpt_path" \
    out_dir="$outdir" \
    inputs="$input_json" \
    n_batches=1 \
    diffusion_batch_size=1 \
    skip_existing=False \
    prevalidate_inputs=True \
    inference_sampler.kind=symmetry \
    inference_sampler.potentials.enabled=True \
    inference_sampler.potentials.apply_mode=atom \
    inference_sampler.potentials.guide_scale=0.20 \
    inference_sampler.potentials.guide_decay=inverse_linear \
    inference_sampler.potentials.guide_clip_rms=0.05 \
    inference_sampler.potentials.include_atoms=real_heavy \
    "inference_sampler.potentials.guiding_potentials=[\"type:symmetry_motif_axis_position,weight:2.0,target_theta:0.0,target_phi:0.0,motif_i:0,reduction:sum\"]"

echo "Done. Outputs are in:"
echo "  $outdir"
