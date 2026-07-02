#!/bin/bash
# Test script for the supermotifs feature.
# Runs three configs from supermotifs_test.json with floating motif projection enabled:
#   supermotif_by_motif_names  — rigid body defined via motif name list
#   supermotif_by_contig_string — rigid body defined via direct contig string
#   no_supermotif_baseline     — same geometry, independent Kabsch per segment (control)
#
# Usage (from this directory):
#   bash run_supermotifs_test.sh [ckpt_path]
#
# The checkpoint path defaults to rfd3_latest.ckpt.  Set FOUNDRY_CHECKPOINT_DIRS
# or pass it explicitly as the first argument.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
foundry="$(realpath "$SCRIPT_DIR/../../../../")"

export PYTHONPATH="$foundry/src:$foundry/models/rfd3/src/"

ckpt_path="${1:-rfd3_latest.ckpt}"
outdir="$SCRIPT_DIR/supermotifs_test_outputs"
mkdir -p "$outdir"

run_config() {
    local config_key="$1"
    local config_outdir="$outdir/$config_key"
    mkdir -p "$config_outdir"
    echo "=== Running: $config_key ==="
    uv run python "$foundry/models/rfd3/src/rfd3/run_inference.py" \
        ckpt_path="$ckpt_path" \
        out_dir="$config_outdir" \
        "inputs={\"$config_key\": $(python3 -c "import json,sys; d=json.load(open('$SCRIPT_DIR/supermotifs_test.json')); print(json.dumps(d['$config_key']))")}" \
        n_batches=1 \
        diffusion_batch_size=1 \
        skip_existing=False \
        prevalidate_inputs=True \
        "inference_sampler.floating_motif_project=True" \
        "inference_sampler.floating_motif_project_every=5" \
        "inference_sampler.floating_motif_burn_in=10"
    echo "=== Done: $config_key — outputs in $config_outdir ==="
}

run_config supermotif_by_motif_names
run_config supermotif_by_contig_string
run_config no_supermotif_baseline

echo ""
echo "All three configs finished.  Compare outputs in:"
echo "  $outdir/supermotif_by_motif_names"
echo "  $outdir/supermotif_by_contig_string"
echo "  $outdir/no_supermotif_baseline"
echo ""
echo "The two supermotif runs should preserve the relative geometry between loop_1"
echo "and loop_2 (Kabsch-aligned together as one rigid body).  The baseline run"
echo "aligns each loop independently, so inter-loop distances are free to drift."
