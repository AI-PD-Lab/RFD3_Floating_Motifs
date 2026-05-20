# RFD3 Local Changes

## Auto Length

Input specs may set:

```json
{
  "length": "auto"
}
```

Only the literal token `"auto"` activates this path. It is resolved during input
validation to a normal `"min-max"` length range, so all existing contig and
length handling remains unchanged afterward.

The estimate uses a prolate ellipsoid:

```text
semi-major axis = distance_between_points / 2
minor axes      = bridge_radius
volume          = 4/3 * pi * a * b * c
median residues = volume / 130
range           = median +/- 20%
```

Defaults are `distance_between_points=50 A` and `bridge_radius=20 A`. Example:

```text
V = 4/3 * pi * 25 * 20 * 20 = 41888 A^3
N = 41888 / 130 = 322 residues
length = 258-386
```

When potentials are provided through `inference_sampler.potentials`, auto length
tries to infer:

- point distance from `motif_distance` / `symmetry_motif_distance`
- point distance from `motif_com_distance` / `symmetry_motif_com_distance` /
  center-distance targets when pair distances are absent
- bridge radius from `max_radius` on bridge potentials

If a value is missing, the defaults above are used.

The `auto` token can also be used inside a contig when top-level `length` is
provided:

```json
{
  "contig": "A1-59,auto,B1-65",
  "length": "auto"
}
```

The contig `auto` slot receives the remaining scaffold length after subtracting
all motif residues and the maximum size of any other scaffold ranges in the
contig. For example, with `length: 300`, `A1-59,auto,B1-65` resolves the
`auto` slot to `176` residues.

If top-level `length` is a range, either written directly or produced by
`length: "auto"`, the contig `auto` slot becomes a range too:

```json
{
  "contig": "A1-59,auto,B1-65",
  "length": "258-386"
}
```

Here the `auto` slot resolves to `134-262`, because both length bounds subtract
the same fixed motif budget: `258 - 59 - 65 = 134` and
`386 - 59 - 65 = 262`. A single-number top-level `length` still produces a
single-number contig replacement.

## Unindexed Motifs

Added a new input field:

```json
{
  "length": 200,
  "motifs": {
    "binderA": "A1-59",
    "binderB": "B1-65"
  },
  "unindexed_motifs": ["binderA", "binderB"]
}
```

`unindexed_motifs` lets you include named motifs from the `motifs` dict on the
main chain without writing an explicit contig placement for them. The current
active behavior is:

- motifs are inserted into a hidden sampled inline layout before diffusion
- the model sees one real contiguous chain from the start
- motifs remain floating / Kabsch-aligned
- `length` is the total final chain length, including the inserted motifs

This differs from:

- `sequence_unrestrained_motifs`: appends motifs as separate floating chains
- `unindex`: uses the original unindexed guidepost path and trainer-side cleanup

`unindexed_motifs` names must exist in `motifs`, and must not also appear in
`contig`, `sequence_unrestrained_motifs`, or `SymMotif` assignments.

## Hetero Pseudo-Symmetry Inference

Added a new sampler kind:

```yaml
inference_sampler:
  kind: hetero_symmetry
```

This mode still uses the existing symmetry input/initialization path, so the
oligomer is built with the normal RFD3 symmetry machinery before diffusion.  It
then disables full monomer-copy symmetry after initialization.

Supported post-initialization modes:

```yaml
inference_sampler:
  hetero_post_init_symmetry: interface_only      # default
  # or
  hetero_post_init_symmetry: initialization_only
```

`initialization_only` performs no post-init symmetry operation.  Potentials and
floating motif Kabsch projection still run if configured.

`interface_only` builds per-step masks from current coordinates using simple
inter-chain distance contacts, then projects only oligomer-interface atoms and
optionally soft support atoms toward the normal symmetric projection.  Mask
priority is:

1. motif atoms
2. motif-contact exclusion atoms
3. oligomer-interface atoms
4. support atoms
5. free atoms

Motif atoms and motif-contact atoms are never overwritten by hetero projection.

Important config fields:

```yaml
inference_sampler:
  hetero_projection_enabled: true
  hetero_projection_hard: false
  hetero_projection_weight: 1.0
  hetero_projection_start_step: 0
  hetero_projection_stop_after: null
  hetero_projection_schedule: constant          # constant | linear_decay
  hetero_interface_distance_cutoff: 8.0
  hetero_interface_sequence_buffer: 2
  hetero_interface_include_sidechains: true
  hetero_motif_contact_exclusion_enabled: true
  hetero_motif_contact_distance_cutoff: 8.0
  hetero_motif_contact_sequence_buffer: 1
  hetero_support_enabled: true
  hetero_support_distance_cutoff: 12.0
  hetero_support_weight: 0.3
  hetero_support_sequence_buffer: 2
  hetero_debug: false
  hetero_require_per_copy_floating_motifs: true
```

When `floating_motif_project: true`, hetero mode validates that a single
floating motif reference is not silently reused across multiple symmetry copies.
Provide independently defined motif references per copy, or set
`hetero_require_per_copy_floating_motifs: false` to bypass that validation.

Implementation files:

- `model/inference_sampler.py`
- `inference/symmetry/hetero_pseudo.py`
- `utils/inference.py`
- `testing/test_hetero_pseudo_symmetry.py`

## Normal Symmetry Compatibility

Normal symmetry remains available as:

```yaml
inference_sampler:
  kind: symmetry
```

It still performs full homomeric symmetry projection.  The sampler now also
projects full symmetry after potential guidance and before floating motif Kabsch
projection, matching the requested order while preserving the existing
pre-update denoised symmetry projection.

Optional cutoff:

```yaml
inference_sampler:
  full_symmetry_stop_after: null
```

When set, full symmetry projection stops after that denoising step.  This cutoff
is independent from `floating_motif_stop_after`.

## Verification Notes

Syntax checks passed for the changed Python files with `python -m py_compile`.

The focused pytest file could not be collected in the current shell because the
available Python environment does not have `torch` installed:

```text
ModuleNotFoundError: No module named 'torch'
```

## Symmetry-Aware Potentials

Added new potential registry names that evaluate motif restraints per symmetric
subunit instead of treating the full oligomer as one monomer.  These use
`sym_transform_id`/`sym_entity_id` metadata from the inference feature dict and
fall back to the old all-motif behavior when symmetry metadata is absent.

By default, active subunit/motif instances are **summed**, not averaged.  This
means hetero pseudo-symmetry gets one real potential contribution per motif
instance, so different motifs are not diluted into an oligomer-wide mean.  Set
`reduction: mean` explicitly only when you want the old averaged scaling.

Symmetry-aware potentials also expose per-instance guidance masks.  During
guidance, each active motif/subunit instance is masked, reduced, clipped, and
scaled separately before the instance guidance tensors are added together.  This
keeps hetero pseudo-symmetry motifs from sharing one oligomer-wide clipping or
application step.

For hetero pseudo-symmetry motifs that are listed as unsymmetrized motifs, the
motif blocks may not carry a symmetry transform id.  In that case the
symmetry-aware potentials assign those motif blocks to subunits by contig/block
order, so C5 with five independent motif blocks is treated as one motif instance
per subunit.

New potential names:

- `symmetry_motif_distance`
- `symmetry_motif_bridge`
- `symmetry_single_motif_bridge`
- `symmetry_motif_center_distance`
- `symmetry_motif_radial_position`
- `symmetry_motif_radial_orientation`
- `symmetry_motif_com_distance`
- `symmetry_motif_com_radial_position`
- `symmetry_motif_com_radial_orientation`

`symmetry_motif_distance` accepts either the old pair style:

```yaml
{type: symmetry_motif_distance, motif_i: 0, motif_j: 1, target_distance: 10.0}
```

or multiple local motif pairs in one potential:

```yaml
{type: symmetry_motif_distance, motif_pairs: [[0, 1], [1, 2]], target_distances: [10.0, 14.0], reduction: sum}
```

`symmetry_motif_bridge` is the two-motif bridge version.  It runs one bridge
instance per subunit using local motif indices:

```yaml
{type: symmetry_motif_bridge, motif_i: 0, motif_j: 1, max_radius: 12.0, reduction: sum}
```

`symmetry_single_motif_bridge` is for a single local motif.  It distributes the
selected movable atoms in each subunit toward/around that motif, using an even
radial distribution out to `max_radius`:

```yaml
{type: symmetry_single_motif_bridge, motif_i: 0, max_radius: 12.0, reduction: sum}
```

The symmetry-center variants use the origin/axis by default:

```yaml
{type: symmetry_motif_center_distance, target_distance: 20.0, center_type: axis, axis: [0, 0, 1]}
```

The COM variants use the current protein COM instead of the symmetry center:

```yaml
{type: symmetry_motif_com_distance, target_distance: 20.0, origin_atom_filter: real}
```

## External-Reference Motif Unindexing

SymKabschPot also supports an inference-only motif unindexing controller under
`inference_sampler.motif_unindexing`.  This is separate from the legacy
input-level `unindex` path: motif PDBs are used only as external Kabsch
references and motif atoms are not inserted into the generated atom array.

When enabled, sampling starts from the normal unconditional generated structure.
At each pre-activation update, the controller scans generated CA windows, aligns
each candidate to each motif PDB with Kabsch, greedily assigns one
non-overlapping window per motif, and applies a Kabsch-shaped bias to those
generated residues.  Once the mean assigned RMSD is below
`activation_threshold`, the inferred windows are converted into ordinary
`FloatingMotifReference` objects and the existing floating motif projector is
used.  Matching and pre-activation biasing are CA-based, but activated
projection expands each matched residue to configured atom names such as
`N,CA,C,O,CB` when those atoms exist.  Projection stops after
`post_activation_guidance_steps`, or at absolute denoising step
`post_activation_stop_after` when that field is set.

Example:

```yaml
inference_sampler:
  motif_unindexing:
    enabled: true
    motif_pdbs:
      - /path/to/motif_1.pdb
      - /path/to/motif_2.pdb
    target_length: 120
    update_frequency: 1
    loss_weight: 1.0
    activation_threshold: 2.0
    post_activation_stop_after: 160
    projection_atom_names: [N, CA, C, O, CB]
    allow_overlap: false
```

Fields:

- `enabled`: opt-in switch. Existing behavior is unchanged when false.
- `motif_pdbs`: external motif PDB files. The current implementation uses
  ordered CA coordinates from each file.
- `target_length`: optional bookkeeping for the intended unconditional length;
  the normal input length controls actual initialization.
- `update_frequency`: how often to rerun candidate matching before activation.
- `loss_weight`: pre-activation Kabsch bias strength.
- `activation_threshold`: mean assignment RMSD threshold in Angstrom.
- `post_activation_guidance_steps`: number of steps to run normal floating motif
  projection after activation.
- `post_activation_stop_after`: absolute denoising step after which
  post-activation projection stops. Takes precedence over
  `post_activation_guidance_steps`.
- `projection_atom_names`: atom names used to expand activated CA matches to
  projection references. Missing atoms are skipped.
- `allow_overlap`: when false, greedy assignment prevents multiple motifs from
  claiming the same generated atoms.
- `debug`: log activation diagnostics.

This path is used by monomer, homomeric symmetry, and hetero-symmetry samplers.
For symmetry sampling, symmetry projection is reapplied after dynamic motif
projection so the post-step coordinates remain symmetric.  Current limitations:
matching is CA-only, assignment is greedy rather than globally optimized, and
this feature should not be confused with the legacy atom-inserting `unindex`
input mode.
