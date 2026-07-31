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
- `symmetry_motif_axis_position`
- `symmetry_motif_inter_instance_distance`

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

`symmetry_motif_axis_position` constrains the angular position of each subunit's
motif COM in **spherical coordinates around the symmetry center**, expressed in
**each subunit's own local frame** so that a single target specification applies
uniformly across all Cn copies.

Coordinates (both in degrees):

- **theta** — signed elevation from the equatorial plane
  (0° = same level as the symmetry center, + toward the positive symmetry axis,
  - toward the negative symmetry axis).  For the default axis `[0, 0, 1]`,
  positive theta moves upward in global Z and negative theta moves downward.
- **phi** — azimuthal angle in the plane perpendicular to the axis, measured
  relative to the centerline of that subunit's symmetry instance.  Concretely,
  the COM vector is transformed into the subunit's local frame and then shifted
  by half of the instance width, so phi = 0° is always the middle of that
  subunit's wedge.  For C4 this means phi = 0° points at global 45°/135°/225°/315°
  for subunits 0/1/2/3.  Positive phi moves toward the next subunit; negative
  phi moves toward the previous subunit.  Values are compared modulo 360°, so
  `-30`, `330`, and `690` describe the same target direction.

Radial distance is not constrained here; combine with `symmetry_motif_center_distance`
for that.

```yaml
# Cn symmetry: keep all copies level with the symmetry center,
# 30° ahead of each copy's wedge centerline
{type: symmetry_motif_axis_position, weight: 2.0, target_theta: 0.0, target_phi: 30.0,
 weight_theta: 1.0, weight_phi: 0.5}

# Signed theta and phi are allowed:
# 20° above the equatorial plane, 30° behind each copy's wedge centerline
{type: symmetry_motif_axis_position, weight: 2.0, target_theta: 20.0, target_phi: -30.0}

# Hetero: per-subunit overrides as [theta_deg, phi_deg]
{type: symmetry_motif_axis_position, weight: 2.0,
 target_positions: [[0.0, 0.0], [20.0, 45.0], [-20.0, -30.0]]}
```

Parameters:

| Parameter | Default | Meaning |
| --- | --- | --- |
| `target_theta` | `null` | Signed elevation target in degrees. `0` means the equatorial plane at the symmetry center; positive moves toward `axis`, negative moves opposite `axis`. Skipped if null. |
| `target_phi` | `null` | Signed azimuthal angle target in the subunit's local frame, in degrees. Positive moves toward the next subunit; negative moves toward the previous subunit. Values are wrapped modulo 360°. Skipped if null. |
| `target_positions` | `[]` | Per-subunit overrides as `[theta]` or `[theta, phi]` entries. Missing entries fall back to `target_theta` / `target_phi`. One entry works for all Cn copies. |
| `weight_theta` | `1.0` | Relative weight for the signed elevation term. |
| `weight_phi` | `1.0` | Relative weight for the azimuthal term. |
| `center` | `[0, 0, 0]` | Symmetry center in global coordinates. |
| `axis` | `[0, 0, 1]` | Symmetry axis direction (Z for all Cn/Dn). |
| `motif_i` | `0` | Which motif block (0-based) to use from each subunit. |
| `reduction` | `sum` | `sum` or `mean` over active subunit/angle terms. |

---

`symmetry_motif_inter_instance_distance` penalises the pairwise COM–COM distance
between motif blocks across **different** symmetry subunits.  By default it only
scores neighbouring symmetry instances: `(0,1), (1,2), ..., (N-1,0)`.  For C2,
this collapses to the single pair `(0,1)`.  This means every subunit is
constrained to both of its ring neighbours, but each neighbour edge is counted
once: in C4, subunit 0 is constrained to subunits 1 and 3 via pairs `(0,1)` and
`(3,0)`.  Set `neighbor_only: false` only if you really want all N·(N−1)/2
cross-subunit pairs.

For hetero symmetry or hand-specified pair layouts, prefer `target_pairs`: a
list of dictionaries where each entry names the two subunit/motif instances and
the target distance.

```yaml
# Cn symmetry: keep neighbouring motif instances at 30 Å
{type: symmetry_motif_inter_instance_distance, weight: 3.0, target_distance: 30.0}

# Cn symmetry: old all-pairs behaviour, if explicitly wanted
{type: symmetry_motif_inter_instance_distance, weight: 3.0,
 target_distance: 30.0, neighbor_only: false}

# Hetero/dictionary style: define the exact two motif instances and distance
type: symmetry_motif_inter_instance_distance
weight: 3.0
target_pairs:
  - {subunit_i: 0, motif_i: 0, subunit_j: 1, motif_j: 0, target_distance: 25.0}
  - {subunit_i: 0, motif_i: 1, subunit_j: 2, motif_j: 0, target_distance: 40.0}
  - {subunit_i: 1, motif_i: 1, subunit_j: 2, motif_j: 1, target_distance: 30.0}

# Use motif block 1 from each subunit instead of block 0
{type: symmetry_motif_inter_instance_distance, weight: 3.0, target_distance: 25.0,
 motif_i: 1}

# Use block 0 from the first subunit and block 1 from the second (asymmetric)
{type: symmetry_motif_inter_instance_distance, weight: 3.0, target_distance: 20.0,
 motif_i: 0, motif_j: 1}
```

Parameters:

| Parameter | Default | Meaning |
| --- | --- | --- |
| `target_distance` | `20.0` | Shared distance target in Å for default generated pairs. |
| `target_distances` | `[]` | Optional per-default-pair targets. With `neighbor_only: true`, order is `(0,1),(1,2),...,(N-1,0)`; with `neighbor_only: false`, order is lexicographic all-pairs `(0,1),(0,2),...`. Missing pairs fall back to `target_distance`. |
| `target_pairs` | `[]` | Explicit dictionary-style pairs. Each entry can define `subunit_i`, `motif_i`, `subunit_j`, `motif_j`, and `target_distance`. If set, this overrides generated neighbour/all-pair selection. |
| `motif_i` | `0` | Default block index (0-based) to pick from the first subunit of generated/default pairs. |
| `motif_j` | same as `motif_i` | Default block index from the second subunit of generated/default pairs. |
| `neighbor_only` | `true` | If true, generated/default pairs include only neighbouring symmetry instances. If false, generated/default pairs include all cross-subunit pairs. Ignored when `target_pairs` is set. |
| `reduction` | `sum` | `sum` or `mean` over active cross-subunit pairs. |

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
