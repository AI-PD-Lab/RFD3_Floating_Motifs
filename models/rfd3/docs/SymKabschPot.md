# RFD3 SymKabschPot Features

Three custom inference-time features on top of stock RFD3, available on the
`SymKabschPot` branch of this repo: differentiable
[potentials](#potentials-differentiable-guidance) for steering a design
toward a goal, [floating motif projection](#floating-motif-projection) for
keeping motif fragments geometrically exact via Kabsch alignment, and
[symmetry](#symmetry) (homomeric + hetero pseudo-symmetry, plus
symmetry-aware potentials) for oligomer design. All three are off by default —
a plain `rfd3 design ...` run behaves like stock RFD3.

## Installation

```bash
git clone https://github.com/Jetzu/RFD3_Potentials.git
cd RFD3_Potentials
git checkout SymKabschPot   # base "production" branch has none of this
uv pip install -e '.[all,dev]'   # or pip install "rc-foundry[rfd3]" for a minimal install
foundry install rfd3 --checkpoint-dir <path/to/ckpt/dir>
```

Sanity check:

```bash
cd models/rfd3
rfd3 design out_dir=logs/inference_outs/demo/0 inputs=docs/examples/demo.json skip_existing=False prevalidate_inputs=True
```

Everything below assumes you're running from `models/rfd3/`, and configures
RFD3 by passing `dotted.config.path=value` overrides directly on the command
line (Hydra syntax — same for `rfd3 design` and `run_inference.py`). List
values need to be a quoted, escaped list, e.g.
`"inference_sampler.potentials.guiding_potentials=[\"type:binder_ROG,weight:0.5\"]"`
(see [`examples/run_symmetry_axis_position_phi0.sh`](examples/run_symmetry_axis_position_phi0.sh)
for a full command). You can even pass a whole `inputs=` spec as inline JSON
instead of a file path — [`examples/run_supermotifs_test.sh`](examples/run_supermotifs_test.sh)
does this to drive three different configs out of one JSON file without three
separate invocations.

## Potentials: differentiable guidance

A potential is a scalar function of the current (noisy) coordinates; each
denoising step RFD3 takes its gradient and nudges coordinates to increase it —
a soft steering wheel on the trajectory. Off by default, configured under
`inference_sampler.potentials`:

```yaml
inference_sampler:
  potentials:
    enabled: true
    apply_mode: atom              # token_translation | atom | hybrid
    guide_scale: 0.25
    guide_decay: quadratic
    guide_clip_rms: 0.02
    guiding_potentials:
      - type: binder_ROG
        weight: 0.5
      - type: interface_ncontacts
        weight: 1.0
```

`apply_mode` is `token_translation` (per-residue average, most stable), `atom`
(raw per-atom, more precise but noisier), or `hybrid` (blend of both). `guide_decay`
schedules strength over the course of denoising —
`constant`/`sqrt`/`linear`/`quadratic`/`cubic`/`quartic`/`exponential`/`cosine`,
each with an `inverse_*` counterpart that starts weak and gets *stronger* near
the end of sampling (useful for pinning down an exact end-state rather than
biasing the early fold). `guide_clip_rms` caps gradient RMS before scaling, so
a runaway gradient can't blow up the structure. `guide_start_step` /
`guide_stop_after` restrict guidance to a step window (same idea as the
floating-motif burn-in/stop-after below, but for potentials). Any of
`guide_scale`, `guide_decay`, `guide_clip_rms`, `apply_mode` can be
overridden per potential inside `guiding_potentials`, and specs can also be
written as compact RFdiffusion1-style strings (`"type:binder_ROG,weight:0.5"`).

Potentials don't need a YAML file at all — they're configured entirely at
inference time, so you can turn a potential on/off or sweep a weight per run
just by adding `key=value` overrides to the `rfd3 design` command itself:

```bash
rfd3 design out_dir=logs/inference_outs/potentials_demo/0 inputs=docs/examples/demo.json \
    skip_existing=False prevalidate_inputs=True \
    inference_sampler.potentials.enabled=True \
    inference_sampler.potentials.apply_mode=atom \
    inference_sampler.potentials.guide_scale=0.25 \
    inference_sampler.potentials.guide_decay=quadratic \
    "inference_sampler.potentials.guiding_potentials=[\"type:binder_ROG,weight:0.5\",\"type:interface_ncontacts,weight:1.0\"]"
```

Registered potentials:

- **`binder_ROG`** / **`monomer_ROG`** — minimize radius of gyration (compact
  the binder chain, or the whole design).
- **`interface_ncontacts`** / **`monomer_contacts`** — maximize soft contacts
  (`1/(1+((d-d_0)/r_0)^6)`), between binder and target, or within the design.
- **`atom_pair_distance`** / **`motif_distance`** — harmonic distance
  restraint, on two atoms or on two motif-block centers.
- **`motif_internal_rotation`** — bias one motif block's rigid rotation
  toward a target Euler offset from its input-PDB pose (`angle_x/y/z`);
  `[0,0,0]` reproduces the input orientation.
- **`motif_relative_pose`** (alias `rigid_pose`) — bias the
  origin→motif_i / origin→motif_j / motif_i→motif_j triangle-edge directions
  around the recentered inference origin, via `angle_x/y/z`.
- **`motif_rigid_body_pose`** (alias `rigid_body_pose`) — frame-invariant
  version of the above: builds motif_i's own Kabsch-fit local frame first, so
  it's unaffected by RFD3's per-step global recentering/rotation
  (`pair_weight`, `origin_weight`, `distance_weight`, `origin_atom_filter`).
- **`motif_bridge`** — spreads scaffold atoms evenly between two motif
  centers with a soft cylindrical tube penalty, for designing a linker or
  pocket between them.
- **`motif_bridge2`** — variant of `motif_bridge` with a rounded-end capped
  cylinder instead of a tube (`radius`, `end_padding`, `end_bias` for
  clustering atoms toward the ends rather than spreading them evenly).
- **`motif_rigid`** — keeps a motif's internal geometry close to the input
  PDB via pairwise-distance matching (`pseudo_huber`/`mse`/`l1` loss).
- **`motif_com_distance`** — harmonic restraint on each motif block's
  distance from the protein center of mass (`target_distances`, one entry
  per motif block).
- **`motif_spherical_position`** — constrains each motif's angular position
  (latitude/longitude) on the sphere around the protein COM, ignoring radial
  distance entirely; complements `motif_radial_orientation` below (position
  vs. facing direction).
- **`motif_radial_orientation`** — preserves a motif's rotation relative to
  the direction from the protein's center of mass, invariant to radius,
  angular position, and inter-motif distance.
- **`symmetry_motif_*`** (12 variants) — per-subunit versions of several of
  the above, for oligomers. See [Symmetry-aware potentials](#symmetry-aware-potentials).

Worked example — hold two motif blocks 50 Å apart while keeping one of them
rigid, straight out of [`../README.md`](../README.md):

```yaml
inference_sampler:
  potentials:
    enabled: true
    apply_mode: atom
    guide_scale: 0.2
    guide_decay: inverse_cosine
    guide_clip_rms: 0.03
    guiding_potentials:
      - type: motif_distance
        weight: 50.0
        motif_i: 0
        motif_j: 1
        target_distance: 50.0
      - type: motif_rigid
        weight: 1000.0
        atom_filter: backbone
        k: 0.25
```

## Floating motif projection

Motif coordinates are normally diffused like everything else, with nothing
forcing their internal geometry to stay exact between steps. Floating motif
projection fixes that: after each denoising step, every non-contiguous motif
fragment is rigidly re-fit — via [Kabsch alignment](https://en.wikipedia.org/wiki/Kabsch_algorithm)
— back onto its exact input-PDB coordinates (`kabsch_align_all_atom` in
[`../src/rfd3/model/floating_motif_projection.py`](../src/rfd3/model/floating_motif_projection.py),
verified to recover a random rigid transform to <1e-4 RMSD in
[`../tests/test_floating_motif_projection.py`](../tests/test_floating_motif_projection.py)).
The motif still moves and rotates as a whole with the rest of the design;
only its internal shape snaps back to ground truth.

```bash
rfd3 design ... \
    inference_sampler.floating_motif_project=True \
    inference_sampler.floating_motif_project_every=5 \
    inference_sampler.floating_motif_burn_in=20
```

`floating_motif_project_every` sets the interval, `floating_motif_burn_in`
skips projection for the first N steps (lets the rough fold settle first),
`floating_motif_stop_after` stops it after step N. With potentials also
enabled, step order is: normal sampler update → potential guidance → floating
motif projection.

### New input fields: `motifs`, `non_fixed_contig`, `unindexed_motifs`, `length: "auto"`

These define what actually counts as a floating motif, on top of the plain
`contig` syntax:

- **`motifs`** names a fragment of the input PDB (`{"helix_1": "A10-25"}`) so
  you can reference it elsewhere instead of repeating the raw contig string.
  It only shows up in the design once its name is used — inline in `contig`,
  in `sequence_unrestrained_motifs`, or via a `SymMotif` placeholder — naming
  it alone does nothing.
- **`non_fixed_contig`** is like `contig`, but selected residues are included
  and Kabsch-aligned like a motif *without* being pinned to a fixed 3D
  position. Mutually exclusive with `contig`; can't overlap `unindex`/`motifs`.
- **`unindexed_motifs`** places named motifs on the main chain at a *sampled*
  position instead of an explicit `contig` slot:
  ```json
  {"length": 200, "motifs": {"binderA": "A1-59", "binderB": "B1-65"}, "unindexed_motifs": ["binderA", "binderB"]}
  ```
  RFD3 builds one contiguous 200-residue chain, picks where each motif lands
  before diffusion starts, and both stay floating/Kabsch-aligned throughout.
- **`length: "auto"`** estimates scaffold length from a prolate ellipsoid
  fit around your motif points (`V = 4/3 * pi * a * b * c`, `N = V / 130`,
  range = median ±20%). With the built-in defaults (50 Å point distance, 20 Å
  radius) that's the exact arithmetic RFD3 runs:
  ```
  V = 4/3 * pi * 25 * 20 * 20 = 41,888 Å^3
  N = 41,888 / 130            = 322 residues
  length                      = 258-386
  ```
  If a `motif_distance`/bridge potential is already configured, `auto` reads
  the point distance and `max_radius` from that instead of the defaults. The
  same `auto` token also works inside a contig slot when a top-level `length`
  is given — `"A1-59,auto,B1-65"` with `length: "258-386"` resolves `auto` to
  `134-262` (both bounds minus the 59+65 fixed motif residues).

### Super-motifs

By default each contiguous motif segment is Kabsch-aligned independently. A
**super-motif** groups non-connected fragments (e.g. two loops of a binding
interface) into one rigid body, so the distance and angle *between* them is
preserved too, not just each one's own shape:

```json
{
    "input": "protein.pdb",
    "motifs": {"loop_1": "A1-10", "loop_2": "A25-34"},
    "supermotifs": {"rigid_interface": ["loop_1", "loop_2"]},
    "contig": "loop_1,15,loop_2",
    "length": "40-50"
}
```

(or point `supermotifs` directly at a contig string instead of motif names).
Referenced residues must already appear in `contig`, `motifs`, or
`non_fixed_contig`; super-motif atom sets can't overlap each other; and
`floating_motif_project=True` must be set or nothing gets aligned at all.
Run [`examples/run_supermotifs_test.sh`](examples/run_supermotifs_test.sh)
to compare a super-motif run against an independent-alignment baseline —
16 unit tests in [`../tests/test_supermotifs.py`](../tests/test_supermotifs.py)
cover the grouping/alignment logic directly.

## Symmetry

Two modes, selected via `inference_sampler.kind`.

### Normal symmetry (`kind=symmetry`)

Builds a full homomeric oligomer — every subunit identical — by designing one
asymmetric unit and mirroring it through a point group (cyclic `Cn` or
dihedral `Dn`; only these two families are supported). Symmetry frames are
Kabsch-fit directly from an input symmetric-motif PDB rather than assumed to
be ideal geometry, with per-subunit RMSD checks
([`../src/rfd3/inference/symmetry/frames.py`](../src/rfd3/inference/symmetry/frames.py)).

```json
{"uncond_C5": {"length": 100, "is_non_loopy": true, "symmetry": {"id": "C5"}}}
```

```bash
rfd3 design out_dir=logs/inference_outs/symmetry_demo/0 inputs=docs/examples/symmetry.json \
    diffusion_batch_size=1 skip_existing=False prevalidate_inputs=True \
    inference_sampler.kind=symmetry
```

`diffusion_batch_size=1` is recommended (symmetry is memory-hungry); add
`low_memory_mode=True` too if you hit CUDA OOM (slower, but works). For
motif-conditioned symmetric design, add `is_unsym_motif` (comma-separated
contig/ligand names that should stay asymmetric, e.g. a bound DNA strand) and
`is_symmetric_motif` (whether the input motif is already symmetric around the
origin — currently the only supported mode, `true` by default). Worked
examples for symmetric enzyme active sites, ligand-bound motifs, and DNA-bound
C3 oligomers are in [`examples/symmetry.json`](examples/symmetry.json) /
[`examples/symmetry.md`](examples/symmetry.md).

### Hetero pseudo-symmetry (`kind=hetero_symmetry`)

Normal symmetry forces every subunit to be identical — wrong for a
hetero-oligomer whose subunits share overall shape but differ in
sequence/motifs. Hetero pseudo-symmetry uses the same machinery only to
*initialize* the oligomer, then relaxes the constraint: from then on, only
detected oligomer-interface atoms (plus optional nearby "support" atoms) get
pulled toward the symmetric pose; motif atoms and their neighbors are never
touched, regardless of settings
([`../src/rfd3/inference/symmetry/hetero_pseudo.py`](../src/rfd3/inference/symmetry/hetero_pseudo.py)).

```yaml
inference_sampler:
  kind: hetero_symmetry
  hetero_post_init_symmetry: interface_only   # or "initialization_only" (symmetrize once, then nothing)
  hetero_projection_weight: 1.0
  hetero_interface_distance_cutoff: 8.0
```

If floating motif projection is also on, `hetero_require_per_copy_floating_motifs`
(default `true`) guards against accidentally reusing one motif's reference
geometry across every symmetry copy — set it `false` only if that's actually
what you want.

**Building a hetero-oligomer with a different motif per copy.** `kind=hetero_symmetry`
only controls *sampling*; you still need to tell RFD3 which named motif each
symmetry copy actually gets. Set `symmetry.mode: heterotypic` and
`symmetry.instances` to map copy index → motif name(s), place a `SymMotif`
placeholder in `contig` where that per-copy motif goes, and set
`is_symmetric_motif: false` (the copies are related by shape, not by an
already-symmetric input motif):

```json
{
    "hetero_C2_receptor": {
        "input": "receptor_dimer.pdb",
        "motifs": {"motif_fgfr": "A1-59", "motif_her2": "B1-65"},
        "symmetry": {
            "id": "C2",
            "is_symmetric_motif": false,
            "mode": "heterotypic",
            "instances": {"0": ["motif_fgfr"], "1": ["motif_her2"]}
        },
        "contig": "SymMotif,150",
        "length": null
    }
}
```

`SymMotif` resolves to `motif_fgfr` for copy 0 and `motif_her2` for copy 1
(falling back to instance `"0"`'s motif if a copy index is missing). Motif
names assigned this way must not also appear in `sequence_unrestrained_motifs`
or be duplicated elsewhere in `contig`. `independent` mode (fully separate
per-copy contigs) is defined in the schema but reserved for future use —
`heterotypic` is the one that actually runs today.

```bash
rfd3 design out_dir=logs/inference_outs/hetero_demo/0 inputs=<your_hetero_input.json> \
    diffusion_batch_size=1 skip_existing=False prevalidate_inputs=True \
    inference_sampler.kind=hetero_symmetry \
    inference_sampler.hetero_post_init_symmetry=interface_only
```

### Symmetry-aware potentials

The potentials above treat the whole structure as one unit; the
`symmetry_motif_*` family evaluates the same restraint independently per
subunit (falling back to normal behavior if no symmetry metadata is present),
and **sums** rather than averages contributions across subunits by default —
so a C4 oligomer's four copies each count fully instead of diluting into one
oligomer-wide mean:

- **`symmetry_motif_distance`** / **`symmetry_motif_bridge`** /
  **`symmetry_single_motif_bridge`** — per-subunit versions of `motif_distance`/`motif_bridge`.
- **`symmetry_ellipsoid_bridge`** — like `symmetry_single_motif_bridge`, but
  spreads scaffold atoms inside a subunit-shaped ellipsoid (axes anchored at
  the subunit COM: one toward the motif, two toward the left/right
  neighboring subunits) instead of a radius from the motif alone — covers the
  non-motif-facing half of the subunit too.
- **`symmetry_motif_center_distance`** / **`symmetry_motif_com_distance`** —
  distance from each subunit's motif to the symmetry axis, or to the current
  center of mass.
- **`symmetry_motif_radial_position`** / **`symmetry_motif_radial_orientation`**
  (and `_com_` variants) — radial positioning/orientation, per subunit.
- **`symmetry_motif_axis_position`** — constrains each subunit's motif to a
  spherical `(theta, phi)` position around the symmetry axis, expressed in
  that subunit's own local wedge frame so one target applies uniformly across
  all `Cn` copies (`theta` = elevation from the equatorial plane, `phi` =
  azimuth relative to the wedge centerline).
- **`symmetry_motif_inter_instance_distance`** — COM–COM distance restraint
  *between* motifs on neighboring subunits (defaults to ring-neighbor pairs
  only; `target_pairs` lets you hand-specify exact cross-subunit pairs for
  hetero layouts).

Runnable example — pin every subunit's motif to the middle of its wedge in a
C3 design ([`examples/run_symmetry_axis_position_phi0.sh`](examples/run_symmetry_axis_position_phi0.sh)):

```bash
rfd3 design out_dir=<outdir> inputs=<input.json> n_batches=1 diffusion_batch_size=1 \
    skip_existing=False prevalidate_inputs=True \
    inference_sampler.kind=symmetry \
    inference_sampler.potentials.enabled=True \
    inference_sampler.potentials.apply_mode=atom \
    inference_sampler.potentials.guide_scale=0.20 \
    inference_sampler.potentials.guide_decay=inverse_linear \
    inference_sampler.potentials.guide_clip_rms=0.05 \
    "inference_sampler.potentials.guiding_potentials=[\"type:symmetry_motif_axis_position,weight:2.0,target_theta:0.0,target_phi:0.0,motif_i:0,reduction:sum\"]"
```

## Putting it together

All three systems are independent config knobs and compose freely — e.g. a
symmetric oligomer, with floating-motif projection keeping each active site
exact, plus a potential holding neighboring active sites 30 Å apart:

```bash
rfd3 design out_dir=logs/inference_outs/combined_demo/0 inputs=<your_input.json> \
    diffusion_batch_size=1 skip_existing=False prevalidate_inputs=True \
    inference_sampler.kind=symmetry \
    inference_sampler.floating_motif_project=True \
    inference_sampler.floating_motif_project_every=5 \
    inference_sampler.floating_motif_burn_in=10 \
    inference_sampler.potentials.enabled=True \
    inference_sampler.potentials.apply_mode=atom \
    inference_sampler.potentials.guide_scale=0.25 \
    inference_sampler.potentials.guide_decay=inverse_linear \
    "inference_sampler.potentials.guiding_potentials=[\"type:symmetry_motif_inter_instance_distance,weight:3.0,target_distance:30.0\"]"
```

## Reference: every new input-spec field

Everything below is new on this branch (confirmed against `git diff origin/production`),
on top of stock RFD3 fields like `contig`, `unindex`, `select_fixed_atoms`.
All go inside a design entry in your `inputs=` JSON/YAML, alongside `contig`/`length`/etc.

- **`motifs`** (`dict[str, str]`, default `None`) — named floating motif
  definitions, `{"name": "contig_str"}`. A motif is only included if its name
  appears in `contig`, in `sequence_unrestrained_motifs`, or via a `SymMotif`
  placeholder.
- **`non_fixed_contig`** (contig string or dict, default `None`) — contig of
  atoms included in the design but not fixed in 3D space (Kabsch-aligned like
  a motif instead). Mutually exclusive with `contig`; must not overlap
  `unindex` or `motifs`.
- **`sequence_unrestrained_motifs`** (`list[str]`, default `None`) — motif
  names (from `motifs`) to append as separate floating chains when their
  position in the scaffold isn't constrained. Only listed motifs are
  appended; motifs placed inline via `contig` or `SymMotif` must not also
  appear here.
- **`unindexed_motifs`** (`list[str]`, default `None`) — motif names (from
  `motifs`) to include on the main chain at a sampled position instead of an
  explicit `contig` slot. Must not also appear in `contig`,
  `sequence_unrestrained_motifs`, or a `SymMotif` assignment.
- **`supermotifs`** (`dict[str, str | list[str]]`, default `None`) — named
  rigid-body groups of non-connected motif fragments, Kabsch-aligned together
  as one unit during floating motif projection. Value is either a contig
  string or a list of names from `motifs`.
- **`length: "auto"`** (string literal) — resolves to a `"min-max"` range
  from an ellipsoid-volume estimate. Tuning fields (all optional floats):
  `auto_length_default_distance` (default `50.0` Å), `auto_length_default_radius`
  (default `20.0` Å), `auto_length_residue_volume` (default `130.0` Å³/residue),
  `auto_length_range_fraction` (default `0.20`, i.e. ±20%).
- **`auto` contig token** — usable inside a `contig` string (e.g.
  `"A1-59,auto,B1-65"`) when a top-level `length` is set; resolves to the
  remaining length after subtracting fixed motif residues and other scaffold
  ranges.
- **`symmetry.mode`** (`str`, default `None`) — hetero-symmetry mode for the
  `hetero_symmetry` sampler. `"heterotypic"`: each symmetric copy engages a
  different named motif (requires `symmetry.instances`). `"independent"`:
  reserved for future use.
- **`symmetry.instances`** (`dict[str, Any]`, default `None`) — per-instance
  motif assignment for `heterotypic` mode: `{"0": ["motif_a"], "1": ["motif_b"]}`.
- **`SymMotif`** (contig placeholder token) — inside `contig`, resolves to
  the motif assigned to the current symmetry copy via `symmetry.instances`
  (falling back to instance `"0"`'s motif). Requires `symmetry.id`,
  `symmetry.instances`, and `motifs` to all be defined, and
  `symmetry.is_symmetric_motif: false`.

## Reference: every new sampler config flag

All go under `inference_sampler.` (as a CLI override, `inference_sampler.<flag>=<value>`,
or nested under `inference_sampler:` in YAML).

**Sampler kind**
- `kind` (`"default" | "symmetry" | "hetero_symmetry"`, default `"default"`)

**Floating motif projection**
- `floating_motif_project` (`bool`, default `false`) — master on/off switch.
  CLI shorthand: bare `--floating_motif_project` (no `=True` needed).
- `floating_motif_project_every` (`int`, default `1`) — project every N steps.
- `floating_motif_burn_in` (`int`, default `0`) — skip projection for the
  first N steps.
- `floating_motif_stop_after` (`int | null`, default `null`) — stop
  projecting after step N.

**Normal symmetry**
- `full_symmetry_stop_after` (`int | null`, default `null`) — stop full
  homomeric symmetry projection after step N (independent of
  `floating_motif_stop_after`).

**Hetero pseudo-symmetry**
- `hetero_post_init_symmetry` (`"interface_only" | "initialization_only"`,
  default `"interface_only"`)
- `hetero_projection_enabled` (`bool`, default `true`)
- `hetero_projection_hard` (`bool`, default `false`)
- `hetero_projection_weight` (`float`, default `1.0`)
- `hetero_projection_start_step` (`int`, default `0`)
- `hetero_projection_stop_after` (`int | null`, default `null`)
- `hetero_projection_schedule` (`"constant" | "linear_decay"`, default `"constant"`)
- `hetero_interface_distance_cutoff` (`float`, default `8.0` Å)
- `hetero_interface_sequence_buffer` (`int`, default `2`)
- `hetero_interface_include_sidechains` (`bool`, default `true`)
- `hetero_motif_contact_exclusion_enabled` (`bool`, default `true`)
- `hetero_motif_contact_distance_cutoff` (`float`, default `8.0` Å)
- `hetero_motif_contact_sequence_buffer` (`int`, default `1`)
- `hetero_support_enabled` (`bool`, default `true`)
- `hetero_support_distance_cutoff` (`float`, default `12.0` Å)
- `hetero_support_weight` (`float`, default `0.3`)
- `hetero_support_sequence_buffer` (`int`, default `2`)
- `hetero_recenter_enabled` (`bool`, default `true`)
- `hetero_motif_follow_scaffold_frame` (`bool`, default `true`)
- `hetero_debug` (`bool`, default `false`)
- `hetero_diagnostics_interval` (`int`, default `0`)
- `hetero_require_per_copy_floating_motifs` (`bool`, default `true`)
- `hetero_init_floating_motifs_from_reference` (`bool`, default `true`)

**Potentials** — under `inference_sampler.potentials.`
- `enabled` (`bool`, default `false`)
- `guiding_potentials` (`list`, default `[]`)
- `apply_mode` (`"token_translation" | "atom" | "hybrid"`, default `"token_translation"`)
- `atom_guidance_fraction` (`float`, default `0.25`) — used by `hybrid`.
- `guide_scale` (`float`, default `0.25`)
- `guide_decay` (see the 16 schedules listed above, default `"quadratic"`)
- `guide_clip_rms` (`float`, default `0.02`)
- `guide_start_step` (`int`, default `0`) — first step guidance is applied.
- `guide_stop_after` (`int | null`, default `null`) — last step guidance is applied.
- `include_atoms` (`"all" | "real" | "real_heavy" | "backbone" | "CA"`, default `"real_heavy"`)
- `exclude_fixed_atoms` (`bool`, default `true`)
- `exclude_virtual_atoms` (`bool`, default `true`)
- `guide_only_generated` (`bool`, default `true`)
- `debug` (`bool`, default `false`)

## Further reading

Full upstream option reference (contigs, conditioning, all sampler options):
[`input.md`](input.md). Branch-specific notes this doc is based on:
[`../README.md`](../README.md) (potentials) and
[`../src/rfd3/README.md`](../src/rfd3/README.md)
(floating motif / symmetry / hetero-symmetry).
