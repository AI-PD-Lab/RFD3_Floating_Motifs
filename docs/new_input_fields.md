# New Input Fields: `length: auto`, `non_fixed_contig`, `motifs`, `unindexed_motifs`, and Hetero-Symmetry Modes

This document describes input specification fields added to
`DesignInputSpecification` in the `SymKabschPot` fork:

- `non_fixed_contig` — standalone floating design pattern (mutually exclusive with `contig`)
- `motifs` — named, independently floating fragments; can be referenced by name in `contig`
- `unindexed_motifs` — named motifs appended inline on the main chain without explicit contig placement
- `length: "auto"` — potential-aware ellipsoid estimate for scaffold length
- `symmetry.mode` / `symmetry.instances` — hetero-symmetry modes for `hetero_symmetry` sampler

---

## `length: "auto"`

**Type**: literal string token `"auto"` in the existing `length` field

**Semantics**: Resolves to a normal `"min-max"` length range before contig expansion. This keeps
the existing length implementation unchanged after validation.

The estimate models the design volume as an ellipsoid:

```text
semi-major axis = distance_between_points / 2
minor axes      = bridge_radius
volume          = 4/3 * pi * a * b * c
median residues = volume / 130
range           = median +/- 20%
```

Defaults:

- `distance_between_points = 50 A`
- `bridge_radius = 20 A`
- `residue_volume = 130 A^3`

Example:

```text
V = 4/3 * pi * 25 * 20 * 20 = 41888 A^3
N = 41888 / 130 = 322 residues
length = 258-386
```

When `inference_sampler.potentials` are provided, auto length uses potential parameters when
available:

- `motif_distance` / `symmetry_motif_distance`: `target_distance`
- `motif_com_distance` / `symmetry_motif_com_distance` / center-distance potentials:
  `target_distance` or `target_distances`
- bridge potentials: `max_radius`

Missing values fall back to the defaults above.

### Contig-Level `auto`

When a top-level `length` is provided, the contig string may also contain the literal token
`auto`:

```json
{
  "contig": "A1-59,auto,B1-65",
  "length": "auto"
}
```

The contig `auto` slot receives the remaining scaffold length after subtracting:

- all motif / PDB residue spans in the contig
- the maximum size of any other scaffold range in the contig

Example with a fixed total length:

```json
{
  "contig": "A1-59,auto,B1-65",
  "length": 300
}
```

The `auto` slot becomes `176`, because `300 - 59 - 65 = 176`.

Example with a ranged total length:

```json
{
  "contig": "A1-59,auto,B1-65",
  "length": "258-386"
}
```

The `auto` slot becomes `134-262`, because both total-length bounds subtract the same
fixed motif budget: `258 - 59 - 65 = 134` and `386 - 59 - 65 = 262`.
This also applies when `length: "auto"` resolves to a range. If top-level `length` is a
single number, the contig `auto` replacement remains a single number.

---

## `non_fixed_contig`

**Type**: contig string (same format as `contig`, e.g. `"A1-20,100,B5-15"`)

**Semantics**: Defines the full design chain just like `contig`, but all PDB-derived residues
in it are marked as coordinate-unfixed (`is_motif_atom_with_fixed_coord=0`). Unlike regular
`contig` atoms (which are pinned in 3D space), NFC atoms start at the origin and are free to
translate and rotate during diffusion. They are still **Kabsch-aligned** at each diffusion step
(`floating_motif_project=True`), which keeps their internal geometry intact while allowing the
ensemble to settle into a globally consistent conformation.

**Mutually exclusive with `contig`**: combining them would leave the positional relationship
between the two sets of atoms undefined. Use one or the other — not both.

**Requires**: `input` (to select atoms from).

**Fixing specific atoms**: use `select_fixed_atoms` to re-fix specific residues within the
`non_fixed_contig`. This is the intended mechanism for pinning parts of a floating motif.

**When to use**: When you want a fully floating design — e.g. redesigning the geometry of
an existing binder interface, or placing a motif flexibly within a new scaffold — and you
do not need to control where in the chain the motif sits.

### Example JSON

```json
{
  "input": "binder.pdb",
  "non_fixed_contig": "A1-60,150",
  "select_fixed_atoms": "A1-10",
  "infer_ori_strategy": "com",
  "is_non_loopy": true
}
```

Here:
- `A1-60` is floating: Kabsch-aligned but free to find its global position
- 150 scaffold residues are generated from scratch
- `A1-10` are re-fixed via `select_fixed_atoms` (highest priority)

---

## `motifs`

**Type**: `Dict[str, str]` — `{motif_name: contig_string}`

**Semantics**: Named floating fragments. Each key is a unique motif name; each value is a
contig string selecting residues from the input PDB. Atoms default to floating
(`is_motif_atom_with_fixed_coord=0`). They are Kabsch-aligned during diffusion.

**Two usage modes:**

### 1. Referenced by name in `contig` (position-defined, inline)

When a motif name appears as a token in the `contig` string, it is resolved to its residue
selection and inlined at that position in the main design chain. This lets you control whether
the motif sits at the N-terminus, C-terminus, or in the middle of the scaffold.

```json
{
  "input": "binder.pdb",
  "contig": "her2_epitope,150",
  "motifs": { "her2_epitope": "A1-60" },
  "infer_ori_strategy": "com",
  "is_non_loopy": true
}
```

Here `her2_epitope` is placed at the N-terminus (before 150 scaffold residues). The atoms are
floating (from `motifs`) despite appearing in the `contig` string.

Contrast with C-terminal placement:
```json
{ "contig": "150,her2_epitope", "motifs": { "her2_epitope": "A1-60" } }
```

Or mid-chain:
```json
{ "contig": "75,her2_epitope,75", "motifs": { "her2_epitope": "A1-60" } }
```

**If position does not matter**: omit the motif name from `contig` (or use `length` alone).
The motif will be appended as a separate chain (mode 2 below).

### 2. Not referenced in `contig` (position-free, separate chain)

When a motif is NOT mentioned in `contig`, it is appended as an independent separate chain
after the scaffold. Use this with `motif_distance` / `motif_bridge` potentials to pull the
scaffold toward floating epitopes when their relative position is unconstrained.

```json
{
  "input": "FGFR_HER2.pdb",
  "contig": "200",
  "motifs": {
    "fgfr_epitope": "B6-55",
    "her2_epitope": "A1-60"
  },
  "infer_ori_strategy": "com",
  "is_non_loopy": true
}
```

**Requires**: `input` (to select atoms from).

**Priority rule**: Motif atoms are floating by default; `select_fixed_atoms` overrides (highest priority).

**Disjointness**: All `contig` direct PDB tokens and all motif selections must be disjoint at the
atom level. A motif name used as a `contig` token is NOT considered an overlap with the `contig`
mask (since motif-name tokens produce empty selections in the contig's atom mask).

---

## `unindexed_motifs`

**Type**: `List[str]` — motif names from the top-level `motifs` dict

**Semantics**: Motifs listed in `unindexed_motifs` are included on the **main chain** without
needing to appear in `contig`. Their exact placement is not specified by the user; instead, a
hidden inline layout is sampled before diffusion so the model sees one real contiguous chain from
the start.

These motifs:

- remain floating / Kabsch-aligned like other named motifs
- are inserted into the chain before diffusion rather than appended afterward
- do **not** use the active `unindex` guidepost cleanup path
- count toward the total `length` budget

Unlike `sequence_unrestrained_motifs`, they do **not** become separate chains.

### Example JSON

```json
{
  "input": "binders.pdb",
  "length": 150,
  "motifs": { "binder": "A1-59" },
  "unindexed_motifs": ["binder"]
}
```

This builds one main chain of total length 150, with the motif placed inline at a sampled
position in that chain.

So if `binder` is 59 residues long, the sampled scaffold part contributes the remaining 91
residues.

**Constraints**:

- names must exist in `motifs`
- names must not also appear in `contig`
- names must not also appear in `sequence_unrestrained_motifs`
- names assigned through `SymMotif` must not also appear in `unindexed_motifs`

---

## Fixed/Floating Priority Summary

| Source | Result |
|--------|--------|
| Global init (`REQUIRED_CONDITIONING_ANNOTATION_VALUES`) | True for all atoms |
| NFC unfix (`non_fixed_contig` present) | False for NFC atoms |
| Motif unfix (`motifs` present) | False for all motif atoms |
| Default `select_fixed_atoms` (NFC present) | False — nothing fixed |
| Default `select_fixed_atoms` (motifs present) | True for direct contig PDB tokens only |
| Default `select_fixed_atoms` (neither present) | True for all input atoms |
| Explicit `select_fixed_atoms` (user-provided) | Overrides everything |

---

## Hetero-Symmetry Modes

`SymmetryConfig` accepts two additional fields for use with the `hetero_symmetry` sampler:

```json
"symmetry": {
  "id": "C2",
  "mode": "heterotypic",
  "instances": {
    "0": ["fgfr_epitope"],
    "1": ["her2_epitope"]
  }
}
```

### `mode`

- **`"heterotypic"`**: Shared scaffold topology, different motif per copy. Each symmetric copy
  engages a different named motif from the top-level `motifs` dict. All copies start from the
  same pseudo-symmetric scaffold (via `hetero_symmetry` sampler) but each binds a different
  target epitope. Requires `instances`.

- **`"independent"`** (reserved): Each copy gets its own independent contig and scaffold.
  Not yet implemented; accepted and stored for forward compatibility.

### `instances`

For `heterotypic`: `{"copy_index": ["motif_name", ...], ...}`  
For `independent` (reserved): `{"copy_index": {"contig": "..."}, ...}`

> **Note**: Mode orchestration (assigning motifs to copies, setting `sym_transform_id` per copy)
> is **not yet implemented** in `_apply_symmetry`. The fields are accepted and stored in
> `SymmetryConfig` for forward compatibility.

---

## Kabsch Alignment Notes

Both `non_fixed_contig` and `motifs` atoms participate in **floating motif projection**
(Kabsch alignment at each diffusion step), provided:

1. `inference_sampler.floating_motif_project=True`
2. `inference_sampler.floating_motif_stop_after` is set high enough

Kabsch eligibility (`_get_contig_motif_atom_mask`) requires:
- `src_component[0].isalpha()` — the atom came from a real PDB chain ✓
- `is_motif_atom_unindexed=False` ✓ (NFC/motif atoms are NOT unindexed)

Both conditions are guaranteed:
- NFC atoms come from `indexed_tokens` (not `unindexed_tokens`), never marked as unindexed.
- Motif atoms (both inline and separate-chain) go through `accumulate_components` →
  `create_motif_residue`, which sets alphabetic `src_component` from the PDB chain/residue ID.

---

## Shell Script Changes

No shell script changes are required — all new fields are specified in JSON only.
