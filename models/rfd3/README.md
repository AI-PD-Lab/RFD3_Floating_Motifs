# De novo Design of Biomolecular Interactions with RFdiffusion3

RFdiffusion3 (RFD3) is a diffusion method that can design protein structures 
under complex constraints. 

This repository contains both the training and inference code, and
both are described in more detail below. 

<p align="center">
  <img src="docs/.assets/overview.png" alt="All-atom design with RFD3">
</p>

> [!IMPORTANT]
> **Looking for the InputSpecification?** The complete reference for all input fields, CLI arguments, and inference sampler options is in [`docs/input.md`](./docs/input.md) ([external docs](https://rosettacommons.github.io/foundry/models/rfd3/input.html)). This covers everything from contig strings and conditioning options to diffusion sampler parameters like `num_timesteps` and `n_recycle`.

## Getting Started
1. Install RFdiffusion3. 
  If you have already installed all the models and **are not** interested in hydrogen bond conditioning skip [here](#running-inference). <br><br>
  If you have already installed all the models and **are** interested in hydrogen bond conditioning skip [here](#hydrogen-bond-conditioning)
  If you would like to install all of the foundry models (recommended), see the [foundry README](../../README.md) for instructions. <br><br>
  If you would like to install only RFD3: 
    ```bash
    pip install rc-foundry[rfd3]
    ```

2. Download checkpoint to your desired checkpoint location.
    ```bash
    foundry install rfd3 --checkpoint-dir <path/to/ckpt/dir>
    ```
    This sets `FOUNDRY_CHECKPOINT_DIRS` and will in future look for checkpoints in that directory (alongside the default `~/.foundry/checkpoints` location), allowing you to run inference without supplying the checkpoint path. The checkpoint directory is optional, defaulting to `~/.foundry/checkpoints` if unset.

### Hydrogen Bond Conditioning
If you would like to use hydrogen bond conditioning in your designs, 
you need to install [HBPLUS](https://www.ebi.ac.uk/thornton-srv/software/HBPLUS/). This is **not** installed by default:

3. Download HBPLUS from here: https://www.ebi.ac.uk/thornton-srv/software/HBPLUS/download.html (available for free)
4. Follow the installation instruction here: https://www.ebi.ac.uk/thornton-srv/software/HBPLUS/install.html
5. Update `HBPLUS_PATH` in `foundry/.env` file with the path to your `hbplus` executable.

## Running Inference

Below is a quick inference example to run to test that your setup
is working correctly. If you are new to RFdiffusion methods or JSON/YAML structure, we recommend that you follow the [PPI tutorial]( https://rosettacommons.github.io/foundry/models/rfd3/tutorials/ppi_design_tutorial.html) to set up your first calculation.

To run inference (with foundry installed in your environment, or RFD3 & Foundry src in PYTHONPATH):
```bash
rfd3 design out_dir=logs/inference_outs/demo/0 inputs=models/rfd3/docs/examples/demo.json skip_existing=False dump_trajectories=True prevalidate_inputs=True
```
To run RFD3, you only need to provide the input (`inputs`) JSON/YAML file (see the [external documentation for more details](https://rosettacommons.github.io/foundry/models/rfd3/index.html#general)) where you specify your design constraints and the output directory (`out_dir`) where you want to store the files RFD3 generates.

Additional unnecessary (but useful!) options are added to the above command:
- `dump_trajectories`: Dumps trajectory structures, can be useful for debugging your setup or making cool gifs. However, trajectory files are large, thus this setting is False by default.
- `prevalidate_inputs`: Checks that your inputs are valid before running inference. Helpful if your JSON/YAML has a number of different configs you want to debug / double check are valid before loading the checkpoints.
- `skip_existing`: Skips any existing files that would be in the same place and have the same name as the calculation being run. If you are testing your setup multiple times, including this option is important so that you actually run RFdiffusion3. 

### Floating motif projection

This checkout includes an optional inference-time floating rigid motif projection. It is disabled by default. When enabled, motif residues derived from the existing contig mapping are still diffused normally at each denoising step, then each non-contiguous motif segment is independently Kabsch-aligned back to its original all-atom PDB geometry.

```bash
rfd3 design ... inference_sampler.floating_motif_project=True inference_sampler.floating_motif_project_every=5 inference_sampler.floating_motif_burn_in=20
```

`floating_motif_project_every` controls the projection interval, `floating_motif_burn_in` skips projection for the first N denoising steps, and `floating_motif_stop_after` optionally stops projection after a specific step. In this potentials-enabled checkout, the step order is normal sampler update, external potential guidance, then floating motif Kabsch projection. The projection is an inference-time approximation of floating-anchor diffusion; it does not change training or model architecture.

There are various interesting ways you can use RFD3 beyond [Atom14](https://www.biorxiv.org/content/10.1101/2024.08.16.608235v4) design as it's trained on a large array of different tasks.
For example, you can fix sequence and not structure (prediction-type task), fix the backbone and unfix the sequence (MPNN-type inverse folding) or unfix the sidechains only (PLACER/ChemNet-style):

<p align="center">
  <img src="docs/.assets/conditioning.png" alt="Conditioning options for RFD3">
</p>

For full details on how to specify inputs, see the [input specification documentation](./docs/input.md). You can also see `foundry/models/rfd3/configs/inference_engine/rfdiffusion3.yaml` for even more options.

## External Potential Guidance

RFD3 includes an optional differentiable potential-guidance hook for inference. Potentials return scalar values that are maximized by gradient ascent on coordinates after each sampler step. They are disabled by default and are configured under `inference_sampler.potentials`.

### Minimal configuration

```yaml
inference_sampler:
  potentials:
    enabled: true
    apply_mode: atom
    guide_scale: 0.25
    guide_decay: quadratic
    guide_clip_rms: 0.02
    include_atoms: real_heavy
    guiding_potentials:
      - type: motif_rigid
        weight: 2.0
        atom_filter: backbone
      - type: binder_ROG
        weight: 0.5
```

Potential specs can also use RFdiffusion1-style strings:

```yaml
guiding_potentials:
  - "type:motif_rigid,weight:2.0,atom_filter:backbone,k:0.5"
  - "type:interface_ncontacts,weight:1.0,r_0:8.0,d_0:2.0"
```

### Top-level potential options

| Option | Default | Values | Meaning |
| --- | --- | --- | --- |
| `enabled` | `false` | `true`, `false` | Enables the potential adapter. If false, coordinates are unchanged. |
| `guiding_potentials` | `[]` | list of dicts or strings | Potential specifications to parse and sum. |
| `apply_mode` | `token_translation` | `token_translation`, `atom`, `hybrid` | How raw atom gradients are applied. `token_translation` averages guided atom gradients per token, `atom` applies raw atom gradients, and `hybrid` mixes both. |
| `atom_guidance_fraction` | `0.25` | `0.0` to `1.0` | In `hybrid`, fraction of raw atom-level residual gradient to add to token translation. |
| `guide_scale` | `0.25` | float | Multiplier applied after clipping and decay. |
| `guide_decay` | `quadratic` | see below | Schedule for reducing guide scale as noise level `t` decreases from `T` to 0. |
| `guide_clip_rms` | `0.02` | float >= 0 | RMS clip threshold before scaling. `0.0` disables guidance after clipping. Very large values effectively disable clipping. |
| `include_atoms` | `real_heavy` | `all`, `real`, `real_heavy`, `backbone`, `CA` | Atoms that may contribute to potentials and guidance masks. |
| `exclude_fixed_atoms` | `true` | `true`, `false` | Prevent fixed-coordinate atoms from moving. |
| `exclude_virtual_atoms` | `true` | `true`, `false` | Prevent virtual atoms from contributing or moving. |
| `guide_only_generated` | `true` | `true`, `false` | Restrict coordinate updates to generated/diffused atoms. |
| `debug` | `false` | `true`, `false` | Print potential value, gradient RMS, clip factor, and scale each step. |

`guide_scale`, `guide_decay`, `guide_clip_rms`, `apply_mode`, and `atom_guidance_fraction` can also be set on an individual potential. Per-potential values override the top-level defaults for that one potential only. RFD3 computes each potential gradient separately, applies that potential's clip and decay, then sums the final guidance vectors.

Decay schedules use `ratio = clamp(t / T, 0, 1)`. Available `guide_decay` values are:

| `guide_decay` | Scale multiplier |
| --- | --- |
| `constant` | `1` |
| `sqrt` | `sqrt(ratio)` |
| `linear` | `ratio` |
| `quadratic` | `ratio^2` |
| `cubic` | `ratio^3` |
| `quartic` | `ratio^4` |
| `exponential` | `(exp(5 * ratio) - 1) / (exp(5) - 1)` |
| `cosine` | `0.5 - 0.5 * cos(pi * ratio)` |

Each non-constant schedule also has an inverse form: `inverse_sqrt`, `inverse_linear`, `inverse_quadratic`, `inverse_cubic`, `inverse_quartic`, `inverse_exponential`, and `inverse_cosine`. Inverse decays use `1 - decay(ratio)`, so the guide scale starts near zero at high noise and becomes stronger as reverse diffusion progresses.

### Registered potentials

All potentials support `weight` unless noted. The value is a scalar to maximize, so attractive or preserving restraints return negative penalties.

| Type | Variables | What it does |
| --- | --- | --- |
| `binder_ROG` | `weight=1.0` | Minimizes radius of gyration of generated non-fixed binder atoms selected by `binder_atom_mask`. |
| `monomer_ROG` | `weight=1.0` | Minimizes radius of gyration over `potential_atom_mask`. |
| `interface_ncontacts` | `weight=1.0`, `r_0=8.0`, `d_0=2.0` | Maximizes differentiable contacts between generated binder atoms and fixed target atoms. |
| `monomer_contacts` | `weight=1.0`, `r_0=8.0`, `d_0=2.0` | Maximizes differentiable internal contacts among selected potential atoms, using only upper-triangle pairs. |
| `atom_pair_distance` | `weight=1.0`, `atom_i=0`, `atom_j=1`, `target_distance=8.0` | Harmonic distance restraint on two flat atom indices. |
| `motif_distance` | `weight=1.0`, `motif_i=0`, `motif_j=1`, `target_distance=10.0` | Harmonic COM-distance restraint between two contiguous motif-token blocks. Motif blocks are inferred from contig order, and guidance applies one translation to all atoms in each selected motif block. |
| `motif_bridge` | `weight=1.0`, `motif_i=0`, `motif_j=1`, `spread_weight=1.0`, `outside_weight=1.0`, `tube_weight=0.2`, `max_radius=12.0`, `atom_filter=guide`, `include_motif_atoms=false` | Encourages generated non-motif atoms to spread evenly between two motif centers. |
| `motif_rigid` | `weight=1.0`, `k=1.0`, `loss=pseudo_huber`, `group_mode=all`, `atom_filter=potential`, `motif_i=null`, `min_separation=0` | Preserves fixed-sequence and fixed-coordinate motif geometry by matching current motif atom-pair distances to RFD3 reference coordinates. |
| `motif_radial_orientation` | `weight=1.0`, `motif_offsets=[]`, `origin_atom_filter=real`, `eps=1e-6` | Biases each motif block's rigid-body rotation to preserve (or offset) its input-PDB radial orientation relative to the current protein COM. Invariant to motif radius, angular position on the sphere, and inter-motif distances. Requires at least two motif blocks with distinct reference centres. |

`interface_ncontacts` and `monomer_contacts` use the soft contact function `1 / (1 + ((distance - d_0) / r_0)^6)`.

### `motif_bridge` guide

`motif_bridge` complements `motif_distance`. `motif_distance` computes each selected motif center from all real motif atoms and applies the same translation to every atom in that motif block, preserving the noisy internal motif geometry for the subsequent floating motif Kabsch projection. `motif_bridge` instead acts on selected non-motif atoms, usually generated scaffold atoms, and encourages them to occupy the region between the two motif centers.

It projects selected atoms onto the axis from `motif_i` to `motif_j`, sorts those projected positions, and penalizes deviation from an even spacing between 0 and 1. It also penalizes atoms outside the two motif endpoints and, optionally, atoms farther than `max_radius` from the motif-motif axis.

`motif_bridge` variables:

| Variable | Default | Values | Meaning |
| --- | --- | --- | --- |
| `weight` | `1.0` | float | Overall strength of the bridge-shaping restraint. |
| `motif_i`, `motif_j` | `0`, `1` | integers | Contiguous motif-token block indices used as the bridge endpoints. |
| `spread_weight` | `1.0` | float | Strength for evenly spacing selected atoms along the motif-motif axis. |
| `outside_weight` | `1.0` | float | Strength for keeping selected atoms between the two motif centers rather than beyond them. |
| `tube_weight` | `0.2` | float | Strength for keeping selected atoms near the inter-motif region. Set to `0.0` to disable. |
| `max_radius` | `12.0` | float >= 0 | Allowed distance from the motif-motif axis before the tube penalty applies. |
| `atom_filter` | `guide` | `guide`, `potential`, `binder`, `generated`, `real`, `backbone`, `CA`, `all` | Which atoms are spread between motifs. `guide` follows the final movable-atom mask. |
| `include_motif_atoms` | `false` | `true`, `false` | If false, motif atoms are excluded so the potential acts only on non-motif bridge/scaffold atoms. |

### `motif_rigid` guide

`motif_rigid` is intended for conserving the secondary and tertiary structure of motifs defined in the contigs, especially motifs with fixed sequence but diffused coordinates. It uses `ref_pos` for fixed-sequence motif atoms and `motif_pos` for fixed-coordinate motif atoms when available. The default `group_mode=all` compares all motif atoms in one pairwise distance matrix, preserving both within-block geometry and distances between separate motif blocks. Use `group_mode=blocks` to preserve each contiguous motif block independently.

`motif_rigid` variables:

| Variable | Default | Values | Meaning |
| --- | --- | --- | --- |
| `weight` | `1.0` | float | Strength of the motif geometry restraint. |
| `k` | `1.0` | float > 0 | Pseudo-Huber transition scale in Angstroms; smaller values are stricter near the reference. |
| `loss` | `pseudo_huber` | `pseudo_huber`, `mse`, `l1` | Pair-distance loss. `pseudo_huber` is robust but still strict near zero error. `mse` is strongest against outliers. |
| `group_mode` | `all` | `all`, `blocks` | `all` preserves inter-motif tertiary distances; `blocks` preserves each motif block internally. |
| `atom_filter` | `potential` | `potential`, `all`, `backbone`, `CA`, `real` | Potential-specific atom subset. `potential` follows top-level `include_atoms`; `backbone` is usually a good strict-but-stable motif setting. |
| `motif_i` | `null` | integer or null | With `group_mode=blocks`, restrain only one contiguous motif block by index. |
| `min_separation` | `0` | integer >= 0 | Ignore atom pairs whose selected-atom index separation is smaller than this value. |

Example fixed-sequence motif conservation:

```yaml
inference_sampler:
  potentials:
    enabled: true
    apply_mode: atom
    guide_scale: 0.2
    guide_decay: inverse_cosine
    guide_clip_rms: 0.03
    include_atoms: real_heavy
    guiding_potentials:
      - type: motif_distance
        weight: 50.0
        guide_scale: 0.5
        guide_decay: inverse_linear
        guide_clip_rms: 0.05
        motif_i: 0
        motif_j: 1
        target_distance: 50.0
      - type: motif_bridge
        weight: 5.0
        guide_scale: 0.5
        guide_decay: inverse_linear
        guide_clip_rms: 0.05
        motif_i: 0
        motif_j: 1
        spread_weight: 1.0
        outside_weight: 1.0
        tube_weight: 0.2
        max_radius: 12.0
        atom_filter: guide
      - type: motif_rigid
        weight: 1000.0
        guide_scale: 1.0
        guide_decay: inverse_cosine
        guide_clip_rms: 0.05
        atom_filter: backbone
        k: 0.25
        loss: pseudo_huber
        group_mode: all
```

Equivalent string spec:

```yaml
guiding_potentials:
  - "type:motif_distance,weight:50.0,guide_scale:0.5,guide_decay:inverse_linear,guide_clip_rms:0.05,motif_i:0,motif_j:1,target_distance:50.0"
  - "type:motif_bridge,weight:5.0,guide_scale:0.5,guide_decay:inverse_linear,guide_clip_rms:0.05,motif_i:0,motif_j:1,spread_weight:1.0,outside_weight:1.0,tube_weight:0.2,max_radius:12.0,atom_filter:guide"
  - "type:motif_rigid,weight:1000.0,guide_scale:1.0,guide_decay:inverse_cosine,guide_clip_rms:0.05,atom_filter:backbone,k:0.25,loss:pseudo_huber,group_mode:all"
```

### External-reference motif unindexing

`motif_unindexing` is an inference-only SymKabschPot feature for motif guidance
without pre-indexed motif residues. It is disabled by default and does not use
the legacy unindexed-token path. Motif PDB files are loaded as external
Kabsch-reference coordinates only; their atoms are never inserted into the
generated atom array.

The generated structure is initialized as the usual unconditional design for
the requested length. During sampling, the unindexing controller scans
candidate generated CA windows, aligns each window to each external motif by
Kabsch, scores the alignment RMSD, and greedily assigns one non-overlapping
window per motif. Before activation, it applies a small Kabsch-based bias to
the selected generated windows so they become more motif-like. When the mean
assignment RMSD is below `activation_threshold`, the controller converts the
assignments into ordinary `FloatingMotifReference` objects and calls the
existing floating motif Kabsch projector. The pre-activation search is CA-based,
but activated projection expands the matched residues to the configured atom
names when those atoms exist in both the generated residue and the motif
reference. Projection stops after `post_activation_guidance_steps`, or at the
absolute denoising step `post_activation_stop_after` when that field is set.

This path works through the same sampler hook used by monomer, homomeric
symmetry, and hetero-symmetry modes. In symmetry mode, the sampler reapplies
the normal symmetry projection after dynamic motif projection so the final
post-step coordinates remain symmetric.

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

Configuration fields:

| Field | Default | Meaning |
| --- | --- | --- |
| `enabled` | `false` | Enables the external-reference motif unindexing controller. Existing behavior is unchanged when false. |
| `motif_pdbs` | `[]` | PDB files used as external motif references. CA atoms are used for matching; configured atom names are used for projection when available. |
| `target_length` | `null` | Optional bookkeeping field for the intended unconditional monomer length. The current sampler uses the normal input length machinery. |
| `update_frequency` | `1` | Re-run Kabsch window matching every N denoising steps before activation. |
| `loss_weight` | `1.0` | Strength of the pre-activation Kabsch bias. |
| `activation_threshold` | `2.0` | Mean assignment RMSD threshold, in Angstrom, for switching to normal floating motif projection. |
| `post_activation_guidance_steps` | `20` | Number of denoising steps to keep applying the existing floating motif projector after activation. |
| `post_activation_stop_after` | `null` | Absolute denoising step after which post-activation projection stops. This takes precedence over `post_activation_guidance_steps`. |
| `projection_atom_names` | `[N, CA, C, O, CB]` | Atom names to use when expanding activated CA matches to projection references. Missing atoms are skipped. |
| `allow_overlap` | `false` | If false, greedy motif assignment masks already-used generated atoms so multiple motifs cannot use the same region. |
| `debug` | `false` | Emit activation diagnostics through the logger. |

Known limitations:

- Matching and pre-activation biasing are CA-only in this first implementation. Activated projection can use matched backbone/CB atoms. It avoids
  requiring residue identities or atom-name compatibility between the generated
  unconditional chain and the external motif PDB.
- Assignment is greedy in motif order. This is robust and small, but it is not
  a global combinatorial optimizer for many motifs.
- The old input-level `unindex` feature still inserts unindexed motif atoms and
  is separate from this feature. Do not use both as if they were the same mode.

### `motif_radial_orientation` guide

`motif_radial_orientation` biases the *rotational pose* of each contig-defined motif block relative to the direction from the current protein center of mass (COM) to the motif's center. This direction is the motif's *radial direction*. The potential asks: "is the motif rotated the same way around its radial axis as it was in the input PDB?"

**What is preserved.** For each motif block, the potential stores a local reference frame at inference start. The frame is built from the motif's reference backbone coordinates expressed relative to the inward/outward radial axis. During diffusion the same frame-relative pose is reconstructed around the *current* radial direction and compared to the current motif atom positions. The loss is the mean squared deviation of centred atom positions from the reconstructed target.

**What is NOT affected.** The potential is invariant by construction to:
- Radius (distance from COM to motif): the radial direction is normalised and the COM is detached from the gradient graph.
- Angular position on the sphere: the radial direction that builds the target frame is detached, so no gradient pushes the motif to a specific location around the COM.
- Inter-motif distances: each block is scored independently.
- Global translation and rotation: gradient is projected to pure rigid rotation for each motif block.

**Requirement.** At least two motif blocks with distinct reference centre positions are needed. A motif whose reference centre coincides with the global reference centre produces a zero radial vector and is silently skipped.

`motif_radial_orientation` variables:

| Variable | Default | Values | Meaning |
| --- | --- | --- | --- |
| `weight` | `1.0` | float | Overall strength of the orientation restraint. |
| `motif_offsets` | `[]` | list of `[ax, ay, az]` in degrees | Per-motif Euler (ZYX) offsets, indexed in contig order. Missing entries use `[0, 0, 0]`. The input-PDB pose corresponds to `[0, 0, 0]`. Offsets rotate the target inside the motif's local radial frame: axis 0 (x) spins around the radial direction, axes 1–2 (y, z) tilt the motif. |
| `origin_atom_filter` | `real` | `real`, `potential`, `guide`, `motif`, `all` | Atom selection used to compute the current protein COM. `real` (all non-virtual atoms) is usually appropriate. |
| `eps` | `1e-6` | float | Denominator clamp for normalisation and frame construction. |

**YAML-only for `motif_offsets`.** Because `motif_offsets` is a list of lists it cannot be expressed in the flat `"type:X,key:val"` string format. Use the YAML dict form:

```yaml
inference_sampler:
  potentials:
    enabled: true
    apply_mode: atom
    guide_scale: 0.15
    guide_decay: inverse_cosine
    guide_clip_rms: 0.03
    include_atoms: real_heavy
    guiding_potentials:
      # Preserve the input-PDB radial orientation for all motif blocks
      - type: motif_radial_orientation
        weight: 5.0
        guide_scale: 0.2
        guide_decay: inverse_linear
        guide_clip_rms: 0.05
        origin_atom_filter: real
        motif_offsets: []          # empty list = reproduce input-PDB orientation

      # Same, but spin motif block 0 by 90° around the radial axis and
      # tilt motif block 1 by 45° around the first tangent axis
      - type: motif_radial_orientation
        weight: 5.0
        origin_atom_filter: real
        motif_offsets:
          - [90.0, 0.0, 0.0]       # block 0: 90° spin around radial direction
          - [0.0, 45.0, 0.0]       # block 1: 45° tilt around tangent axis 1
```

`motif_radial_orientation` is designed to complement `motif_rigid` (which preserves internal motif geometry) and `motif_distance` / `motif_bridge` (which control inter-motif spacing). A typical multi-motif scaffold design might use all three together:

```yaml
guiding_potentials:
  - type: motif_distance
    weight: 30.0
    guide_decay: inverse_linear
    motif_i: 0
    motif_j: 1
    target_distance: 40.0
  - type: motif_radial_orientation
    weight: 3.0
    guide_decay: inverse_cosine
    origin_atom_filter: real
  - type: motif_rigid
    weight: 500.0
    guide_decay: inverse_cosine
    atom_filter: backbone
    k: 0.5
```

## Further example JSONs for different applications
Additional examples are broken up by use case. If you have cloned the
repository, matching `.json` files are in `foundry/models/rfd3/docs/examples`
that can be run directly, similar to the previous example. 

In the examples, the paths to the input files are specified assuming
that you are running the examples from the `foundry/models/rfd3/docs/examples`
directory. If you would like to run RFD3 from a different location, 
you will need to change the path in the `.json` file(s) before running.

<table>
  <tr>
    <td align="center">
      <h3><a href="./docs/na_binder_design.md">Nucleic acid binder design</a></h3>
      <img src="docs/.assets/dna.png" height="150" />
    </td>
    <td align="center">
      <h3><a href="./docs/sm_binder_design.md">Small molecule binder design</a></h3>
      <img src="docs/.assets/sm.png" height="150" />
    </td>
    <td align="center">
      <h3><a href="./docs/protein_binder_design.md">Protein binder design</a></h3>
      <img src="docs/.assets/ppi.png" height="150" />
    </td>
  </tr>
  <tr>
    <td align="center">
      <h3><a href="./docs/enzyme_design.md">Enzyme design</a></h3>
      <img src="docs/.assets/enzyme.png" height="150" />
    </td>
    <td align="center">
      <h3><a href="./docs/symmetry.md">Symmetric design</a></h3>
      <img src="docs/.assets/symm.png" height="150" />
    </td>
  </tr>
</table>

## Training and Fine-Tuning

We make available to the community not only the weights to run RFdiffusion3 but also the complete training code, easily extendable to additional use cases. Any AtomWorks-compatible dataset (and thus, any collection of structure files) can be readily incorporated and used for training or fine-tuning.

### Dataset Configuration

#### PDB Training

To train on the PDB:

1. Set up PDB and CCD mirrors as described in the [AtomWorks documentation](https://rosettacommons.github.io/atomworks/latest/mirrors.html)
2. Update the [path configs](/models/rfd3/configs/paths/) to point to the correct base directories for the metadata parquets
3. Set the `PDB_MIRROR` and `CCD_PATH` variables in your `.env` file

#### Custom Datasets

RFdiffusion3 supports arbitrary datasets of structure files for training and fine-tuning via AtomWorks. See the [AtomWorks dataset documentation](https://rosettacommons.github.io/atomworks/latest/auto_examples/dataset_exploration.html) for details on creating custom datasets.

### Running Training

After setting up Hydra configs, launch a training run:
```bash
uv run python models/rfd3/src/rfd3/train.py experiment=pretrain ckpt_path=<path/to/ckpt>
```

Supplying `ckpt_path=null` (default) will start with fresh weights.
See the [path configs](/models/rfd3/configs/paths/) to customize data input and log output directories.

### Logging Configuration

Training runs support logging via [Weights & Biases](https://wandb.ai/). To enable wandb logging:

```bash
uv run python models/rfd3/src/rfd3/train.py experiment=pretrain logger=wandb
```

To run training without wandb (default):
```bash
uv run python models/rfd3/src/rfd3/train.py experiment=pretrain logger=csv
``` 

### Install HBPLUS for training with hydrogen bond conditioning:

1. Download hbplus from here: https://www.ebi.ac.uk/thornton-srv/software/HBPLUS/download.html (available for free)
2. Follow the installation instruction here: https://www.ebi.ac.uk/thornton-srv/software/HBPLUS/install.html
3. Update `HBPLUS_PATH` in `foundry/.env` file with the path to your `hbplus` executable.

## Distributed Training
To use distributed training, you could use a command such as this (we use Lightning Fabric to handle ddp)
```
EFFECTIVE_BATCH_SIZE=16
DEVICES_PER_NODE= #INSERT NUMBER OF DEVICES PER NODE
NNODES = # INSERT NUMBER OF NODES
GRAD_ACCUM_STEPS=$((EFFECTIVE_BATCH_SIZE / (DEVICES_PER_NODE * NNODES)))
uv run python models/rfd3/src/rfd3/train.py \
    experiment=pretrain \
    trainer.devices_per_node=$DEVICES_PER_NODE \
    trainer.num_nodes=$SLURM_NNODES \
    trainer.grad_accum_steps=$GRAD_ACCUM_STEPS"
```
Notably, fabric must receive `devices_per_node` and the number of nodes (`num_nodes`) you're training on.

**Dataset Paths:** See the paths [configs](/models/rfd3/configs/paths/) to customize the paths where data is read from and where logs are written. There is also a wandb config that can be enabled if you want to log training through wandb. 

**Hydra configs and experiments:** In the example above, the `experiment` argument is a hydra-native argument. For RFD3, it will look for config overrides in `/models/rfd3/configs/experiment/<experiment-name>.yaml` and apply them on top of the base configs

**Conditioning during training:** RFD3 is trained on a multitude of conditioning tasks, and does so by randomly 'creating problems' for it to solve during training. For example, for a random training example it gets a random set of tokens to be 'motif tokens', then subsets those to whether specific atoms should be fixed, and further subsets the information to whether, say, sequence, coordinates or the sequence index should be fixed. It's pretty complicated to evaluate and how it was put together was more of an art than a science. There's likely still room for 
further optimization!

In `models/rfd3/configs/datasets/design_base.yaml` there's the shared configs for all datasets under `global_transform_args`. The dials that control the conditioning described above go under `training_conditions`, where for example `tipatom` - a specific preset conditioning sampler which more frequently fixes few tokens with few atoms - and others can be found.

**Training with WandB:** We strongly recommend tracking your runs via wandb. To use it, simply have your WANDB_API_KEY set and use the wandb logger. For more details see [here](https://wandb.ai/site/)

# Appendix

## Install HBPLUS for hydrogen bond conditioning:
One of the examples shows how to incorporate hydrogen bond conditioning 
into your designs. To make use of this feature, you will need to 
additionally complete the following steps:

1. Download hbplus from here: https://www.ebi.ac.uk/thornton-srv/software/HBPLUS/download.html (available for free)
2. Follow the installation instruction here: https://www.ebi.ac.uk/thornton-srv/software/HBPLUS/install.html
3. Update `HBPLUS_PATH` in `foundry/.env` file with the path to your `hbplus` executable.

## Citation

If you use this code or data in your work, please consider citing:

```bibtex
@article {butcher2025_rfdiffusion3,
	author = {Butcher, Jasper and Krishna, Rohith and Mitra, Raktim and Brent, Rafael Isaac and Li, Yanjing and Corley, Nathaniel and Kim, Paul T and Funk, Jonathan and Mathis, Simon Valentin and Salike, Saman and Muraishi, Aiko and Eisenach, Helen and Thompson, Tuscan Rock and Chen, Jie and Politanska, Yuliya and Sehgal, Enisha and Coventry, Brian and Zhang, Odin and Qiang, Bo and Didi, Kieran and Kazman, Maxwell and DiMaio, Frank and Baker, David},
	title = {De novo Design of All-atom Biomolecular Interactions with RFdiffusion3},
	elocation-id = {2025.09.18.676967},
	year = {2025},
	doi = {10.1101/2025.09.18.676967},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/early/2025/11/19/2025.09.18.676967},
	eprint = {https://www.biorxiv.org/content/early/2025/11/19/2025.09.18.676967.full.pdf},
	journal = {bioRxiv}
}
```
