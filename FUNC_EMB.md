# Level-of-theory conditioning (`func_emb` branch)

MACE, ScaleShiftMACE and PolarMACE use one shared energy model, conditioned by a learned embedding for
each level of theory. For PolarMACE, charge-only Coulomb propagation, tied alpha/beta field
weights, atomic-spin targets and tolerance loss are retained. There are no
additional output heads. With no func_emb configuration the existing model
path is unchanged, including loading old spinlock checkpoints.

## Data and configuration

Use a categorical string in each extended-XYZ header:

```
level_of_theory="PBE0_def2-TZVP"
```

One labelled structure-theory pair is one record. Duplicate the geometry for
multiple theories; do not invent missing targets. Keep all copies and related
trajectory/conformer records in the same train/validation/test partition.
Names are case-sensitive categorical labels; there is no interpolation to an
unseen method, inferred relation to basis sets, or implied continuous physics.
Include all electronic-structure settings that define a distinct target method.

Pass `--func_emb=func_emb.yaml` (JSON is also accepted). For example:

```yaml
key: level_of_theory
theories: [PBE0_def2-TZVP, wB97M-V_def2-TZVPD]
reference_theory: PBE0_def2-TZVP
dim: 16
sites: [node_embedding, products.0, products.1, charge_updates.0]
# Optional relative multipliers on configuration loss weights, not sampling:
loss_weights: {PBE0_def2-TZVP: 1.0, wB97M-V_def2-TZVPD: 1.0}
e0s:
  PBE0_def2-TZVP: {H: null, O: null}
  wB97M-V_def2-TZVPD: {H: null, O: null}
```

Replace every `null` with the appropriate isolated atomic energy in **eV**.
Every supported model element requires an entry at every theory; omitted
entries fail rather than falling back to another method. Element symbols or
atomic numbers are accepted and canonicalized. To keep a foundation model's
entire element table, supply E0s for the entire table, not just H/O. Otherwise
use the training code's filtered-element model setup.

Missing and unknown names fail. An optional explicit `default_theory: NAME`
permits untagged records to use that method; unknown names still fail. Ordering
of `theories` fixes numeric IDs and is saved in the checkpoint. Never manually
reorder IDs in preprocessed data. Preprocessing with `--func_emb` encodes IDs
into Configuration properties; train those files with exactly the same config.

## Conditioning sites and initialization

The same learned vector feeds separate equivariant FiLM adapters at selected
sites. `node_embedding` is before message passing; `products.i` is after product
layer i and before its local readout and source map; `charge_updates.i` modifies
only the local feature input to recursion update i. No other sites are inferred.
Unknown names or indices fail. An empty sites list permits E0-only conditioning.

At each site h' = (1 + delta_g(e))*h + delta_b(e). Gates scale an entire irrep
channel uniformly across m; additive biases are restricted to 0e channels.
Adapter final weights/biases start at zero, while embedding rows start distinct.
Thus all adapters are initially identity. The embedding rows receive gradients
after adapter weights move from zero; zero initial embedding gradients alone
are not a failure. Once trained, adapters and embedding rows are unconstrained.

Supplying the same reference E0s and preserving the starting model's other
weights/scales reproduces its energy, forces, charges and spins before fitting.
Changing E0s intentionally changes energies but not forces. The fixed E0 table
is a buffer [theory, element] selected per graph, independent of output heads.
The legacy E0/scaling setup receives the designated reference row; the full
table is used directly in PolarMACE's forward. Pooled dataset normalization is
not automatically an identity transformation when changing training data.

## Training and inference

Use single-head MACE, ScaleShiftMACE or PolarMACE with `--multiheads_finetuning=False`. Existing energy,
force and atomic-spin loss flags continue to apply. All selected adapters and
the embedding table are registered with the optimizer. Theory loss weights
multiply existing configuration weights. Validation logs/results include
per-theory energy RMSE (eV/atom), force RMSE (eV/Angstrom), and label counts;
pooled results alone may hide method-specific failures. No balanced sampler is
introduced. Targets and masks continue to be supplied independently per record.

A first adaptation can initialize from an ordinary spinlock foundation. To
resume/adapt an already conditioned foundation, provide the same func_emb config
on the CLI so the reader encodes names correctly. Changing an existing theory
vocabulary/config currently requires explicit transfer; it is rejected to avoid
silently misassigning embedding rows. Parameter/optimizer checkpoints from the
unconditioned architecture are not interchangeable with conditioned ones.

`MACECalculator(..., model_type="MACE")` for regular MACE, or
`model_type="PolarMACE"` for PolarMACE, restores the config, vocabulary,
adapters and E0s from a normal `.model` checkpoint. Select the theory with:

```python
atoms.info["level_of_theory"] = "PBE0_def2-TZVP"
atoms.calc = calc
energy = atoms.get_potential_energy()
```

Switching the metadata changes the graph ID on the next ASE calculation. For
manual caching workflows call `calc.reset()` when changing the theory. Committee
models must have matching func_emb configurations. Currently the standard
e3nn mul_ir backend and normal ASE execution are supported; CuEq/OEq and LAMMPS
domain decomposition are rejected. TorchScript/compiled deployment is not
validated. No graph_longrange modifications are needed.


## Regular MACE and the alloy example

Regular MACE/ScaleShiftMACE support `node_embedding` and `products.i` sites;
`charge_updates.i` remains specific to PolarMACE and is rejected for regular
models. Their forward paths select the same per-theory E0 table. No charge,
spin or electrostatics modules are introduced into regular models. Identity
conditioning preserves energies, forces and stresses (with matching E0s).

The workspace notebook `TESTING/func_emb_alloy.ipynb` generates periodic
Cu/Ni-labelled FCC cells at several compositions, strains and displacements.
Two custom analytic ASE EAM potentials supply partial-overlap labels. These
are synthetic test functions, not fitted real-material potentials. Isolated
atomic E0s are evaluated explicitly, and an EAM force is checked against an
energy finite difference. A tiny ScaleShiftMACE fits both potentials, reports
held-out parity metrics, plots a volume scan and label coverage, and reloads
its checkpoint through ASE. No external potential-file download is required.
