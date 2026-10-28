# Optional scalar atomic spin targets

PolarMACE predicts `spins[i] = alpha[i,0] - beta[i,0]`. This convention is
independent of `magmom` (vector input to magnetic MACE) and `total_spin`
(structure spin multiplicity). The existing global constraint fixes the sum
of predicted scalar spins to multiplicity minus one. Local positive and
negative values are both allowed. No spin-vector direction or conversion
from a population-analysis scheme is inferred.

## Extended XYZ and training

Use a scalar per-atom array, e.g. `Properties=species:S:1:pos:R:3:REF_spin:R:1`.
Each atom line then contains its alpha-minus-beta label after its coordinates.
Keep the usual energy/force columns and charge/multiplicity metadata as needed.
The array must have shape `[number_of_atoms]`; `[number_of_atoms, 3]` vector
moments are not accepted. Independent data conversion/readers are future work.

Add these training options (string values are supported by notebook command builders):

```python
"atomic_spin_key": "REF_spin",
"atomic_spin_weight": "1.0",
```

The default weight is zero: existing energy/force losses are unchanged.
A positive weight wraps the selected energy/force/etc loss with a masked
per-labelled-atom spin MSE. Configuration weights and optional
`config_atomic_spin_weight` multiply the squared residuals. The denominator
counts active labelled atoms, globally across ranks for distributed training.
Zero-weight configurations are excluded from that count. Stage Two retains
the same atomic spin weight. Only PolarMACE supports this supervision.

Missing arrays mask the entire structure. Individual unknown labels may be
NaN, or may be masked using a boolean/0/1 array named `REF_spin_mask` (rename
with `--atomic_spin_mask_key`). False/unknown entries do not train toward zero.
Infinity and malformed shapes are rejected. An all-unlabelled batch contributes
zero spin loss and zero spin gradient. No target sum is silently renormalized;
labels should be compatible with the global multiplicity constraint.

Validation logs and result records include atomic spin RMSE/MAE and label count.
Metrics use active labels without configuration-weight rescaling. No atomic
spin quantities are converted to magnetic units or interpreted as ORCA values.

The scalar targets and masks are carried through Configuration -> AtomicData,
including mixed labelled/unlabelled graph batches. The configuration-based
HDF5 preprocessing path preserves them; the preprocessing parser exposes both
key options. Existing raw AtomicData archives must be regenerated to carry
these new fields before using spin supervision.


## Tolerance band

Set `"atomic_spin_tolerance": "0.2"` in the training options (or pass
`--atomic_spin_tolerance=0.2`). The spin penalty per labelled atom becomes
`max(0, abs(prediction - target) - tolerance)**2`. Both penalty and gradient
are zero inside the band, and the penalty grows quadratically beyond it.
Tolerance must be finite and nonnegative; zero preserves ordinary squared
error. The same value is used in Stage Two. Masking and configuration weights
are unchanged; all active labelled atoms still count in the denominator,
including those inside the tolerance band. Validation RMSE and MAE continue
to measure raw spin errors, not tolerance-adjusted errors.

For a zero target and tolerance 0.2, [-0.2, 0.2] is accepted without spin-loss
pressure. Other losses may still change the prediction. This defines an
acceptable range, not an uncertainty distribution or a modified global spin
constraint. The tolerance is a training loss setting, not an inference option.
