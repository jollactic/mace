# Charge-only tied spin-field branch

This branch always sends alpha+beta multipoles through the existing Coulomb
feature evaluator. Alpha and beta each receive half of the resulting field.
Local spin multipoles, their charge embedding, source updates, and charge energy
remain in the model. No Yukawa or Coulomb spin field is propagated.

The first field embedding has one learned equivariant map applied to the sum of
its alpha/beta inputs. Conversion initializes its effective map to the mean of
the original effective maps, accounting for e3nn instruction path normalization.
For a charge-only reference this preserves the field embedding, energies,
forces, and multipole updates to floating-point accuracy. Other field consumers
still receive the same half-charge fields as in the original spin-off model.

Construction and whole-model loading convert the source-update embedding
idempotently. Old pretrained weights therefore load as charge-only tied models;
they no longer represent their original spin-enabled evaluator on this branch.
The `spin-yukawa` branch retains the original options and baseline comparisons.
`spin_interaction_enabled=True` and positive `spin_yukawa_kappa` are rejected.
The absent/default flag is False. Use the updated model_config extraction and
foundation fine-tuning path to preserve this structure. Raw state dictionaries
from the untied architecture need explicit conversion through a whole-model
checkpoint first; their parameter names and sizes differ.

Only the first source-update field sum is tied. Local multipole embeddings and
other alpha/beta-dependent learned components are not tied. Restoring spin
propagation would require returning to a different branch/implementation.
