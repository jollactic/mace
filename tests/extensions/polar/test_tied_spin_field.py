"""Tied field weights preserve the effective map of the spin-off reference."""
import unittest
import torch
if hasattr(torch.serialization, "add_safe_globals"):
    torch.serialization.add_safe_globals([slice])
from e3nn import o3
from mace.modules.field_blocks import TiedSpinFieldLinear


class TiedSpinFieldTests(unittest.TestCase):
    def test_effective_map_and_spin_cancellation(self):
        for irreps in ["3x0e", "2x0e+2x1o", "1x0e+1x0e+2x1o"]:
            channel = o3.Irreps(irreps)
            original = o3.Linear(channel+channel, o3.Irreps("4x0e+3x1o")).double()
            tied = TiedSpinFieldLinear(original)
            x = torch.randn(5, channel.dim, dtype=torch.float64)
            y = torch.randn_like(x)
            actual = tied(torch.cat([x, y], dim=-1))
            reference = original(torch.cat([.5*(x+y), .5*(x+y)], dim=-1))
            torch.testing.assert_close(actual, reference, rtol=1e-12, atol=1e-12)
            torch.testing.assert_close(tied(torch.cat([x, -x], dim=-1)), torch.zeros_like(actual))
            self.assertEqual(tied.linear.weight.numel()*2, original.weight.numel())

    def test_gradients_tie_channels(self):
        channel = o3.Irreps("2x0e+2x1o")
        tied = TiedSpinFieldLinear(o3.Linear(channel+channel, channel).double())
        inputs = torch.randn(3, 2*channel.dim, dtype=torch.float64, requires_grad=True)
        grad, = torch.autograd.grad(tied(inputs).sum(), inputs)
        torch.testing.assert_close(grad[:, :channel.dim], grad[:, channel.dim:])


if __name__ == "__main__":
    unittest.main()
