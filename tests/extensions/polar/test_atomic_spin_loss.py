"""Scalar alpha-minus-beta labels: reader, masks and gradients."""
import unittest
import torch
from ase import Atoms
from mace.data import AtomicData, KeySpecification
from mace.data.utils import config_from_atoms
from mace.tools import AtomicNumberTable
from mace.tools.torch_geometric import Batch
from mace.modules.loss import AtomicSpinLoss


class ZeroLoss(torch.nn.Module):
    def forward(self, ref, pred, ddp=None):
        return pred["spins"].sum()*0


class AtomicSpinTests(unittest.TestCase):
    def graph(self, targets=None, mask=None):
        atoms = Atoms("H2", positions=[[0,0,0],[0,0,1]])
        if targets is not None:
            atoms.new_array("REF_spin", torch.tensor(targets, dtype=torch.float64).numpy())
        if mask is not None:
            atoms.new_array("REF_spin_mask", torch.tensor(mask).numpy())
        config = config_from_atoms(atoms, key_specification=KeySpecification.from_defaults())
        return AtomicData.from_config(config, z_table=AtomicNumberTable([1]), cutoff=3.)

    def test_masked_mixed_batch_and_gradient(self):
        batch = Batch.from_data_list([self.graph([1., float("nan")]), self.graph()])
        prediction = torch.tensor([2., 9., 8., 7.], requires_grad=True)
        loss = AtomicSpinLoss(ZeroLoss(), 2.)(batch, {"spins": prediction})
        self.assertAlmostEqual(loss.item(), 2.)
        loss.backward()
        torch.testing.assert_close(prediction.grad, torch.tensor([4., 0., 0., 0.]))

    def test_explicit_mask_and_no_labels(self):
        graph = self.graph([1., 2.], [True, False])
        self.assertEqual(graph.atomic_spin_mask.tolist(), [True, False])
        batch = Batch.from_data_list([self.graph()])
        prediction = torch.ones(2, requires_grad=True)
        loss = AtomicSpinLoss(ZeroLoss(), 1.)(batch, {"spins": prediction})
        self.assertEqual(loss.item(), 0.)
        loss.backward()
        torch.testing.assert_close(prediction.grad, torch.zeros(2))

    def test_zero_weight_and_invalid_shapes(self):
        batch = Batch.from_data_list([self.graph([1., 0.])])
        prediction = torch.ones(2, requires_grad=True)
        self.assertEqual(AtomicSpinLoss(ZeroLoss(), 0.)(batch, {"spins": prediction}).item(), 0.)
        with self.assertRaises(ValueError):
            AtomicSpinLoss(ZeroLoss(), -1.)
        with self.assertRaises(ValueError):
            self.graph([[1.], [0.]])
        with self.assertRaises(ValueError):
            self.graph([float("inf"), 0.])

    def test_metrics_and_factory_defaults(self):
        from types import SimpleNamespace
        from mace.tools.train import MACELoss
        from mace.tools.scripts_utils import _with_atomic_spin_loss
        base = ZeroLoss()
        self.assertIs(_with_atomic_spin_loss(SimpleNamespace(atomic_spin_weight=0.), base), base)
        with self.assertRaises(ValueError):
            _with_atomic_spin_loss(SimpleNamespace(atomic_spin_weight=1., model="MACE"), base)
        batch = Batch.from_data_list([self.graph([1., float("nan")]), self.graph()])
        metrics = MACELoss(AtomicSpinLoss(base, 1.))
        metrics.update(batch, {"spins": torch.tensor([2., 9., 8., 7.])})
        _, values = metrics.compute()
        self.assertEqual(values["num_atomic_spin_labels"], 1)
        self.assertEqual(values["rmse_atomic_spin"], 1.)
        self.assertEqual(values["mae_atomic_spin"], 1.)

    def test_config_weight(self):
        first = self.graph([1., 2.]); first.atomic_spin_weight.fill_(2.)
        batch = Batch.from_data_list([first])
        loss = AtomicSpinLoss(ZeroLoss(), 1.)(batch, {"spins": torch.tensor([2., 3.])})
        self.assertEqual(loss.item(), 2.)

    def test_tolerance_band_loss_and_gradients(self):
        batch = Batch.from_data_list([self.graph([0., 0.]), self.graph([0., 0.]), self.graph()])
        prediction = torch.tensor([.1, .2, .3, -.5, 8., 9.], dtype=torch.float64, requires_grad=True)
        loss = AtomicSpinLoss(ZeroLoss(), 2., tolerance=.2).double()(batch, {"spins": prediction})
        # Four labelled atoms, including those within the band, in the denominator.
        self.assertAlmostEqual(loss.item(), 2*(.1**2+.3**2)/4, places=8)
        loss.backward()
        torch.testing.assert_close(prediction.grad, torch.tensor([0., 0., .1, -.3, 0., 0.], dtype=torch.float64), atol=1e-8, rtol=1e-6)

    def test_tolerance_zero_matches_squared_error(self):
        batch = Batch.from_data_list([self.graph([1., 0.])])
        prediction = torch.tensor([1.5, -.3], requires_grad=True)
        actual = AtomicSpinLoss(ZeroLoss(), 2., tolerance=0.)(batch, {"spins": prediction})
        expected = 2*(prediction-batch.atomic_spin).square().mean()
        torch.testing.assert_close(actual, expected)
        actual_gradient, = torch.autograd.grad(actual, prediction, retain_graph=True)
        expected_gradient, = torch.autograd.grad(expected, prediction)
        torch.testing.assert_close(actual_gradient, expected_gradient)

    def test_invalid_tolerances_and_factory(self):
        from types import SimpleNamespace
        from mace.tools.scripts_utils import _with_atomic_spin_loss
        for value in [-.1, float("nan"), float("inf")]:
            with self.assertRaises(ValueError):
                AtomicSpinLoss(ZeroLoss(), 1., tolerance=value)
            with self.assertRaises(ValueError):
                _with_atomic_spin_loss(SimpleNamespace(atomic_spin_weight=1., atomic_spin_tolerance=value, model="PolarMACE"), ZeroLoss())
        loss = _with_atomic_spin_loss(SimpleNamespace(atomic_spin_weight=1., atomic_spin_tolerance=.2, model="PolarMACE"), ZeroLoss())
        self.assertAlmostEqual(loss.atomic_spin_tolerance.item(), .2)


if __name__ == "__main__":
    unittest.main()
