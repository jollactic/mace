"""Theory conditioning, categorical metadata and model invariants."""
import ast
import copy
import io
import math
from pathlib import Path
import unittest
import numpy as np
import torch
from ase import Atoms
from e3nn import o3
from mace.data import AtomicData, KeySpecification
from mace.data.utils import config_from_atoms, update_keyspec_from_kwargs
from mace.tools import AtomicNumberTable
from mace.tools.torch_geometric import Batch
from mace.modules.func_emb import FunctionalEmbedding, TheoryFiLM
from mace.modules import interaction_classes
from mace.modules.extensions import PolarMACE
from mace.tools.scripts_utils import extract_config_mace_model

# Reuse the small model fixture without importing network tests/pytest.
fixture_path = Path(__file__).with_name("test_polar_models.py")
fixture_nodes = [n for n in ast.parse(fixture_path.read_text()).body
                 if isinstance(n, ast.FunctionDef) and n.name == "_build_minimal_model"]
exec(compile(ast.Module(body=fixture_nodes, type_ignores=[]), str(fixture_path), "exec"))


class FuncEmbTests(unittest.TestCase):
    def setUp(self):
        self.dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.float64)
        torch.manual_seed(17)
        self.config = {"theories": ["A", "B"], "dim": 4,
                       "sites": ["node_embedding", "products.0", "products.1", "charge_updates.0"],
                       "e0s": {"A": {"H": 0., "O": 0.}, "B": {"H": 1., "O": 2.}}}

    def tearDown(self):
        torch.set_default_dtype(self.dtype)

    def batch(self, names):
        spec = update_keyspec_from_kwargs(KeySpecification.from_defaults(), {"func_emb": self.config})
        graphs = []
        for name in names:
            atoms = Atoms("HO", positions=[[0,0,0],[1.,.2,0]])
            atoms.info.update(level_of_theory=name, total_spin=2, total_charge=0)
            config = config_from_atoms(atoms, key_specification=spec)
            graphs.append(AtomicData.from_config(config, AtomicNumberTable([1,8]), cutoff=4.))
        return Batch.from_data_list(graphs).to_dict()

    def test_identity_e0_mixed_batch_and_roundtrip(self):
        model = _build_minimal_model("cpu", torch.float64)
        inputs = self.batch(["A", "B"])
        reference = model(copy.deepcopy(inputs), compute_force=True)
        model.configure_func_emb(self.config)
        actual = model(copy.deepcopy(inputs), compute_force=True)
        torch.testing.assert_close(actual["energy"]-reference["energy"], torch.tensor([0.,3.]), rtol=0, atol=1e-10)
        for key in ["forces", "charges", "spins", "spin_charge_density"]:
            torch.testing.assert_close(actual[key], reference[key], rtol=1e-10, atol=1e-10)
        stream = io.BytesIO(); torch.save(model, stream); stream.seek(0)
        loaded = torch.load(stream, weights_only=False)
        result = loaded(copy.deepcopy(inputs), compute_force=True)
        torch.testing.assert_close(result["energy"], actual["energy"])
        config = extract_config_mace_model(loaded)
        rebuilt = PolarMACE(**config).double(); rebuilt.load_state_dict(loaded.state_dict())
        torch.testing.assert_close(rebuilt(copy.deepcopy(inputs), compute_force=True)["forces"], actual["forces"])

    def test_equivariance_and_gradient(self):
        irreps = o3.Irreps("2x0e+2x1o+1x2e+1x0o")
        film = TheoryFiLM(irreps, 4)
        with torch.no_grad(): film.projection.weight.normal_(); film.projection.bias.normal_()
        x = torch.randn(3, irreps.dim); embedding = torch.randn(3,4,requires_grad=True)
        rotation = irreps.D_from_matrix(o3.rand_matrix())
        torch.testing.assert_close(film(x @ rotation.T, embedding), film(x, embedding) @ rotation.T, atol=1e-9, rtol=1e-9)
        film(x, embedding).square().sum().backward()
        self.assertGreater(embedding.grad.abs().sum().item(),0)
        model = _build_minimal_model("cpu", torch.float64); model.configure_func_emb(self.config)
        out = model(self.batch(["A", "B"]), compute_force=False)
        out["energy"].square().sum().backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in model.func_emb.adapters.parameters()))

    def test_reject_unknown_metadata_sites_and_missing_e0(self):
        with self.assertRaises(ValueError): self.batch(["unknown"])
        config = copy.deepcopy(self.config); config["sites"] = ["wrong"]
        model = _build_minimal_model("cpu", torch.float64)
        with self.assertRaises(ValueError): model.configure_func_emb(config)
        config = copy.deepcopy(self.config); del config["e0s"]["B"]["O"]
        with self.assertRaises(ValueError): model.configure_func_emb(config)
        spec = update_keyspec_from_kwargs(KeySpecification(), {"func_emb": self.config})
        with self.assertRaises(ValueError): config_from_atoms(Atoms("H"), key_specification=spec)

    def test_periodic_identity(self):
        model = _build_minimal_model("cpu", torch.float64)
        inputs = self.batch(["A"])
        inputs["pbc"].fill_(True)
        inputs["cell"] = torch.eye(3).reshape(3,3)*10
        inputs["rcell"] = torch.eye(3).reshape(3,3)*(2*math.pi/10)
        inputs["volume"] = torch.tensor([1000.])
        reference = model(copy.deepcopy(inputs), compute_force=True)
        model.configure_func_emb(self.config)
        actual = model(copy.deepcopy(inputs), compute_force=True)
        for key in ["energy", "forces", "charges", "spins"]:
            torch.testing.assert_close(actual[key], reference[key], rtol=1e-10, atol=1e-10)

    def test_metrics_and_default_metadata(self):
        from mace.tools.train import MACELoss
        class ZeroLoss(torch.nn.Module):
            def forward(self, ref, pred, ddp=None): return pred["energy"].sum()*0
        batch = Batch(**self.batch(["A", "B"]))
        batch.energy_weight.fill_(1); batch.forces_weight.fill_(1)
        metrics = MACELoss(ZeroLoss(), theory_names=["A", "B"])
        metrics.update(batch, {"energy": torch.tensor([2.,4.]), "forces": torch.ones(4,3)})
        _, values = metrics.compute()
        self.assertEqual(values["theory_metrics"]["A"]["energy_rmse_eV_per_atom"],1.)
        self.assertEqual(values["theory_metrics"]["B"]["energy_rmse_eV_per_atom"],2.)
        self.assertEqual(values["theory_metrics"]["B"]["force_rmse_eV_per_A"],1.)
        config = copy.deepcopy(self.config); config["default_theory"] = "B"
        spec = update_keyspec_from_kwargs(KeySpecification(), {"func_emb": config})
        parsed = config_from_atoms(Atoms("H"), key_specification=spec)
        self.assertEqual(int(parsed.properties["theory_id"][0]),1)

    def test_ase_theory_switch_invalidates_cache(self):
        from mace.calculators import MACECalculator
        model = _build_minimal_model("cpu", torch.float64); model.configure_func_emb(self.config)
        calc = MACECalculator(models=[model], model_type="PolarMACE", device="cpu", default_dtype="float64", pbc_handling="auto")
        atoms = Atoms("HO", positions=[[0,0,0],[1.,.2,0]])
        atoms.info.update(charge=0, spin=2, level_of_theory="A"); atoms.calc = calc
        ea, fa = atoms.get_potential_energy(), atoms.get_forces()
        atoms.info["level_of_theory"] = "B"
        eb, fb = atoms.get_potential_energy(), atoms.get_forces()
        self.assertAlmostEqual(eb-ea,3.,places=9)
        np.testing.assert_allclose(fa,fb,atol=1e-10)
        atoms.info["level_of_theory"] = "not-in-checkpoint"
        with self.assertRaises(ValueError): atoms.get_potential_energy()


if __name__ == "__main__": unittest.main()
