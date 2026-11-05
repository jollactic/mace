"""func_emb works without PolarMACE's electrostatic machinery."""
import copy
import io
import unittest
import torch
from ase import Atoms
from mace.modules.models import MACE, ScaleShiftMACE
from mace.modules import interaction_classes
from mace.tools.scripts_utils import extract_config_mace_model
from mace.calculators import MACECalculator
import test_func_emb as fixture
from e3nn import o3


class RegularFuncEmbTests(unittest.TestCase):
    def test_identity_forces_stress_config_and_ase(self):
        old_dtype = torch.get_default_dtype(); torch.set_default_dtype(torch.float64)
        try:
            helper = fixture.FuncEmbTests(); helper.setUp()
            config = copy.deepcopy(helper.config)
            config["sites"] = ["node_embedding", "products.0", "products.1"]
            cls_interaction = interaction_classes["RealAgnosticResidualNonLinearInteractionBlock"]
            kwargs = dict(r_max=4., num_bessel=3, num_polynomial_cutoff=3, max_ell=1,
                interaction_cls=cls_interaction, interaction_cls_first=cls_interaction,
                num_interactions=2, num_elements=2, atomic_numbers=[1,8],
                hidden_irreps=o3.Irreps("4x0e+4x1o"), MLP_irreps=o3.Irreps("8x0e"), atomic_energies=[0.,0.],
                avg_num_neighbors=1., correlation=1, gate=torch.nn.functional.silu, heads=["Default"])
            for model_class in [MACE, ScaleShiftMACE]:
                extra = dict(atomic_inter_scale=2., atomic_inter_shift=.3) if model_class is ScaleShiftMACE else {}
                model = model_class(**kwargs, **extra).double()
                inputs = helper.batch(["A","B"])
                inputs["pbc"].fill_(True)
                inputs["cell"] = torch.eye(3).repeat(2,1)*5
                inputs["rcell"] = torch.eye(3).repeat(2,1)*(2*torch.pi/5)
                inputs["volume"] = torch.tensor([125.,125.])
                reference = model(copy.deepcopy(inputs), compute_force=True, compute_stress=True)
                model.configure_func_emb(config)
                actual = model(copy.deepcopy(inputs), compute_force=True, compute_stress=True)
                torch.testing.assert_close(actual["energy"]-reference["energy"], torch.tensor([0.,3.]),atol=1e-10,rtol=0)
                for key in ["forces","stress"]:
                    torch.testing.assert_close(actual[key],reference[key],atol=1e-10,rtol=1e-10)
                cfg = extract_config_mace_model(model)
                rebuilt = model_class(**cfg).double(); rebuilt.load_state_dict(model.state_dict())
                stream=io.BytesIO();torch.save(rebuilt,stream);stream.seek(0)
                loaded=torch.load(stream,weights_only=False)
                calc=MACECalculator(models=[loaded],model_type="MACE",default_dtype="float64",device="cpu")
                atoms=Atoms("HO",positions=[[0,0,0],[1.,.2,0]])
                atoms.info["level_of_theory"]="A";atoms.calc=calc;ea=atoms.get_potential_energy()
                atoms.info["level_of_theory"]="B";eb=atoms.get_potential_energy()
                self.assertAlmostEqual(eb-ea,3.,places=9)
        finally:
            torch.set_default_dtype(old_dtype)


if __name__ == "__main__": unittest.main()
