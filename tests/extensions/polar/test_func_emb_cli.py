"""Tiny end-to-end mixed-theory training without network access."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np
import torch
from ase import Atoms
from ase.io import write
from test_func_emb import _build_minimal_model
from mace.calculators import MACECalculator


class FuncEmbCLITest(unittest.TestCase):
    def test_mixed_theory_training(self):
        old_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.float64)
        try:
            with tempfile.TemporaryDirectory() as directory:
                work = Path(directory)
                model = _build_minimal_model("cpu", torch.float64)
                torch.save(model, work/"base.model")
                config = {"theories": ["A", "B"], "dim": 4,
                          "sites": ["node_embedding", "products.0", "charge_updates.0"],
                          "e0s": {"A": {"H": 0, "O": 0}, "B": {"H": 1, "O": 2}}}
                (work/"func.json").write_text(json.dumps(config))
                frames = []
                for i in range(4):
                    atoms = Atoms("HO", positions=[[0,0,0],[1.+.1*i,0,0]])
                    atoms.info.update(charge=0, spin=2, level_of_theory=["A","B"][i%2], REF_energy=float(3*(i%2)))
                    atoms.arrays["REF_forces"] = np.zeros((2,3))
                    frames.append(atoms)
                write(work/"train.xyz",frames); write(work/"valid.xyz",frames[:2])
                options = dict(name="func_smoke", model="PolarMACE", foundation_model=str(work/"base.model"),
                    func_emb=str(work/"func.json"), multiheads_finetuning="False", foundation_model_readout="True",
                    train_file=str(work/"train.xyz"), valid_file=str(work/"valid.xyz"),
                    total_charge_key="charge", total_spin_key="spin", batch_size=2, valid_batch_size=2,
                    max_num_epochs=1, default_dtype="float64", device="cpu", lr=1e-4,
                    model_dir=str(work), log_dir=str(work/"logs"), results_dir=str(work/"results"), checkpoints_dir=str(work/"checkpoints"))
                env = os.environ.copy(); env["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"]="1"
                result = subprocess.run([sys.executable,"-m","mace.cli.run_train"]+[f"--{k}={v}" for k,v in options.items()],
                    env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=180)
                self.assertEqual(result.returncode,0,result.stdout[-8000:])
                self.assertIn("Theory A validation",result.stdout)
                self.assertIn("Theory B validation",result.stdout)
                calc = MACECalculator(model_paths=str(work/"func_smoke.model"), model_type="PolarMACE", device="cpu", default_dtype="float64", pbc_handling="auto")
                self.assertEqual(calc.models[0].func_emb_config["theories"],["A","B"])
                self.assertGreater(torch.count_nonzero(calc.models[0].func_emb.adapters["node_embedding"].projection.weight).item(),0)
                atoms=frames[0].copy(); atoms.calc=calc
                self.assertTrue(np.isfinite(atoms.get_potential_energy()))
        finally:
            torch.set_default_dtype(old_dtype)


if __name__ == "__main__": unittest.main()
