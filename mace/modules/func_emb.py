"""Scalar theory embeddings with identity-initialized equivariant FiLM."""
import torch
from e3nn import o3
from mace.tools.func_emb_utils import normalize_func_emb


class TheoryFiLM(torch.nn.Module):
    def __init__(self, irreps, dim):
        super().__init__()
        irreps = o3.Irreps(irreps)
        gate_indices, bias_indices = [], []
        channel = 0
        for mul, ir in irreps:
            for _ in range(mul):
                gate_indices.extend([channel]*ir.dim)
                if ir == o3.Irrep("0e"):
                    bias_indices.append(len(gate_indices)-1)
                channel += 1
        self.register_buffer("gate_indices", torch.tensor(gate_indices, dtype=torch.long))
        self.register_buffer("bias_indices", torch.tensor(bias_indices, dtype=torch.long))
        self.num_gates = channel
        self.projection = torch.nn.Linear(dim, channel+len(bias_indices))
        torch.nn.init.zeros_(self.projection.weight)
        torch.nn.init.zeros_(self.projection.bias)

    def forward(self, features, embedding):
        parameters = self.projection(embedding)
        gates = parameters[..., :self.num_gates].index_select(-1, self.gate_indices)
        bias = torch.zeros_like(features)
        bias = bias.index_copy(-1, self.bias_indices, parameters[..., self.num_gates:])
        return features*(1+gates) + bias


class FunctionalEmbedding(torch.nn.Module):
    def __init__(self, config, atomic_numbers, site_irreps):
        super().__init__()
        self.config = normalize_func_emb(config)
        self.embedding = torch.nn.Embedding(len(self.config["theories"]), self.config["dim"])
        self.adapters = torch.nn.ModuleDict()
        self.site_keys = {}
        for site in self.config["sites"]:
            if site not in site_irreps:
                raise ValueError(f"Unknown func_emb site {site!r}; supported: {list(site_irreps)}")
            key = site.replace(".", "_")
            self.site_keys[site] = key
            self.adapters[key] = TheoryFiLM(site_irreps[site], self.config["dim"])
        rows = []
        for theory in self.config["theories"]:
            table = self.config["e0s"][theory]
            missing = [int(z) for z in atomic_numbers if str(int(z)) not in table]
            if missing:
                raise ValueError(f"Missing E0s for theory {theory}: atomic numbers {missing}")
            rows.append([table[str(int(z))] for z in atomic_numbers])
        self.register_buffer("e0_table", torch.tensor(rows, dtype=torch.get_default_dtype()))

    def node_embeddings(self, theory_id, batch):
        ids = theory_id.reshape(-1)
        if ids.dtype != torch.long or ids.numel() != int(batch.max())+1:
            raise ValueError("func_emb requires one integer theory_id per graph")
        if bool(((ids < 0) | (ids >= self.embedding.num_embeddings)).any()):
            raise ValueError("Theory ID outside checkpoint vocabulary")
        return self.embedding(ids)[batch]

    def condition(self, site, features, embedding):
        if site not in self.site_keys:
            return features
        return self.adapters[self.site_keys[site]](features, embedding)

    def atomic_energies(self, theory_id, batch, node_attrs):
        return (self.e0_table[theory_id.reshape(-1)[batch]].to(node_attrs.dtype)*node_attrs).sum(-1)
