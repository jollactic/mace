"""Validation and categorical encoding for level-of-theory conditioning."""
import copy
import math
from ase.data import atomic_numbers


def normalize_func_emb(config):
    if not isinstance(config, dict):
        raise ValueError("func_emb must be a configuration dictionary")
    allowed = {"key", "dim", "sites", "theories", "reference_theory", "default_theory", "e0s", "loss_weights"}
    if set(config)-allowed:
        raise ValueError(f"Unknown func_emb configuration fields: {set(config)-allowed}")
    config = copy.deepcopy(config)
    theories = config.get("theories", [])
    if not isinstance(theories, list) or not theories or any(not isinstance(t, str) or not t for t in theories) or len(set(theories)) != len(theories):
        raise ValueError("func_emb.theories must be an ordered list of unique nonempty names")
    config.setdefault("key", "level_of_theory")
    if not isinstance(config["key"], str) or not config["key"]:
        raise ValueError("func_emb.key must be a nonempty metadata key")
    config.setdefault("dim", 16)
    config.setdefault("sites", ["node_embedding"])
    config.setdefault("reference_theory", theories[0])
    config.setdefault("default_theory", None)
    if type(config["dim"]) is not int or config["dim"] < 1:
        raise ValueError("func_emb.dim must be a positive integer")
    if not isinstance(config["sites"], list) or any(not isinstance(site, str) for site in config["sites"]) or len(set(config["sites"])) != len(config["sites"]):
        raise ValueError("func_emb.sites must be a list without duplicates")
    for key in ["reference_theory", "default_theory"]:
        if config[key] is not None and config[key] not in theories:
            raise ValueError(f"Unknown {key}: {config[key]}")
    e0s = config.get("e0s", {})
    if set(e0s) != set(theories):
        raise ValueError("func_emb.e0s must contain exactly the configured theory names")
    normalized = {}
    for theory in theories:
        normalized[theory] = {}
        for element, value in e0s[theory].items():
            z = atomic_numbers.get(str(element))
            if z is None:
                try: z = int(element)
                except (ValueError, TypeError): raise ValueError(f"Unknown element {element}")
            try:
                value = float(value)
            except (TypeError, ValueError):
                raise ValueError(f"Supply a numerical E0 in eV for {theory}/{element}") from None
            if z < 1 or z > 118 or str(z) in normalized[theory] or not math.isfinite(value):
                raise ValueError(f"Invalid/duplicate E0 for {theory}/{element}")
            normalized[theory][str(z)] = float(value)
    weights = config.get("loss_weights", {})
    if set(weights)-set(theories):
        raise ValueError("Unknown theory in func_emb.loss_weights")
    config["loss_weights"] = {name: float(weights.get(name, 1.0)) for name in theories}
    if any(not math.isfinite(w) or w < 0 for w in config["loss_weights"].values()):
        raise ValueError("Theory loss weights must be finite and nonnegative")
    config["e0s"] = normalized
    return config


def encode_theory(info, config):
    name = info.get(config["key"], config["default_theory"])
    if name not in config["theories"]:
        raise ValueError(f"Missing or unknown {config['key']}: {name!r}; expected {config['theories']}")
    return config["theories"].index(name)
