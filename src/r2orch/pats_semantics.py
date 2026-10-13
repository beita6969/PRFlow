"""The Director's runtime contract (reference text for the skill reviewer) and its hash."""

from __future__ import annotations

import hashlib
import json

SEMANTIC_REVISION = "director_runtime_semantics_action_json"


def runtime_contract(prompt_variant: str) -> str:
    # Reuse the real Director's action examples and instructions. This avoids
    # maintaining a second, shorter action API that omits important conditions.
    from .director import director_prompt_components

    director_prompt, _ = director_prompt_components(prompt_variant)
    from .actions import UNIFIED_ACTION_FIELDS
    return director_prompt + "\nAction fields: " + json.dumps(UNIFIED_ACTION_FIELDS, sort_keys=True)


def contract_hash(prompt_variant: str) -> str:
    return hashlib.sha256(runtime_contract(prompt_variant).encode()).hexdigest()


def director_design_reference(prompt_variant: str) -> str:
    """Design guidance is useful to the Refiner, but is not an API legality test."""
    from .director import director_prompt_components

    director_prompt, _ = director_prompt_components(prompt_variant)
    return director_prompt
