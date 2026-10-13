"""Exact-token causal timeline layout. No retokenization or truncation."""


def timeline_positions(calls, limit):
    if not calls:
        return None
    final = tuple(calls[-1].token_ids)
    positions = []
    for call in calls:
        if len(call.token_ids) != len(call.action_mask):
            raise ValueError("timeline token/mask alignment mismatch")
        if len(call.token_ids) > limit:
            raise ValueError("individual call exceeds model limit; cannot truncate")
        if tuple(call.token_ids) != final[: len(call.token_ids)]:
            return None
        row = [i for i, keep in enumerate(call.action_mask[1:]) if keep]
        if not row:
            return None
        if call.relation_token_span is not None:
            start, end = call.relation_token_span
            if end - start != 1 or row != [start - 1]:
                raise ValueError("invalid timeline relation target")
        positions.append(row)
    flat = [p for row in positions for p in row]
    if len(flat) != len(set(flat)):
        return None
    return positions


def stabilize_gated_delta_projections(model):
    """Use FP64 accumulation only for the two small GDN decay/gate projections.

    Parameters and returned activations retain their original dtype. This avoids
    sequence-length-dependent rounding in small-output GEMMs; it does not cast
    the full model or disconnect gradients. Applied once to an isolated learner.
    """
    import torch

    count = 0
    for name, module in model.named_modules():
        if not name.endswith((".linear_attn.in_proj_a", ".linear_attn.in_proj_b")):
            continue
        if getattr(module, "_timeline_stable_projection", False):
            continue

        def forward(x, _module=module):
            return torch.nn.functional.linear(
                x.double(),
                _module.weight.double(),
                _module.bias.double() if _module.bias is not None else None,
            ).to(x.dtype)

        module.forward = forward
        module._timeline_stable_projection = True
        count += 1
    return count


def enable_expanded_fp32_sdpa(model):
    """Permit memory-efficient FP32 SDPA without global backend switches.

    PyTorch's memory-efficient kernel needs equal Q/K/V head counts. Reuse
    Transformers' own SDPA adapter with explicit repeat_kv, on this model only.
    Causal/padding masks, scaling and dropout stay in that upstream adapter.
    """
    import copy
    import types
    from transformers.integrations.sdpa_attention import sdpa_attention_forward
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    name = "r2orch_expanded_fp32_sdpa"
    namespace = dict(sdpa_attention_forward.__globals__)
    namespace["use_gqa_in_sdpa"] = lambda *args, **kwargs: False
    adapted = types.FunctionType(
        sdpa_attention_forward.__code__,
        namespace,
        name,
        sdpa_attention_forward.__defaults__,
        sdpa_attention_forward.__closure__,
    )
    adapted.__kwdefaults__ = sdpa_attention_forward.__kwdefaults__
    ALL_ATTENTION_FUNCTIONS.register(name, adapted)
    count = 0
    for module_name, module in model.named_modules():
        if (
            module_name.endswith(".self_attn")
            and getattr(getattr(module, "config", None), "_attn_implementation", None) == "sdpa"
        ):
            module.config = copy.copy(module.config)
            module.config._attn_implementation = name
            count += 1
    return count
