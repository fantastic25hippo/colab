"""Chat-template prompt construction for LFM2.5-24B-A2B on TPU.

Pure Python (no vLLM / transformers import) so it is unit-testable locally.
2026-11-01 diagnostic: the raw-completion path loops ("astphysics"), the chat
template is coherent, so every prompt here is rendered by the chat template.
"""

BOS_ID = 124894
_TAIL_MIN = 3


def _ids(x):
    if isinstance(x, dict):
        x = x.get("input_ids", [])
    if hasattr(x, "tolist"):
        x = x.tolist()
    return [int(i) for i in x]


def chat_ids(tok, content):
    """Render one user turn. The template prepends BOS; vLLM adds none here."""
    return _ids(
        tok.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=True,
            add_generation_prompt=True,
        )
    )


def filler_text(tok, unit, target_ids):
    """Repeat `unit` so the rendered prompt overshoots `target_ids` (~8%)."""
    n_unit = max(1, len(tok.encode(unit)))
    return unit * (int(target_ids * 1.08 / n_unit) + 8)


def _suffix_len(a, b, cap):
    n = 0
    while n < min(len(a), len(b), cap) and a[-1 - n] == b[-1 - n]:
        n += 1
    return n


def check_prompt_ids(ids, target=None):
    """BOS must appear exactly once, at position 0. Returns the BOS count."""
    if target is not None and len(ids) != target:
        raise AssertionError(f"prompt has {len(ids)} ids, expected {target}")
    n = ids.count(BOS_ID)
    if ids[0] != BOS_ID or n != 1:
        raise AssertionError(
            f"BOS {BOS_ID} must appear once at position 0; first={ids[0]} count={n}")
    return n


def build_long_prompt_ids(tok, content, target):
    """One user message holding the filler, truncated to exactly `target` ids.

    The closing chat markers (im_end / im_start assistant / <think>) are taken
    from the empty-content render and re-appended, so the prompt still ends on
    the real generation prompt instead of a truncated filler fragment.
    """
    full = chat_ids(tok, content)
    empty = chat_ids(tok, "")
    if len(full) < target:
        raise ValueError(f"filler rendered {len(full)} ids, need >= {target}")
    tail_n = _suffix_len(full, empty, max(0, len(empty) - 1))
    if tail_n < _TAIL_MIN:
        raise ValueError(f"chat template tail not found (tail_n={tail_n})")
    ids = full[: target - tail_n] + empty[-tail_n:]
    check_prompt_ids(ids, target)
    return ids, {"target": target, "tail_n": tail_n, "rendered_ids": len(full)}
