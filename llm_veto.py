#!/usr/bin/env python3
"""llm_veto.py — fine-tuned Qwen2.5-7B decision layer for the autotrader.

Loads the 4-bit base + LoRA adapter once (lazy singleton), then answers
market-state prompts in the engine's decision language. Used by
supervisor.py as a confirm/veto filter before entries:

    veto = llm_veto.VetoLayer()
    d = veto.decide("ES 3m. RSI 45, EMA10 above EMA30, stochastic 62 rising, ATR 8. Score +2.")
    # d == {"action": "BUY"|"SELL"|"NO TRADE", "reason": "...", "agree": bool}

Load pattern matches vram_profile.py (proven on GTX 1070): 4-bit NF4 base,
explicit device_map="cuda:0" (NO device_map="auto" — offload meta-devices
break tied lm_head + peft adapter load), then PeftModel.from_pretrained.
"""
import os
import re
import time
from collections import OrderedDict

import torch

BASE = os.path.join(os.path.expanduser("~"), "qwen-dl")
ADAPTER = os.environ.get(
    "VETO_ADAPTER",
    os.path.join(os.path.expanduser("~"), "projects/algoTraderBot/finetune/output8b"))
MODEL_LOCK = time.time()  # placeholder replaced by lazy singleton below

_inst = None

# ── decision cache ─────────────────────────────────────────────────────
# The veto decodes GREEDILY (do_sample=False), so the same state_text ALWAYS
# yields the same action. In quiet markets the state line repeats (RSI 45,
# EMA10 above EMA30, stoch 62 rising, ATR 8...) — caching makes the repeat
# answer instant instead of a ~17s GPU call. Bounded LRU; safe by
# determinism (identical input -> identical output, verified 2026-08-17).
_DECIDE_CACHE = OrderedDict()
_DECIDE_CACHE_MAX = 256
_cache_hits = 0
_cache_misses = 0


def cache_stats() -> dict:
    """Hit/miss counters — lets the watchdog/doctor see the cache working."""
    return {"hits": _cache_hits, "misses": _cache_misses,
            "size": len(_DECIDE_CACHE),
            "hit_rate": round(_cache_hits / max(1, _cache_hits + _cache_misses), 3)}


def _cache_get(state_text: str):
    global _cache_hits, _cache_misses
    if state_text in _DECIDE_CACHE:
        _cache_hits += 1
        _DECIDE_CACHE.move_to_end(state_text)
        return _DECIDE_CACHE[state_text]
    _cache_misses += 1
    return None


def _cache_put(state_text: str, value: dict):
    _DECIDE_CACHE[state_text] = value
    _DECIDE_CACHE.move_to_end(state_text)
    while len(_DECIDE_CACHE) > _DECIDE_CACHE_MAX:
        _DECIDE_CACHE.popitem(last=False)


def _load():
    global _inst
    if _inst is not None:
        return _inst
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              BitsAndBytesConfig)
    from peft import PeftModel

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(BASE)
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.float16,
                             bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(
        BASE, quantization_config=bnb, torch_dtype=torch.float16,
        device_map="cuda:0")
    model = PeftModel.from_pretrained(model, ADAPTER, is_trainable=False)
    model.eval()
    print(f"[llm_veto] loaded base+adapter in {time.time()-t0:.0f}s", flush=True)
    _inst = (model, tok)
    return _inst


def _extract_state_numbers(state_text: str) -> set:
    """Numeric facts from the state line: RSI/stoch/ATR/score values + the
    symbol. Used by the evidence-grounding check (Trading-R1 stage-II idea:
    a trade decision must cite at least one concrete market fact)."""
    facts = set(re.findall(r"\d+(?:\.\d+)?", state_text or ""))
    # symbol token(s) like "ES", "NQ"
    m = re.match(r"\s*([A-Z]{2,3})\s+\dm\.", state_text or "")
    if m:
        facts.add(m.group(1))
    return facts


def grounding_check(state_text: str, reason: str) -> dict:
    """Trading-R1-style evidence-grounding check (ADVISORY — logged, never
    blocks). A decision reason should reference at least one concrete fact
    from the market state it was given (a number like the RSI/ATR value, a
    direction word tied to EMA, or the symbol itself). Ungrounded reasons
    are flagged so we can track how often the model answers generically.

    Returns {"grounded": bool, "matched": str}."""
    facts = _extract_state_numbers(state_text)
    r = (reason or "").lower()
    # numeric fact match (any number from the state line appearing in reason)
    for f in facts:
        if f.lower() in r and any(c.isdigit() for c in f):
            return {"grounded": True, "matched": f}
    # indicator-word match (EMA / RSI / stochastic / ATR / score mentioned)
    words = ("ema", "rsi", "stochastic", "stoch", "atr", "score", "trend")
    if any(w in r for w in words):
        return {"grounded": True, "matched": next(w for w in words if w in r)}
    # symbol match
    for f in facts:
        if not any(c.isdigit() for c in f) and f.lower() in r:
            return {"grounded": True, "matched": f}
    return {"grounded": False, "matched": ""}


_UNGROUNDED = {"n": 0, "total": 0}


def grounding_stats() -> dict:
    """Ungrounded-reason rate — surfaced by doctor/watchdog."""
    t = max(1, _UNGROUNDED["total"])
    return {**_UNGROUNDED, "ungrounded_rate": round(_UNGROUNDED["n"] / t, 3)}


def decide(state_text: str, max_new_tokens: int = 100) -> dict:
    """Ask the fine-tuned model for a trade decision on a market state.

    state_text: engine-style description, e.g.
      "ES 3m. RSI 45, EMA10 above EMA30, stochastic 62 rising, ATR 8. Score +2."
    Returns {"action": ..., "reason": ..., "quality": int 1-10, "raw": ...}.
    Never raises for a bad model response — action falls back to "NO TRADE"
    and quality to 0 (both safe defaults).

    NOTE (2026-08-17): the v1 veto model (output8b) is trained on the plain
    "Answer with the trade action and one reason line." suffix — the quality
    variant (output8b_qual) failed its holdout eval, so we reverted the
    adapter. Keep THIS prompt byte-identical to v1's training format.
    """
    cached = _cache_get(state_text)
    if cached is not None:
        return dict(cached)             # copy — caller may mutate

    model, tok = _load()
    msgs = [{"role": "user",
             "content": state_text + " Answer with the trade action and one reason line."}]
    prompt = tok.apply_chat_template(msgs, tokenize=False,
                                     add_generation_prompt=True)
    ids = tok(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=max_new_tokens,
                             do_sample=False, pad_token_id=tok.eos_token_id)
    raw = tok.decode(out[0][ids["input_ids"].shape[1]:],
                     skip_special_tokens=True).strip()
    first = raw.split()[0].upper() if raw.split() else ""
    action = first if first in ("BUY", "SELL", "NO", "NO TRADE") else "NO TRADE"
    if action == "NO":
        action = "NO TRADE"
    reason = re.sub(r"^\S+\s*", "", raw).strip() or raw
    # quality: only if the model volunteered it (v1 usually doesn't — that's
    # fine; quality_min=0 in prod means the gate is inert)
    m = re.search(r"(?:Quality Score|quality)[:\s]+(\d{1,2})(?:/10)?",
                  raw, re.IGNORECASE)
    quality = int(m.group(1)) if m else 0
    quality = quality if 1 <= quality <= 10 else 0
    d = {"action": action, "reason": reason[:200], "quality": quality,
         "raw": raw}
    # evidence-grounding check (Trading-R1 stage-II idea) — ADVISORY ONLY:
    # logs/counts ungrounded reasons, NEVER changes the decision.
    g = grounding_check(state_text, reason)
    _UNGROUNDED["total"] += 1
    if not g["grounded"]:
        _UNGROUNDED["n"] += 1
        print(f"[llm_veto] ⚠ UNGROUNDED reason for '{action}': "
              f"'{reason[:80]}' | state: {state_text[:60]}", flush=True)
    d["grounded"] = g["grounded"]
    _cache_put(state_text, d)
    return d


def quality(state_text: str) -> int:
    """Rate a market state 1-10 (terrible → perfect setup) for a futures trade.

    SEPARATE short call from decide(): the decision prompt must stay exactly
    in the training distribution; appending the score request to it degraded
    decisions (verified). max_new_tokens=8 so it's ~2x faster than decide.
    Returns an int 1-10, or 0 if unparseable (0 → supervisor blocks).
    Cached like decide(): greedy decode is deterministic, so the same state
    always yields the same score — a prefetched score is an instant hit.
    """
    cached = _cache_get("Q:" + state_text)
    if cached is not None:
        return int(cached)
    model, tok = _load()
    msgs = [{"role": "user",
             "content": state_text +
             " Rate this exact market state for a futures trade from 1 "
             "(terrible setup) to 10 (perfect setup). Reply with just the number."}]
    prompt = tok.apply_chat_template(msgs, tokenize=False,
                                     add_generation_prompt=True)
    ids = tok(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=8, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    raw = tok.decode(out[0][ids["input_ids"].shape[1]:],
                     skip_special_tokens=True).strip()
    m = re.search(r"\b([1-9]|10)\b", raw)
    q = int(m.group(1)) if m else 0
    _cache_put("Q:" + state_text, q)
    return q


def decide_batch(state_texts, max_new_tokens=40):
    """Batched decide: ONE model.generate() call for many states (true GPU
    batching — ~2-3x throughput vs the serialized loop). Cached states are
    answered from the cache, only misses hit the GPU. max_new_tokens=40 is
    safe: the action is always the FIRST tokens; reasons truncate rarely.
    Returns a list of dicts in input order (action/reason/quality, no raw).
    """
    model, tok = _load()
    missing, idx_miss = [], []
    out = [None] * len(state_texts)
    for i, t in enumerate(state_texts):
        c = _cache_get(t)
        if c is not None:
            out[i] = dict(c)
        else:
            missing.append(t)
            idx_miss.append(i)
    if missing:
        msgs = [[{"role": "user",
                  "content": t + " Answer with the trade action and one reason line."}]
                for t in missing]
        prompts = [tok.apply_chat_template(m, tokenize=False,
                                           add_generation_prompt=True) for m in msgs]
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        enc = tok(prompts, return_tensors="pt", padding=True,
                  truncation=True, max_length=512).to("cuda")
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=max_new_tokens,
                                 do_sample=False, pad_token_id=tok.eos_token_id)
        for j, i in enumerate(idx_miss):
            raw = tok.decode(gen[j][enc["input_ids"][j].shape[0]:],
                             skip_special_tokens=True).strip()
            first = raw.split()[0].upper() if raw.split() else ""
            action = first if first in ("BUY", "SELL", "NO", "NO TRADE") else "NO TRADE"
            if action == "NO":
                action = "NO TRADE"
            reason = re.sub(r"^\S+\s*", "", raw).strip() or raw
            m = re.search(r"(?:Quality Score|quality)[:\s]+(\d{1,2})(?:/10)?",
                          raw, re.IGNORECASE)
            quality = int(m.group(1)) if m else 0
            quality = quality if 1 <= quality <= 10 else 0
            d = {"action": action, "reason": reason[:200], "quality": quality}
            _cache_put(missing[j], d)
            out[i] = d
    return out


def quality_batch(state_texts, max_new_tokens: int = 8):
    """Batched quality: ONE model.generate() call for many states (true GPU
    batching, mirrors decide_batch). Same prompt + parsing as quality() —
    identical scores, ~10-20x throughput. Cached states answered from cache.
    Returns list of ints in input order (0 = unparseable).
    """
    model, tok = _load()
    missing, idx_miss = [], []
    out = [0] * len(state_texts)
    for i, t in enumerate(state_texts):
        c = _cache_get("Q:" + t)
        if c is not None:
            out[i] = int(c)
        else:
            missing.append(t)
            idx_miss.append(i)
    if missing:
        msgs = [[{"role": "user",
                  "content": t +
                  " Rate this exact market state for a futures trade from 1 "
                  "(terrible setup) to 10 (perfect setup). Reply with just the number."}]
                for t in missing]
        prompts = [tok.apply_chat_template(m, tokenize=False,
                                           add_generation_prompt=True) for m in msgs]
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        enc = tok(prompts, return_tensors="pt", padding=True,
                  truncation=True, max_length=512).to("cuda")
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=max_new_tokens,
                                 do_sample=False, pad_token_id=tok.eos_token_id)
        for j, i in enumerate(idx_miss):
            raw = tok.decode(gen[j][enc["input_ids"][j].shape[0]:],
                             skip_special_tokens=True).strip()
            m = re.search(r"\b([1-9]|10)\b", raw)
            q = int(m.group(1)) if m else 0
            _cache_put("Q:" + missing[j], q)
            out[i] = q
    return out


def engine_agrees(engine_side: str, d: dict) -> bool:
    """True when the model confirms the engine's proposed side."""
    if d["action"] == "NO TRADE":
        return False
    return d["action"] == engine_side.upper()


if __name__ == "__main__":
    import sys
    tests = [
        "ES 3m. RSI 45, EMA10 above EMA30, stochastic 62 rising, ATR 8. Score +2.",
        "NQ 3m. RSI 68, EMA10 above EMA30, stochastic 95 falling, ATR 6. Score +1.",
        "GC 3m. RSI 32, EMA10 below EMA30, stochastic 20 rising, ATR 4. Score -3.",
    ]
    for t in tests:
        d = decide(t)
        print(f"PROMPT: {t}")
        print(f"  -> {d['action']} | {d['reason']}")
        print()
