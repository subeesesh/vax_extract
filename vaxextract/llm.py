"""OpenAI-compatible chat client (works with Gemini, OpenAI, vLLM, Ollama, ...)."""
from __future__ import annotations

import base64
import json
import os
import re
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import httpx

from .prompt import EXPAND_PROMPT, SYSTEM_PROMPT, user_prompt
from .schema import RawExtraction


@dataclass
class LlmConfig:
    base_url: str
    api_key: str
    model: str
    timeout: float = 120.0
    max_tokens: int = 16384  # long multi-page histories overflowed 4096 (truncated JSON)
    vision: bool = False
    reasoning_effort: Optional[str] = None  # e.g. 'low' for Gemini 3.x / OpenAI reasoning models
    hedge_after: float = 0.0  # >0: send a duplicate request if no answer after this many seconds; first wins


class LlmError(RuntimeError):
    pass


def _nullable(t: dict) -> dict:
    return {"anyOf": [t, {"type": "null"}]}


def output_schema(targets: Optional[list[str]] = None) -> dict:
    """JSON Schema for constrained decoding: no malformed JSON, no repair round-trip, and match names
    restricted to the requested names."""
    s, i = _nullable({"type": "string"}), _nullable({"type": "integer"})
    name = {"type": "string", "enum": list(targets)} if targets else {"type": "string"}

    def obj(props: dict) -> dict:
        return {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}

    match = obj({"name": name, "partial": {"type": "boolean"}, "duration_months": i})
    event = obj({"raw_name": {"type": "string"}, "kind": {"type": "string", "enum": ["vaccine", "diagnostic", "other"]},
                 "given_date": s, "due_date": s, "duration_months": i,
                 "matches": {"type": "array", "items": match, **({} if targets else {"maxItems": 0})}})
    pet = obj({**{k: s for k in ("name", "species", "breed", "sex", "weight", "color", "microchipNumber")},
               "events": {"type": "array", "items": event}})
    party = obj({"name": s, "address": s, "phone": s, "email": s})
    return obj({"pets": {"type": "array", "items": pet}, "owner": party,
                "clinic": obj({"name": s, "address": s, "phone": s, "email": s, "website": s}),
                "veterinarian": obj({"name": s, "licenseNumber": s})})


_FORMATS = ("json_schema", "json_object", None)
_POOL = ThreadPoolExecutor(max_workers=8, thread_name_prefix="llm-hedge")  # not joined: a losing request just finishes


def _send(url: str, payload: dict, headers: dict, cfg: LlmConfig) -> httpx.Response:
    """One HTTP call, hedged: the provider's latency has a long tail (same request: 5 s usually, 60-80 s sometimes),
    so after `hedge_after` seconds an identical request is raced against it and the first response is used."""
    if cfg.hedge_after <= 0:
        return httpx.post(url, json=payload, headers=headers, timeout=cfg.timeout)
    futs = {_POOL.submit(httpx.post, url, json=payload, headers=headers, timeout=cfg.timeout)}
    done, _ = wait(futs, timeout=cfg.hedge_after)
    if not done:
        futs.add(_POOL.submit(httpx.post, url, json=payload, headers=headers, timeout=cfg.timeout))
    pending, err = futs, None
    while pending:
        done, pending = wait(pending, return_when=FIRST_COMPLETED)
        for f in done:
            try:
                r = f.result()
            except httpx.HTTPError as e:
                err = e
                continue
            if r.status_code == 200 or not pending:
                return r
    raise err


def _post(cfg: LlmConfig, messages: list, schema: Optional[dict] = None, fmt: Optional[str] = "json_schema"):
    if fmt == "json_schema" and schema is None:
        fmt = "json_object"
    payload = {"model": cfg.model, "messages": messages, "temperature": 0, "max_tokens": cfg.max_tokens}
    if cfg.reasoning_effort:
        payload["reasoning_effort"] = cfg.reasoning_effort
    if fmt == "json_schema":
        payload["response_format"] = {"type": "json_schema",
                                      "json_schema": {"name": "extraction", "strict": True, "schema": schema}}
    elif fmt == "json_object":
        payload["response_format"] = {"type": "json_object"}
    headers = {"Authorization": f"Bearer {cfg.api_key}", "Content-Type": "application/json"}
    url = cfg.base_url.rstrip("/") + "/chat/completions"
    last: Optional[Exception] = None
    for attempt in range(3):
        try:
            r = _send(url, payload, headers, cfg)
            if r.status_code == 400 and fmt and ("response_format" in r.text or "schema" in r.text):
                return _post(cfg, messages, schema, _FORMATS[_FORMATS.index(fmt) + 1])
            if r.status_code in (429, 500, 502, 503, 504):
                raise httpx.HTTPStatusError(f"{r.status_code}", request=r.request, response=r)
            r.raise_for_status()
            body = r.json()
            usage = body.get("usage") or {}
            return body["choices"][0]["message"]["content"] or "", usage
        except (httpx.HTTPError, KeyError, ValueError) as e:
            last = e
            time.sleep(1.5 * (attempt + 1))
    raise LlmError(f"LLM call failed: {last}")


def parse_json(text: str) -> dict:
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t)
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        i, j = t.find("{"), t.rfind("}")
        if i == -1 or j <= i:
            raise
        return json.loads(t[i : j + 1])


_EXPANSIONS: dict = {}  # (model, targets) -> {name: [{"label", "species", "diseases"}]}
_CACHE_LOCK = threading.Lock()


def _cache_file() -> Optional[Path]:
    """Expansions persist across restarts here (VAX_EXPANSION_CACHE; empty disables)."""
    p = os.environ.get("VAX_EXPANSION_CACHE", ".vax_cache/expansions.json")
    return Path(p) if p else None


def _cache_key(key: tuple) -> str:
    return key[0] + "||" + ";".join(key[1])


def _load_cached(key: tuple):
    f = _cache_file()
    if not f or not f.exists():
        return None
    try:
        return json.loads(f.read_text(encoding="utf-8")).get(_cache_key(key))
    except (OSError, ValueError):
        return None


def _store_cached(key: tuple, exp: dict) -> None:
    f = _cache_file()
    if not f:
        return
    with _CACHE_LOCK:
        try:
            all_ = json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}
        except (OSError, ValueError):
            all_ = {}
        all_[_cache_key(key)] = exp
        try:
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text(json.dumps(all_, ensure_ascii=False, indent=1), encoding="utf-8")
        except OSError:
            pass


def _expansion_schema(targets: list[str]) -> dict:
    alt = {"type": "object", "additionalProperties": False, "required": ["label", "species", "diseases"],
           "properties": {"label": {"type": "string"}, "species": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                          "diseases": {"type": "array", "items": {"type": "string"}}}}
    item = {"type": "object", "additionalProperties": False, "required": ["name", "alternatives"],
            "properties": {"name": {"type": "string", "enum": list(targets)},
                           "alternatives": {"type": "array", "items": alt}}}
    return {"type": "object", "additionalProperties": False, "required": ["names"],
            "properties": {"names": {"type": "array", "items": item}}}


def expand_targets(cfg: LlmConfig, targets: Optional[list[str]]):
    """Requested names -> the diseases each stands for, written by the LLM once per target list and cached.
    The extraction call applies these definitions far more reliably than it expands abbreviations itself.
    -> (expansion dict, usage dict); ({}, {}) when there are no targets or the call fails (extraction still works)."""
    if not targets:
        return {}, {}
    key = (cfg.model, tuple(targets))
    if key in _EXPANSIONS:
        return _EXPANSIONS[key], {}
    cached = _load_cached(key)
    if cached is not None:
        _EXPANSIONS[key] = cached
        return cached, {}
    msgs = [{"role": "system", "content": EXPAND_PROMPT}, {"role": "user", "content": "\n".join(f"- {t}" for t in targets)}]
    try:
        out, usage = _post(cfg, msgs, _expansion_schema(targets))
        exp = {}
        for n in parse_json(out).get("names", []):
            if n.get("name") in targets and n.get("alternatives"):
                exp[n["name"]] = n["alternatives"]
    except (LlmError, ValueError, AttributeError):
        return {}, {}
    _EXPANSIONS[key] = exp
    if exp:
        _store_cached(key, exp)
    return exp, usage


def extract(cfg: LlmConfig, text: str, images: Optional[list[bytes]] = None, targets: Optional[list[str]] = None):
    """-> (RawExtraction, usage dict). One repair retry on invalid JSON."""
    total = {"prompt_tokens": 0, "completion_tokens": 0}
    expansion, usage0 = expand_targets(cfg, targets)
    for k in total:
        total[k] += int(usage0.get(k, 0) or 0)
    use_imgs = bool(images) and cfg.vision
    if use_imgs:
        content = [{"type": "text", "text": user_prompt(text, True, targets, expansion)}] + [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(b).decode()}}
            for b in images
        ]
    else:
        content = user_prompt(text, targets=targets, expansion=expansion)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": content}]
    schema = output_schema(targets)
    out, usage = _post(cfg, messages, schema)
    for k in total:
        total[k] += int(usage.get(k, 0) or 0)
    try:
        return RawExtraction.model_validate(parse_json(out)), total
    except Exception as err:  # invalid JSON or schema -> one repair attempt
        messages += [
            {"role": "assistant", "content": out},
            {"role": "user", "content": f"Invalid output ({str(err)[:300]}). Return ONLY the corrected JSON object."},
        ]
        out2, usage2 = _post(cfg, messages, schema)
        for k in total:
            total[k] += int(usage2.get(k, 0) or 0)
        try:
            return RawExtraction.model_validate(parse_json(out2)), total
        except Exception as err2:
            raise LlmError(f"unparseable LLM output: {err2}")
