import json
import re
from typing import Optional


def extract_first_json(text: str) -> Optional[dict]:
    """Attempt to extract the first JSON object from a string.

    - Try direct json.loads first.
    - If that fails, search for the first {...} balanced braces block and parse it.
    - Returns dict if parsed, else None.
    """
    if not text or not isinstance(text, str):
        return None

    # Try direct parse
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    # Find first {...} block using a simple stack-based scan
    start = None
    stack = 0
    for i, ch in enumerate(text):
        if ch == '{':
            if start is None:
                start = i
            stack += 1
        elif ch == '}' and start is not None:
            stack -= 1
            if stack == 0:
                candidate = text[start:i + 1]
                try:
                    parsed = json.loads(candidate)
                    if isinstance(parsed, dict):
                        return parsed
                except Exception:
                    # continue searching
                    start = None
                    stack = 0

    # As a last resort, try to find a JSON-like substring with regex (simple heuristic)
    m = re.search(r"\{[\s\S]*?\}", text)
    if m:
        try:
            parsed = json.loads(m.group(0))
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            return None

    return None


def extract_all_jsons(text: str) -> list:
    """Extract all top-level JSON objects or arrays from text.

    Returns a list of parsed Python objects (dict or list). If none found, returns [].
    """
    results = []
    if not text or not isinstance(text, str):
        return results

    # Try to parse entire text as JSON (single object/array)
    try:
        parsed = json.loads(text)
        if isinstance(parsed, (dict, list)):
            return [parsed]
    except Exception:
        pass

    # Find balanced {...} and [...] blocks using simple scans
    stack = []
    start = None
    kind = None
    for i, ch in enumerate(text):
        if ch == '{' or ch == '[':
            if start is None:
                start = i
                kind = ch
            stack.append(ch)
        elif ch == '}' or ch == ']':
            if stack:
                stack.pop()
            if not stack and start is not None:
                candidate = text[start:i+1]
                try:
                    parsed = json.loads(candidate)
                    if isinstance(parsed, (dict, list)):
                        results.append(parsed)
                except Exception:
                    pass
                start = None
                kind = None

    # As a last resort, regex find arrays or objects
    if not results:
        for m in re.finditer(r"(\{[\s\S]*?\}|\[[\s\S]*?\])", text):
            try:
                parsed = json.loads(m.group(0))
                if isinstance(parsed, (dict, list)):
                    results.append(parsed)
            except Exception:
                continue

    return results
