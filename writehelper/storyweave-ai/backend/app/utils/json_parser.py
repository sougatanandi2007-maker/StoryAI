"""
StoryWeave AI - Robust LLM JSON Parser

Extracts and parses valid JSON from LLM responses even if wrapped in markdown fences,
prefixed with commentary, or containing trailing characters.
"""

import json
import re
from typing import Dict, Any


def extract_json_from_llm_response(text: str) -> Dict[str, Any]:
    """
    Extract a valid JSON object from a raw LLM output string.

    Handles:
    - Clean JSON
    - Markdown code fences (```json ... ``` or ``` ... ```)
    - Preamble or postscript text outside the JSON structure
    """
    cleaned = text.strip()

    # 1. Strip markdown fences if present
    fence_pattern = r"```(?:json)?\s*([\s\S]*?)\s*```"
    match = re.search(fence_pattern, cleaned)
    if match:
        cleaned = match.group(1).strip()

    # 2. Try direct parse
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # 3. Find outermost curly braces if surrounded by commentary
    start_idx = cleaned.find("{")
    end_idx = cleaned.rfind("}")

    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        substring = cleaned[start_idx : end_idx + 1]
        try:
            return json.loads(substring)
        except json.JSONDecodeError as err:
            raise ValueError(f"Failed to parse extracted JSON substring: {err}")

    raise ValueError("No valid JSON object structure found in response.")
