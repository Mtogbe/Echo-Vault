import google.generativeai as genai
import os
import json
from pathlib import Path
from dotenv import load_dotenv

_BACKEND_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _BACKEND_DIR.parent
# Load .env from repo root first (Flask cwd is often backend/)
load_dotenv(_PROJECT_ROOT / ".env")
load_dotenv(_BACKEND_DIR / ".env")

_api_key = (os.getenv("GEMINI_API_KEY") or "").strip()
if _api_key:
    genai.configure(api_key=_api_key)

# Used only if list_models() fails or returns nothing
STATIC_MODEL_FALLBACKS = [
    "gemini-2.5-flash",
    "gemini-2.5-flash-preview-05-20",
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
    "gemini-1.5-flash-8b",
    "gemini-1.5-flash-001",
    "gemini-1.5-flash-latest",
    "gemini-1.5-flash",
]


def _strip_models_prefix(name: str) -> str:
    if name.startswith("models/"):
        return name[len("models/") :]
    return name


def discover_generate_content_models():
    """Ask the API which model IDs support generateContent for this key."""
    names = []
    try:
        for m in genai.list_models():
            methods = getattr(m, "supported_generation_methods", None) or []
            if "generateContent" in methods and m.name:
                names.append(_strip_models_prefix(m.name))
    except Exception as err:
        print(f"list_models failed: {err}")
    return names


def ordered_model_candidates():
    """Prefer newer flash models, then any other generateContent-capable model."""
    discovered = discover_generate_content_models()
    priority_keywords = (
        "gemini-2.5",
        "gemini-2.0",
        "gemini-1.5",
        "gemini",
    )
    ordered = []
    for kw in priority_keywords:
        for n in discovered:
            if kw in n.lower() and n not in ordered:
                ordered.append(n)
    for n in discovered:
        if n not in ordered:
            ordered.append(n)
    for n in STATIC_MODEL_FALLBACKS:
        if n not in ordered:
            ordered.append(n)
    return ordered


def parse_gemini_json(raw_text):
    """Safely extracts JSON from the AI's text response."""
    try:
      
        clean_text = raw_text.strip()
        if clean_text.startswith("```json"):
            clean_text = clean_text[7:]
        if clean_text.endswith("```"):
            clean_text = clean_text[:-3]
        
        return json.loads(clean_text)
    except json.JSONDecodeError as e:
        print(f"JSON Parse Error: {e}")
        
        return {
            "error": "Failed to parse AI response",
            "fingerprint_summary": "Analysis failed due to a formatting error.",
            "patterns": [],
            "attacker_wordlist": [],
            "vulnerability_scores": {},
            "tips": []
        }

def analyze_passwords(passwords, crackability_scores):
    system_instruction = (
        "You are a cybersecurity pattern-matching engine. "
        "Analyze the provided passwords for cognitive biases and psychological patterns. "
        "Your output must be a single, valid JSON object."
    )

    # JSON struvture
    user_input = f"""
    Analyze these passwords: {passwords}
    
    The backend algorithmic crackability scores for these passwords are: {crackability_scores}
    (Note: A score of 0 means uncrackable, 100 means instantly crackable).
    
    Incorporate these scores into your analysis. If the score is high, your fingerprint_summary 
    should reflect the severe vulnerability, and your tips should be more urgent.
    
    Return a JSON object with this exact structure:
    {{
      "fingerprint_summary": "A 3-sentence psychological profile that mentions their overall crackability risk.",
      "patterns": ["list", "of", "detected", "habits"],
      "attacker_wordlist": ["10", "predicted", "variations"],
      "vulnerability_scores": {{
        "dictionary_words": 0,
        "years_and_dates": 0,
        "symbol_substitution": 0,
        "keyboard_patterns": 0,
        "personal_references": 0
      }},
      "tips": ["tip1", "tip2", "tip3"]
    }}
    """

    if not _api_key:
        raise RuntimeError("GEMINI_API_KEY is missing. Set it in .env at the project root.")

    full_prompt = f"{system_instruction}\n{user_input}"
    last_error = None

    for model_name in ordered_model_candidates():
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(full_prompt)
            return parse_gemini_json(response.text)
        except Exception as err:
            last_error = err
            print(f"Model attempt failed ({model_name}): {err}")

    raise RuntimeError(f"All Gemini model attempts failed: {last_error}")


if __name__ == "__main__":
    test_passwords = ["Dragon2022!", "Dragon2023!", "Dragon2024*"]
    mock_scores = {"Dragon2022!": 85, "Dragon2023!": 90, "Dragon2024*": 95}
    
    print("Sending data to Gemini API...")
    result = analyze_passwords(test_passwords, mock_scores)
    
    # terminal print the result remove later
    print("\n--- GEMINI RESPONSE ---")
    print(json.dumps(result, indent=2))