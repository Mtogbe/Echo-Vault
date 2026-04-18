import google.generativeai as genai
import os
import json
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

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
    model = genai.GenerativeModel('gemini-1.5-flash')
    
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

    full_prompt = f"{system_instruction}\n{user_input}"
    response = model.generate_content(full_prompt)
    
    return parse_gemini_json(response.text)


if __name__ == "__main__":
    test_passwords = ["Dragon2022!", "Dragon2023!", "Dragon2024*"]
    mock_scores = {"Dragon2022!": 85, "Dragon2023!": 90, "Dragon2024*": 95}
    
    print("Sending data to Gemini API...")
    result = analyze_passwords(test_passwords, mock_scores)
    
    # terminal print the result remove later
    print("\n--- GEMINI RESPONSE ---")
    print(json.dumps(result, indent=2))