import os
from pathlib import Path

from flask import Flask, request, jsonify
from flask_cors import CORS
from gemini_prompt_return import analyze_passwords
from password_analysis import score_passwords

# password_analysis.py uses paths like data/password_model.pkl relative to CWD
_BACKEND_DIR = Path(__file__).resolve().parent
os.chdir(_BACKEND_DIR)

app = Flask(__name__)
CORS(app)


def scores_dict_from_ml_rows(passwords, ml_rows):
    """Map original passwords to ML scores; order matches score_passwords output."""
    return {pwd: ml_rows[i]["score"] for i, pwd in enumerate(passwords)}


def build_fallback_report(passwords, scores, reason):
    average_score = sum(scores.values()) / max(len(scores), 1)
    top_password = max(scores, key=scores.get) if scores else "N/A"

    risk_level = "low"
    if average_score >= 70:
        risk_level = "high"
    elif average_score >= 40:
        risk_level = "medium"

    patterns = []
    if any(any(ch.isdigit() for ch in p) for p in passwords):
        patterns.append("Uses numbers in most passwords")
    if any(any(not ch.isalnum() for ch in p) for p in passwords):
        patterns.append("Uses symbol substitutions")
    if any(len(p) < 10 for p in passwords):
        patterns.append("Multiple short passwords detected")
    if not patterns:
        patterns.append("No obvious repeated structure detected")

    return {
        "fingerprint_summary": (
            f"AI analysis is temporarily unavailable ({reason}). "
            f"Local fallback indicates an overall {risk_level} risk profile with an average crackability score of {average_score:.1f}. "
            f"Most vulnerable password in this batch is '{top_password}'."
        ),
        "patterns": patterns,
        "attacker_wordlist": [
            "year suffixes",
            "keyboard patterns",
            "common substitutions",
            "pet names",
            "sports teams",
        ],
        "vulnerability_scores": {
            "dictionary_words": int(min(100, average_score)),
            "years_and_dates": int(min(100, average_score + 5)),
            "symbol_substitution": int(min(100, average_score - 5)),
            "keyboard_patterns": int(min(100, average_score)),
            "personal_references": int(min(100, average_score - 10)),
        },
        "tips": [
            "Use passphrases of 14+ characters with random words.",
            "Avoid predictable years, names, and keyboard patterns.",
            "Use a password manager to generate unique passwords.",
        ],
        "warning": "Generated from fallback mode because Gemini API is unavailable.",
    }


@app.route("/analyze", methods=["POST"])
def analyze():

    incoming_data = request.get_json()
    

    if not incoming_data or "passwords" not in incoming_data:
        return jsonify({"error": "Missing 'passwords' array"}), 400
        
    passwords = incoming_data.get("passwords", [])
    if len(passwords) < 2:
        return jsonify({"error": "Need at least 2 passwords"}), 400

    try:
        ml_rows = score_passwords(passwords)
        scores = scores_dict_from_ml_rows(passwords, ml_rows)
        try:
            ai_report = analyze_passwords(passwords, scores)
        except Exception as ai_error:
            print(f"Gemini unavailable, using fallback report: {ai_error}")
            ai_report = build_fallback_report(passwords, scores, str(ai_error))

        return jsonify({
            "scores": scores,
            "ml_scores": ml_rows,
            "ai_report": ai_report
        }), 200

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": "Server failed to process request"}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)