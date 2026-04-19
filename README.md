# EchoVault — Password Psychology Analyzer

Built at hackUMBC Mini Hackathon — April 2026

---

## What It Does

EchoVault takes a set of passwords and analyzes the psychological patterns behind them. Most password checkers evaluate a single password in isolation. EchoVault looks at multiple passwords together to identify the unconscious habits a person repeats — the same base words, appended years, predictable substitutions — and shows what a targeted attacker could infer from them.

---

## How It Works

The user enters 3–5 past passwords. The backend runs two processes:

1. A trained Random Forest Classifier scores each password from 0–100 based on extracted features like length, entropy, character diversity, and pattern flags.
2. Those scores are passed to the Google Gemini API, which analyzes the passwords for psychological patterns and generates a plain-English fingerprint summary, a predicted attacker wordlist, and personalized tips.

Results are displayed on a results page with a crackability score per password and a vulnerability breakdown across five categories.

---

## Tech Stack

- Frontend: HTML, CSS, JavaScript
- Backend: Python, Flask
- AI: Google Gemini 2.0 Flash
- ML: scikit-learn Random Forest Classifier
- Dataset: Xato 10-million: 10,000 most common passwords 

---

## Setup

### 1. Clone the repo
```bash
git clone https://github.com/yourusername/Echo-Vault.git
cd Echo-Vault
```

### 2. Install dependencies
```bash
py -3 -m pip install -r requirements.txt
```

### 3. Create a .env file in the root folder
```
GEMINI_API_KEY=your_key_here
```

### 4. Run the backend
```bash
py -3 backend/app.py
or
py -3 app.py (if in backend directory)
```

### 5. Open the frontend

Option 1: Open `index.html` with localhost on your browser.

Option 2: Download Live Server and Open 'index.html' with Live Server

---

## The ML Model

The scoring model is a Random Forest Classifier trained on 5,000 weak passwords from the Xato dataset and 5,000 programmatically generated strong passwords. It extracts 13 features per password including length, entropy, presence of years, keyboard patterns, symbol substitutions, and character diversity. It outputs a weak probability which is converted to a 0–100 crackability score.

The model is pre-trained and saved as `password_model.pkl` so it loads instantly without retraining on each run.

---

## API

### POST /analyze

Request:
```json
{
  "passwords": ["fluffy2015", "Fluffy@2016", "ilovefluffy99"]
}
```

Response:
```json
{
  "scores": [
    { "password": "fl********", "score": 100, "crackability": "Critical" }
  ],
  "ai_report": {
    "fingerprint_summary": "...",
    "patterns": ["..."],
    "attacker_wordlist": ["..."],
    "vulnerability_scores": {
      "dictionary_words": 9,
      "years_and_dates": 8,
      "symbol_substitution": 6,
      "keyboard_patterns": 0,
      "personal_references": 9
    },
    "tips": ["..."]
  }
}
```

---

## Team

| Name | Role |
|---|---|
| Anugrah Zachary| Backend & Gemini API |
| Nahom Kassaye | Frontend |
| Michael Togbe | Machine Learning & DS |

---

## Notes

Passwords are never stored. They exist in memory only for the duration of the request and are masked in all outputs.

---

MIT License