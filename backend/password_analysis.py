# -------------------------------------------------------------
# EchoVault — Password Analysis & ML Scoring w/ Random Forest clf
# Author: Michael Togbe
# HackUMBC Mini Hackathon — April 2026
# -------------------------------------------------------------

import re
import math
import random 
import string
import numpy as np 
from pathlib import Path
from typing import Optional
from sklearn.ensemble import RandomForestClassifier 
from sklearn.preprocessing import StandardScaler 
import joblib
import os 

_BACKEND_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _BACKEND_DIR.parent

# Wordlist ships at repo `data/`; Flask cwd is `backend/`, so a bare `data/` path misses it.
def _wordlist_path() -> Optional[Path]:
    candidates = [
        _PROJECT_ROOT / "data" / "xato-net-10-million-passwords-10000.txt",
        _BACKEND_DIR / "data" / "xato-net-10-million-passwords-10000.txt",
    ]
    for p in candidates:
        if p.is_file():
            return p
    return None


#-----------Feature Extraction Fn -----------#
def extract_features(password: str) -> list:
    length = len(password)

    has_upper = int(bool(re.search(r'[A-Z]', password)))
    has_lower = int(bool(re.search(r'[a-z]', password)))
    has_digit = int(bool(re.search(r'[0-9]', password)))
    has_symbol = int(bool(re.search(r'[^a-zA-Z0-9]', password)))

    #Entropy calculation
    charset = 0 
    if has_upper: charset +=26
    if has_lower: charset += 26
    if has_digit: charset += 10
    if has_symbol: charset += 32
    entropy = length * math.log2(charset) if charset > 0 else 0

    #Pattern flags 
    has_year = int(bool(re.search(r'(19|20)\d{2}', password)))
    has_keyboard = int(any(w in password.lower()
                           for w in ['qwerty', 'asdf', 'zxcv', '1234', 'qwer']))
    has_repeat = int(bool(re.search(r'(.)\1{2,}', password)))

    substitutions = {'@': 'a', '3' : 'e', '1':'i', '0':'o', '$':'s', '!':'i'}
    base = password.lower()
    for k, v in substitutions.items():
        base = base.replace(k, v)
    has_substitution = int(base != password.lower())

    unique_chars = len(set(password))
    digit_count = sum(c.isdigit() for c in password)
    symbol_count = sum(not c.isalnum() for c in password)

    return [
        length, has_upper, has_lower, has_digit, has_symbol,
        entropy, has_year, has_keyboard, has_repeat, has_substitution,
        unique_chars, digit_count, symbol_count
    ]


#-----------Generates a strong password------------#
def generate_strong_password() -> str:
    length = random.randint(14,24)
    chars = (
        random.choices(string.ascii_uppercase,k=3) +
        random.choices(string.ascii_lowercase, k=5) +
        random.choices(string.digits, k=3) +
        random.choices('!@#$%^&*()_+-=[]{}|;:,.<>?', k=3) +
        random.choices(string.ascii_letters + string.digits,
                       k=length-14)
    )

    random.shuffle(chars)
    return ''.join(chars)

#-----------Loads in file of 10,000 common passwords------------#
def build_dataset():
    X, y = [],[]

    # Weak passwords from the file (path resolved so training works when cwd is backend/)
    weak = []
    wl = _wordlist_path()
    if wl is not None:
        with open(wl, "r", encoding="utf-8", errors="ignore") as f:
            weak = [line.strip() for line in f if len(line.strip()) >= 4]
    if len(weak) < 100:
        # If the file is missing or tiny, mix in common weak strings so labels stay binary.
        seeds = (
            "123456 password 123456789 qwerty abc123 monkey dragon letmein "
            "trustno1 iloveyou welcome admin login passw0rd football baseball"
        ).split()
        seen = set()
        for p in weak + seeds:
            if len(p) >= 4 and p not in seen:
                seen.add(p)
                weak.append(p)

    for p in weak[:5000]:
        X.append(extract_features(p))
        y.append(1) # 1 = weak password

    #strong passwords generated 
    for _ in range(5000):
        p = generate_strong_password()
        X.append(extract_features(p))
        y.append(0) # 0 = strong password

    return np.array(X), np.array(y)


def _model_is_binary(model: RandomForestClassifier) -> bool:
    classes = getattr(model, "classes_", None)
    return classes is not None and len(classes) >= 2


def _train_and_save(model_path: Path, scaler_path: Path):
    X, y = build_dataset()
    if len(np.unique(y)) < 2:
        raise RuntimeError("Training data must include both weak and strong password labels.")

    model_path.parent.mkdir(parents=True, exist_ok=True)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
    model.fit(X_scaled, y)
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    print("Model has been trained and saved.")


#-----------Train/getmodel------------#
def get_model():
    data_dir = _BACKEND_DIR / "data"
    model_path = data_dir / "password_model.pkl"
    scaler_path = data_dir / "scaler.pkl"

    if model_path.is_file() and scaler_path.is_file():
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        if _model_is_binary(model):
            print("Model loaded from disk.")
            return model, scaler
        print("Saved model is invalid (single-class); removing and retraining.")
        try:
            model_path.unlink()
            scaler_path.unlink()
        except OSError:
            pass

    print("Training Model...")
    _train_and_save(model_path, scaler_path)
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

#-----------Model Scoring------------#

def score_passwords(passwords: list[str]) -> list[dict]:
    model, scaler = get_model()
    results = []

    for password in passwords:
        features = np.array([extract_features(password)])
        features_scaled = scaler.transform(features)


        # Probability of being weak (0 - 1) -> convert to 0-100 score
        # If training only saw one class, predict_proba has one column; [0][1] would crash.
        proba_row = model.predict_proba(features_scaled)[0]
        prob_by_class = {int(c): float(proba_row[i]) for i, c in enumerate(model.classes_)}
        weak_probability = prob_by_class.get(1, 0.0)
        score = round(weak_probability * 100)

        results.append({
            "password" : password[:2] + "*" * (len(password) - 2),
            "score": score,
            "crackability": (
                "Critical" if score > 75 else
                "High" if score > 50 else 
                "Medium" if score > 25 else
                "Low"
            )
        })

    return results
    


#Testing 
if __name__ == "__main__":
    test_passwords = [
    # CRITICAL (should be 75-100)
    "letmein",
    "baseball",
    "welcome",
    "shadow",
    "master",
    "666666",

    # HIGH (should be 50-75)
    "Jennifer1995",
    "Dallas2021!",
    "Lakers@2016",
    "Pepper2014",
    "Ranger99!",
    "Chicago2018",

    # MEDIUM (should be 25-50)
    "Wh1skey@Night",
    "Blu3$Berry99",
    "Fr0zen!Lake22",
    "M0untain$Top1",
    "Purpl3@Rain55",

    # LOW (should be 0-25)
    "nP$8vL!3qW#mZx",
    "Ry7@kT!2mX$pQw",
    "Bv#9Lq$3Nz!wKm",
    "Jx$5Rp!8Wn#qTv",
    "Mw@4Yz$7Hk!rNp",
    ]

    print(f"\n{'Password':<35} {'Score':<10} {'Crackability'}")
    print("-" * 60)

    results = score_passwords(test_passwords)
    for r in results:
        print(f"{r['password']:<35} {r['score']:<10} {r['crackability']}")