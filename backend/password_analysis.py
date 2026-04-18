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
from sklearn.ensemble import RandomForestClassifier 
from sklearn.preprocessing import StandardScaler 
import joblib
import os 


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

    #weak passwords from the file
    try:
        with open("data/xato-net-10-million-passwords-10000.txt", "r") as f:
            weak = [line.strip() for line in f if len(line.strip()) >= 4]
    except FileNotFoundError:
        weak = []

    for p in weak[:5000]:
        X.append(extract_features(p))
        y.append(1) # 1 = weak password

    #strong passwords generated 
    for _ in range(5000):
        p = generate_strong_password()
        X.append(extract_features(p))
        y.append(0) # 0 = strong password

    return np.array(X), np.array(y)


#-----------Train/getmodel------------#
def get_model():
    model_path = "data/password_model.pkl"
    scaler_path = "data/scaler.pkl"

    if os.path.exists(model_path):
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        print("Model loaded from disk.")
    else:
        print("Training Model...")
        X, y = build_dataset()

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )

        model.fit(X_scaled,y)

        joblib.dump(model, model_path)
        joblib.dump(scaler, scaler_path)
        print("Model has been trained and saved.")

    return model, scaler

#-----------Model Scoring------------#

def score_passwords(passwords: list[str]) -> list[dict]:
    model, scaler = get_model()
    results = []

    for password in passwords:
        features = np.array([extract_features(password)])
        features_scaled = scaler.transform(features)


        #Probability of being weak (0 - 1) -> convert to 0-100 score
        weak_probability = model.predict_proba(features_scaled)[0][1]
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