from flask import Flask, request, jsonify
from flask_cors import CORS
from gemini_service import analyze_passwords

app = Flask(__name__)
CORS(app)


@app.route("/analyze", methods=["POST"])
def analyze():

    incoming_data = request.get_json()
    

    if not incoming_data or "passwords" not in incoming_data:
        return jsonify({"error": "Missing 'passwords' array"}), 400
        
    passwords = incoming_data.get("passwords", [])
    if len(passwords) < 2:
        return jsonify({"error": "Need at least 2 passwords"}), 400

    try:
        
        scores = get_crackability_score(passwords)
        ai_report = analyze_passwords(passwords, scores)
        
        
        return jsonify({
            "scores": scores,
            "ai_report": ai_report
        }), 200
        
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": "Server failed to process request"}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)