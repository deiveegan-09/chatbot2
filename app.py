import os
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from google import genai
from google.genai import types

from chatbot_config import SYSTEM_PROMPT

load_dotenv()

app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False

API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_NAME = "gemini-3.1-flash-lite"

client = genai.Client(api_key=API_KEY) if API_KEY else None


@app.get("/")
def home():
    return render_template("index.html")


@app.post("/chat")
def chat():
    data = request.get_json(silent=True) or {}
    message = (data.get("message") or "").strip()

    if not message:
        return jsonify({"error": "Please enter a message."}), 400

    if client is None:
        return jsonify({"error": "Gemini API key is not configured."}), 500

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=message,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.6,
                max_output_tokens=800,
            ),
        )
        answer = response.text or "I couldn't generate a response. Please try again."
        return jsonify({"reply": answer})
    except Exception:
        app.logger.exception("Gemini request failed")
        return jsonify({"error": "Something went wrong while generating a reply. Please try again."}), 502


if __name__ == "__main__":
    app.run(debug=os.getenv("FLASK_DEBUG", "false").lower() == "true")
