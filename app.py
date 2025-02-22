import os
import openai
from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

app = Flask(__name__)
openai.api_key = OPENAI_API_KEY

def analyze_text_chunk(chunk_text: str) -> dict:
    """
    Sends chunk_text to your specialized OpenAI model (>=1.0.0)
    to produce a single-sentence prompt or 'NOT RELEVANT'.

    We forcibly instruct the model to:
      - Never return an empty string
      - Always return EITHER 'NOT RELEVANT' or exactly one sentence
    """
    result = {
        "chunk_text": chunk_text,
        "model": "o3-mini",  # or "gpt-3.5-turbo" if "o3-mini" isn't available
        "messages": [
            {
                "role": "system",
                "content": (
                    "You analyze this chunk of user speech. "
                    "You MUST respond with exactly one short sentence that can serve as a prompt to investigate further, "
                    "OR respond with 'NOT RELEVANT' if it truly is irrelevant. "
                    "Never return an empty answer."
                )
            },
            {
                "role": "user",
                "content": chunk_text.strip()
            }
        ],
        "openai_response": ""
    }

    if not chunk_text.strip():
        return result

    try:
        print(f"[LOG] Sending chunk to OpenAI: '{chunk_text}'")

        # This model doesn't support 'temperature' or 'max_tokens'; 
        # it wants 'max_completion_tokens' but no 'temperature'.
        response = openai.chat.completions.create(
            model=result["model"],
            messages=result["messages"],
            max_completion_tokens=50  # limit to keep response short
        )

        content = response.choices[0].message.content.strip()
        print(f"[LOG] Raw OpenAI response: '{content}'")

        # If the model somehow returns an empty string, we fallback
        if not content:
            content = "NO RESPONSE FROM OPENAI"

        result["openai_response"] = content
    except Exception as e:
        print("[ERROR] OpenAI error:", e)
        result["openai_response"] = f"OpenAI Error: {str(e)}"

    return result

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/process-chunk", methods=["POST"])
def process_chunk():
    data = request.json
    chunk_text = data.get("chunk", "")
    print(f"[LOG] /api/process-chunk chunk='{chunk_text}'")

    analysis = analyze_text_chunk(chunk_text)
    return jsonify({
        "chunk_text": analysis["chunk_text"],
        "model": analysis["model"],
        "messages": analysis["messages"],
        "openai_response": analysis["openai_response"]
    })

if __name__ == "__main__":
    app.run(debug=True, port=5000)
