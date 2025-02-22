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
    Sends chunk_text to a specialized OpenAI model (>=1.0.0) 
    to produce a single-sentence prompt or 'NOT RELEVANT'.

    - If chunk_text is empty, we skip calling the API.
    - If the model returns an empty response, we fallback to "NO RESPONSE FROM OPENAI".
    - No temperature or unsupported parameters are included.
    """
    result = {
        "chunk_text": chunk_text.strip(),
        "model": "o3-mini",  # or "gpt-3.5-turbo" if "o3-mini" isn't available
        "messages": [
            {
                "role": "system",
                "content": (
                    "You analyze this chunk of user speech. Provide a single-sentence short prompt "
                    "to investigate that chunk further, or 'NOT RELEVANT' if it is irrelevant. "
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

    # If there's no text, we won't call OpenAI
    if not chunk_text.strip():
        return result

    try:
        print(f"[LOG] Sending chunk to OpenAI: '{chunk_text}'")
        # Make sure your model supports max_completion_tokens, remove if not.
        response = openai.chat.completions.create(
            model=result["model"],
            messages=result["messages"],
            max_completion_tokens=50  # Keep responses short
        )

        content = response.choices[0].message.content.strip()
        print(f"[LOG] Raw OpenAI response: '{content}'")

        # If the model returns an empty string, fallback
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
