import os
import openai
from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("Missing OPENAI_API_KEY in environment variables (.env).")

app = Flask(__name__)

# Set the API key
openai.api_key = OPENAI_API_KEY

def analyze_text_chunk(chunk_text: str) -> dict:
    """
    Sends chunk_text to 'gpt-4o' to produce a single-sentence prompt or 'NOT RELEVANT'.
    - If chunk_text is empty, we skip calling the API.
    - If the model returns an empty response, fallback to 'NO RESPONSE FROM OPENAI'.
    - Includes a fallback to 'gpt-3.5-turbo' if 'gpt-4o' doesn't exist in your org.
    """
    result = {
        "chunk_text": chunk_text.strip(),
        "model": "gpt-4o",  # Attempt to use this custom model
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

    if not chunk_text.strip():
        # No text, just return the default structure
        return result

    try:
        print(f"[LOG] Sending chunk to OpenAI using model='{result['model']}': '{chunk_text}'")
        response = None
        try:
            # Primary attempt: gpt-4o
            response = openai.chat.completions.create(
                model=result["model"],
                messages=result["messages"],
                max_tokens=50
            )
        except openai.error.InvalidRequestError as e:
            # If gpt-4o doesn't exist, fallback to gpt-3.5-turbo
            if "The model `gpt-4o` does not exist" in str(e):
                print(f"[WARN] {e} - Falling back to 'gpt-3.5-turbo' ...")
                result["model"] = "gpt-3.5-turbo"
                response = openai.chat.completions.create(
                    model=result["model"],
                    messages=result["messages"],
                    max_tokens=50
                )
            else:
                # Some other error
                raise e

        # Extract the assistant's reply
        content = response.choices[0].message.content.strip()
        print(f"[LOG] Raw OpenAI response: '{content}'")

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
    # Run the Flask app on port 5000
    app.run(debug=True, port=5000)
