import os
import asyncio
import openai
from openai import OpenAI  # For custom Perplexity usage
from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv

load_dotenv()

# ENV VARS
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("Missing OPENAI_API_KEY in .env")
if not PERPLEXITY_API_KEY:
    raise ValueError("Missing PERPLEXITY_API_KEY in .env")

app = Flask(__name__)

# 1) Standard OpenAI config for GPT chunk analysis:
openai.api_key = OPENAI_API_KEY

# 2) Perplexity client using the snippet you provided:
#    from openai import OpenAI
perplexity_client = OpenAI(api_key=PERPLEXITY_API_KEY, base_url="https://api.perplexity.ai")


def analyze_text_chunk(chunk_text: str) -> dict:
    """
    Sends chunk_text to 'gpt-4o' to produce a single-sentence prompt or 'NOT RELEVANT'.
    If 'gpt-4o' fails (model not found), fallback to 'gpt-3.5-turbo'.
    """
    result = {
        "chunk_text": chunk_text.strip(),
        "model": "gpt-4o",
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
        return result

    try:
        print(f"[LOG] Sending chunk to OpenAI using model='{result['model']}': '{chunk_text}'")
        response = None
        try:
            # Attempt GPT-4o
            response = openai.chat.completions.create(
                model=result["model"],
                messages=result["messages"],
                max_tokens=50
            )
        except openai.error.InvalidRequestError as e:
            # If GPT-4o doesn't exist, fallback to GPT-3.5-turbo
            if "The model `gpt-4o` does not exist" in str(e):
                print(f"[WARN] {e} - Falling back to 'gpt-3.5-turbo'")
                result["model"] = "gpt-3.5-turbo"
                response = openai.chat.completions.create(
                    model=result["model"],
                    messages=result["messages"],
                    max_tokens=50
                )
            else:
                # Another error
                raise e

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
    """
    Existing route for your chunk-based GPT-4o / GPT-3.5 fallback analysis.
    """
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


@app.route("/api/perplexity-chat", methods=["POST"])
def perplexity_chat():
    """
    Non-streaming conversation with Perplexity using the 'sonar-pro' model.
    The user prompt is in 'prompt'. This calls Perplexity and returns the entire response.
    """
    data = request.json or {}
    user_prompt = data.get("prompt", "").strip()
    if not user_prompt:
        return jsonify({"error": "Please provide a prompt."}), 400

    messages = [
        {
            "role": "system",
            "content": (
                "You are an artificial intelligence assistant and you need to engage in "
                "a helpful, detailed, polite conversation with a user."
            ),
        },
        {
            "role": "user",
            "content": user_prompt,
        },
    ]

    try:
        # Non-streaming call
        response = perplexity_client.chat.completions.create(
            model="sonar-pro",
            messages=messages
        )
        # Typically, you'd get the text from response.choices[0].message.content
        content = response.choices[0].message.content
        return jsonify({"perplexity_response": content})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/perplexity-chat-stream", methods=["POST"])
def perplexity_chat_stream():
    """
    Streaming conversation with Perplexity. The user prompt is in 'prompt'.
    We'll stream the tokens in real-time back to the client using Flask's generator response.
    """
    data = request.json or {}
    user_prompt = data.get("prompt", "").strip()
    if not user_prompt:
        return jsonify({"error": "Please provide a prompt."}), 400

    messages = [
        {
            "role": "system",
            "content": (
                "You are an artificial intelligence assistant and you need to engage in "
                "a helpful, detailed, polite conversation with a user."
            ),
        },
        {
            "role": "user",
            "content": user_prompt,
        },
    ]

    def token_generator():
        try:
            response_stream = perplexity_client.chat.completions.create(
                model="sonar-pro",
                messages=messages,
                stream=True,  # streaming
            )
            for chunk in response_stream:
                # Each chunk might look like: {"choices": [{"delta": {"content": "..."}}]}
                yield f"{chunk}\n"
        except Exception as e:
            yield f"Error: {str(e)}\n"

    # Return a streaming (chunked) response
    return app.response_class(token_generator(), mimetype="text/plain")


if __name__ == "__main__":
    app.run(debug=True, port=5000)
