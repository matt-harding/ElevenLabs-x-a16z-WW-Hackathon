import os
import tempfile
import openai
import requests
from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv
from faster_whisper import WhisperModel

# Load environment variables (OPENAI_API_KEY, PERPLEXITY_API_KEY, etc.)
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")

app = Flask(__name__)
openai.api_key = OPENAI_API_KEY

# Example: Initialize local Whisper model (uncomment and adjust as needed)
# whisper_model = WhisperModel("small", device="cpu", compute_type="int8")


def transcribe_audio_with_whisper(audio_path: str) -> str:
    """
    Uses the local faster-whisper model to transcribe an audio file.
    Returns the final transcript as a string.
    """
    # Uncomment and adapt as needed if you're using whisper_model
    # segments, info = whisper_model.transcribe(audio_path, beam_size=5)
    # parts = [segment.text for segment in segments]
    # transcript = " ".join(parts)
    # return transcript.strip()

    # For testing/demo purposes, returning static text
    return "Last month, Meta which owns popular social media platforms Facebook and Instagram, sent a stern warning to employees that it was planning to cut roughly 3,600 jobs."


def check_relevancy_openai(text: str):
    """
    Uses OpenAI Chat API to decide if 'text' is relevant enough to investigate further.
    Returns (bool is_relevant, str short_query).
    """
    if not text.strip():
        return False, ""

    try:
        # Must use ChatCompletion for chat-based models (like 'o3-mini' or 'gpt-3.5-turbo')
        response = openai.ChatCompletion.create(
            model="o3-mini",  # Replace with a valid model name in your OpenAI account
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You decide if user text is relevant enough to investigate further. "
                        "If it's not relevant, respond with 'NOT RELEVANT'. "
                        "Otherwise, respond with a short search query."
                    )
                },
                {
                    "role": "user",
                    "content": text
                }
            ]
        )

        # Extract the AI's response content
        content = response.choices[0].message.content.strip()
        print("CHECKING RELEVANCY")
        print(content)

        # Simple logic: if it starts with "NOT RELEVANT", we consider it not relevant
        if content.lower().startswith("not relevant"):
            return False, ""
        else:
            return True, content

    except Exception as e:
        print("OpenAI relevancy check error:", e)
        return False, ""


def call_perplexity(query: str):
    """
    Calls Perplexity using the openai library with base_url="https://api.perplexity.ai".
    Returns the text content or None on error.
    """
    if not PERPLEXITY_API_KEY:
        return None

    from openai import OpenAI
    client = OpenAI(api_key=PERPLEXITY_API_KEY, base_url="https://api.perplexity.ai")

    messages = [
        {
            "role": "system",
            "content": "You a friend assistant that retrieves relevant resources to the user's query."
        },
        {
            "role": "user",
            "content": query
        }
    ]

    try:
        response = client.chat.completions.create(
            model="sonar-pro",
            messages=messages
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print("Perplexity call error:", e)
        return None


@app.route("/")
def index():
    # Renders an index.html template (make sure you have this in your templates folder)
    return render_template("index.html")


@app.route("/api/transcribe-audio", methods=["POST"])
def transcribe_audio():
    """
    Receives the recorded audio file from the browser, 
    saves it to a temp file, runs faster-whisper for transcription,
    checks relevancy with OpenAI, and returns JSON.
    """
    if not OPENAI_API_KEY:
        return jsonify({"error": "Missing OPENAI_API_KEY"}), 500

    audio_file = request.files.get("audio")
    if not audio_file:
        return jsonify({"error": "No audio file found"}), 400

    # Save to temp for transcription
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        audio_path = tmp.name
        audio_file.save(audio_path)

    try:
        # Transcribe using local faster-whisper
        transcript = transcribe_audio_with_whisper(audio_path)
    except Exception as e:
        print("Error transcribing audio:", e)
        return jsonify({"error": "Transcription failed"}), 500
    finally:
        # Cleanup temp file
        if os.path.exists(audio_path):
            os.remove(audio_path)

    # Check relevancy with OpenAI
    is_relevant, suggested_query = check_relevancy_openai(transcript)

    return jsonify({
        "transcript": transcript,
        "is_relevant": is_relevant,
        "suggested_query": suggested_query
    })


@app.route("/api/perplexity-search", methods=["POST"])
def perplexity_search():
    # Example route to call the Perplexity-like endpoint
    print("PERPLEXITY endpoint called.")
    data = request.json
    query = data.get("query", "").strip()
    if not query:
        return jsonify({"error": "Empty query"}), 400

    result = call_perplexity(query)
    print("RESULT FROM PERPLEXITY:")
    print(result)
    if result is None:
        return jsonify({"error": "Perplexity error or missing key"}), 500

    return jsonify({"content": result})


if __name__ == "__main__":
    # Run Flask app
    app.run(debug=True, port=5000)
