import os
import openai
from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")

app = Flask(__name__)
openai.api_key = OPENAI_API_KEY

def call_perplexity(query: str):
    """
    Calls Perplexity using the openai library with base_url="https://api.perplexity.ai".
    Returns the text content or None on error.
    """
    if not PERPLEXITY_API_KEY:
        print("[ERROR] Missing PERPLEXITY_API_KEY.")
        return None

    from openai import OpenAI
    client = OpenAI(api_key=PERPLEXITY_API_KEY, base_url="https://api.perplexity.ai")

    messages = [
        {
            "role": "system",
            "content": "You are an assistant that retrieves relevant resources for the user's query."
        },
        {
            "role": "user",
            "content": query
        }
    ]

    try:
        print(f"[LOG] Sending query to Perplexity: {query}")
        response = client.chat.completions.create(
            model="sonar-pro",  # or whichever Perplexity model is correct
            messages=messages
        )
        content = response.choices[0].message.content.strip()
        print(f"[LOG] Perplexity response: {content}")
        return content
    except Exception as e:
        print("[ERROR] Perplexity call error:", e)
        return None

def decide_and_search(user_text: str):
    """
    1) Uses OpenAI (o3-mini) to see if user_text is 'NOT RELEVANT' 
       or produce a short search query capturing the main topic.
    2) If relevant, calls Perplexity with that query.
    Returns:
       {
         "is_relevant": bool,
         "query": str,
         "perplexity_result": str or None
       }
    """
    result = {
        "is_relevant": False,
        "query": "",
        "perplexity_result": None
    }

    # If user text is empty or just whitespace
    if not user_text.strip():
        print("[LOG] Empty user text. No analysis done.")
        return result

    print(f"[LOG] Checking relevancy for text: {user_text}")
    try:
        openai_response = openai.ChatCompletion.create(
            model="o3-mini",  # or "gpt-3.5-turbo" if "o3-mini" is not available
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You decide if user text is relevant enough to investigate further. "
                        "If it's not relevant, respond with 'NOT RELEVANT'. "
                        "Otherwise, respond with a short search query capturing the main topic."
                    )
                },
                {
                    "role": "user",
                    "content": user_text
                }
            ]
        )
        content = openai_response.choices[0].message.content.strip()
        print(f"[LOG] OpenAI (o3-mini) responded: {content}")

        if content.lower().startswith("not relevant"):
            print("[LOG] Marked as NOT RELEVANT by OpenAI.")
            return result
        else:
            # It's relevant
            result["is_relevant"] = True
            result["query"] = content

    except Exception as e:
        print("[ERROR] OpenAI error:", e)
        return result

    if result["is_relevant"]:
        px_content = call_perplexity(result["query"])
        if px_content:
            result["perplexity_result"] = px_content

    return result

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/process-chunk", methods=["POST"])
def process_chunk():
    """
    Receives small chunk of text from last 3 seconds of speech:
        { "chunk": "<some partial transcript>" }
    Returns JSON with is_relevant, query, perplexity_result (if relevant).
    """
    data = request.json
    chunk_text = data.get("chunk", "").strip()
    print(f"[LOG] /api/process-chunk called. chunk_text={chunk_text}")

    analysis = decide_and_search(chunk_text)
    response_json = {
        "is_relevant": analysis["is_relevant"],
        "query": analysis["query"],
        "perplexity_result": analysis["perplexity_result"]
    }
    print(f"[LOG] Chunk analysis result: {response_json}")
    return jsonify(response_json)

if __name__ == "__main__":
    app.run(debug=True, port=5000)
