import os
import openai
import requests
from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")

app = Flask(__name__)
openai.api_key = OPENAI_API_KEY

def decide_and_search(user_text: str):
    """
    1) Uses OpenAI (o3-mini or similar) to see if the text is 'NOT RELEVANT'
       or return a short search query (topic).
    2) If relevant, calls Perplexity with that query to gather deeper info.
    Returns a dict:
       {
         "is_relevant": bool,
         "query": str,  # short search query from OpenAI
         "perplexity_result": str or None
       }
    """
    result = {
        "is_relevant": False,
        "query": "",
        "perplexity_result": None
    }

    if not user_text.strip():
        print("[LOG] Received empty user text. Returning early.")
        return result

    print(f"[LOG] Received user text for analysis: {user_text}")

    # 1) Check relevancy via OpenAI
    try:
        print("[LOG] Sending text to OpenAI for relevancy check...")
        openai_response = openai.ChatCompletion.create(
            model="o3-mini",  # or "gpt-3.5-turbo" if "o3-mini" is not available
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You analyze user text. If it's not relevant, respond with 'NOT RELEVANT'. "
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
        print(f"[LOG] OpenAI response content: {content}")

        if content.lower().startswith("not relevant"):
            print("[LOG] OpenAI says NOT RELEVANT.")
            return result  # is_relevant remains False
        else:
            result["is_relevant"] = True
            result["query"] = content

    except Exception as e:
        print("[ERROR] OpenAI error:", e)
        return result

    # 2) If relevant, call Perplexity
    if result["is_relevant"] and PERPLEXITY_API_KEY:
        print(f"[LOG] Query is relevant. Now calling Perplexity with query: {result['query']}")
        from openai import OpenAI
        client = OpenAI(api_key=PERPLEXITY_API_KEY, base_url="https://api.perplexity.ai")

        messages = [
            {
                "role": "system",
                "content": "You are an assistant that retrieves relevant research resources for the user's query."
            },
            {
                "role": "user",
                "content": result["query"]
            }
        ]
        try:
            px_response = client.chat.completions.create(
                model="sonar-pro",  # or whichever Perplexity model is correct
                messages=messages
            )
            perplexity_text = px_response.choices[0].message.content.strip()
            print(f"[LOG] Perplexity response content: {perplexity_text}")
            result["perplexity_result"] = perplexity_text
        except Exception as e:
            print("[ERROR] Perplexity call error:", e)
            # We won't override result, just show the error in logs

    return result

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/process-text", methods=["POST"])
def process_text():
    """
    Expects JSON: { "text": "..." }
    1) Decides if it's relevant via OpenAI
    2) If relevant, calls Perplexity
    3) Returns JSON with all needed fields
    """
    data = request.json
    user_text = data.get("text", "").strip()
    print(f"[LOG] /api/process-text called with text: {user_text}")

    analysis = decide_and_search(user_text)

    # Our final response
    response_json = {
        "original_text": user_text,
        "is_relevant": analysis["is_relevant"],
        "query": analysis["query"],
        "perplexity_result": analysis["perplexity_result"]
    }
    print(f"[LOG] Final response JSON: {response_json}")
    return jsonify(response_json)

if __name__ == "__main__":
    app.run(debug=True, port=5000)
