#!/usr/bin/env python3

import json
from openai import OpenAI  # Using the openai python package with a custom base_url if needed

def load_analysis_results(json_file="analysis_output.json"):
    """
    Load the transcript analysis results from JSON.
    Returns a list of 'turns', each containing:
       "speaker", "text", "entities", "topics", "key_phrases", etc.
    """
    try:
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("turns", [])
    except FileNotFoundError:
        print(f"Could not find {json_file}. Please run main.py first.")
        return []

def call_openai_for_entity_check(entity, openai_api_key):
    """
    Use an OpenAI LLM (e.g. 'o3-mini') to decide if 'entity' is interesting
    to investigate further. Return the short query or 'Not relevant.' if not interesting.
    """
    # Standard OpenAI client (default https://api.openai.com)
    client = OpenAI(api_key=openai_api_key)

    messages = [
        {
            "role": "system",
            "content": (
                "You are an AI assistant that helps decide if an entity is interesting "
                "to investigate further in a podcast. If yes, provide a short search query. "
                "If no, say 'Not relevant.'"
            ),
        },
        {
            "role": "user",
            "content": (
                f"Entity: {entity}\n"
                "Is this entity interesting to investigate further? "
                "If yes, provide a short search query. If not, say 'Not relevant.'"
            )
        }
    ]

    try:
        response = client.chat.completions.create(
            model="o3-mini",  # Replace with a valid model (e.g., "gpt-3.5-turbo")
            messages=messages
        )
        # Extract the content from the assistant's reply
        content = response.choices[0].message.content.strip()
        return content
    except Exception as e:
        print(f"Error calling OpenAI for entity check: {e}")
        return "Not relevant."

def call_perplexity(query, perplexity_api_key):
    """
    Make a single call to Perplexity using the OpenAI client with a custom base_url.
    The returned response is the entire JSON from Perplexity.
    """
    # Create OpenAI client for Perplexity
    client = OpenAI(api_key=perplexity_api_key, base_url="https://api.perplexity.ai")

    messages = [
        {
            "role": "system",
            "content": (
                "You are an AI assistant that retrieves relevant resources "
                "like articles, YouTube videos, or images about a given query."
            ),
        },
        {
            "role": "user",
            "content": query,
        },
    ]

    print(f"\n[Querying Perplexity]: {query}")
    try:
        response = client.chat.completions.create(
            model="sonar-pro",
            messages=messages
        )
        return response
    except Exception as e:
        print(f"Error calling Perplexity: {e}")
        return None

def main():
    """
    1) Load transcript analysis results from 'analysis_output.json'.
    2) For each turn:
       - Pick the first entity found (if any).
       - Ask an OpenAI LLM if it's relevant; if yes, return a short query, else 'Not relevant.'
       - Let user choose to call Perplexity with that query or skip.
    """

    # 1) Load the analysis
    turns = load_analysis_results("analysis_output.json")
    if not turns:
        return

    print("=== Single-Query Searcher ===")
    print("We'll walk through each chunk, pick the first entity, and see if it's interesting.\n")

    # 2) Prompt for OpenAI API key (for entity check)
    openai_api_key = input("Enter your OpenAI API key (or press enter to skip entity checks): ").strip()
    if not openai_api_key:
        print("No OpenAI API key provided. Exiting since we can't do entity check.\n")
        return

    # 3) Prompt for Perplexity API key (for actual searching)
    perplexity_api_key = input("Enter your Perplexity API key (or press enter to skip perplexity searches): ").strip()
    if not perplexity_api_key:
        print("No Perplexity API key provided. We'll just do entity checks, no searching.\n")

    # 4) Iterate each turn
    for idx, turn in enumerate(turns):
        speaker = turn.get("speaker", "Unknown")
        text = turn.get("text", "")
        snippet = (text[:60] + "...") if len(text) > 60 else text

        print(f"--- Turn #{idx+1} | Speaker: {speaker} | Text: {snippet}")

        # Check if there's at least one entity
        entity_dict = turn.get("entities", {})
        all_entities = []
        for ent_list in entity_dict.values():
            all_entities.extend(ent_list)
        if not all_entities:
            print("No entities found in this turn. Skipping.")
            print("-" * 50)
            continue

        # Use the FIRST entity for a single query approach
        chosen_entity = all_entities[0]
        print(f"First entity in this turn: {chosen_entity}")

        # Ask LLM if it's relevant
        llm_result = call_openai_for_entity_check(chosen_entity, openai_api_key)
        print(f"LLM Response: {llm_result}")

        if llm_result.lower().startswith("not relevant"):
            print("Skipping because the LLM says it's not interesting.")
            print("-" * 50)
            continue

        # We got a single short query from the LLM
        final_query = llm_result

        # If we do have a Perplexity key, let user choose to search
        if perplexity_api_key:
            choice = input("Do you want to perform a Perplexity search for this query? (y/n): ").lower().strip()
            if choice.startswith("y"):
                response = call_perplexity(final_query, perplexity_api_key)
                if response:
                    print("=== Perplexity Response ===")
                    print(response)
                else:
                    print("No response or error.")
        else:
            print("(No Perplexity key provided, skipping search.)")

        print("-" * 50)

    print("\nAll done!\n")


if __name__ == "__main__":
    main()
