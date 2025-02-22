#!/usr/bin/env python3

import re
import spacy
import pandas as pd
from collections import defaultdict

def extract_topics(text, nlp, top_n=5):
    """
    Given a chunk of text and a spaCy model, return a naive list of potential topics.
    This approach:
      - Parses text with spaCy
      - Collects frequent NOUN and PROPN lemmas (excluding stop words)
      - Returns the top N most frequent
    """
    doc = nlp(text)
    freq = defaultdict(int)

    for token in doc:
        # Check if it's a noun (NOUN or PROPN), and not a stopword or punctuation
        if token.pos_ in ("NOUN", "PROPN") and not token.is_stop and not token.is_punct:
            # Use the lowercase lemma for frequency counting
            freq[token.lemma_.lower()] += 1

    # Sort by descending frequency
    sorted_terms = sorted(freq.items(), key=lambda x: x[1], reverse=True)

    # Return the top_n lemmas as potential topics
    return [term for term, count in sorted_terms[:top_n]]

def main():
    """
    Reads 'transcript.txt' line by line.
    Detects any speaker name generically by matching "<Speaker>: <Text>".
    Groups subsequent lines under the same speaker until the next speaker appears.
    Extracts naive topics for each speaker's chunk of text using spaCy.
    """

    # 1. Load a spaCy model for English
    nlp = spacy.load("en_core_web_sm")

    # 2. Manually read transcript.txt and store lines in a list
    with open("transcript.txt", "r", encoding="utf-8") as f:
        lines = f.read().splitlines()

    # 3. Create a structure to hold each "turn" = one chunk of text from a single speaker
    #    We'll store a list of dicts, each with {"speaker": str, "lines": [str, str, ...]}
    turns = []
    current_speaker = None
    current_lines = []

    # Regex to match: SomeSpeakerName: text...
    speaker_pattern = re.compile(r'^(.+?):\s*(.*)')

    # 4. Build the list of speaker turns
    for line in lines:
        line = line.strip()
        match = speaker_pattern.match(line)

        if match:
            # Found a new speaker, so first save the previous chunk (if any)
            if current_speaker is not None:
                turns.append({
                    "speaker": current_speaker,
                    "text": " ".join(current_lines)
                })

            # Start a new speaker chunk
            current_speaker = match.group(1).strip()
            first_text = match.group(2).strip()
            current_lines = [first_text] if first_text else []

        else:
            # No speaker pattern; this line continues the current speaker's chunk
            if current_speaker is not None:
                current_lines.append(line)
            else:
                # If we haven't encountered a speaker yet, we could handle it as "Unattributed"
                # or skip. Let's just skip or store in "Unknown"
                if len(turns) == 0 or turns[-1].get("speaker") != "Unknown":
                    turns.append({"speaker": "Unknown", "text": line})
                else:
                    turns[-1]["text"] += " " + line

    # After the loop, if we still have an unfinished chunk, append it
    if current_speaker is not None:
        turns.append({
            "speaker": current_speaker,
            "text": " ".join(current_lines)
        })

    # 5. Now let's do spaCy-based topic extraction for each speaker chunk
    for turn in turns:
        speaker_name = turn["speaker"]
        chunk_text = turn["text"]
        if not chunk_text.strip():
            continue  # Skip empty chunks

        # Identify potential topics
        topics = extract_topics(chunk_text, nlp, top_n=5)

        # Print results
        print(f"--- Speaker: {speaker_name} ---")
        print(chunk_text)
        print("Potential Topics:", topics)
        print()

if __name__ == "__main__":
    main()
