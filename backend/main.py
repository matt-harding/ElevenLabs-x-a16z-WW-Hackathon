#!/usr/bin/env python3

import re
import spacy
import pandas as pd

def main():
    """
    Reads a transcript from 'transcript.txt', line-by-line,
    and converts it into a DataFrame manually (one line per row).
    Then identifies lines for Chris Dixon and David George,
    accumulates their text, and uses spaCy to extract named entities.
    """

    # Load the small English model from spaCy
    # (Ensure you've installed it: python -m spacy download en_core_web_sm)
    nlp = spacy.load("en_core_web_sm")

    # 1. Manually read transcript.txt and store lines in a list
    #    This bypasses the CSV parser limitations on "\n".
    with open("transcript.txt", "r", encoding="utf-8") as f:
        lines = f.read().splitlines()

    # 2. Create a DataFrame with a single column "line"
    df = pd.DataFrame(lines, columns=["line"])
    print("=== RAW DATAFRAME ===")
    print(df)
    print()

    # 3. Prepare data structures to store speaker text
    speakers = {
        "Chris Dixon": [],
        "David George": [],
        "Unattributed": []  # Add storage for unattributed lines
    }

    # We'll keep track of the current speaker as we move through lines
    current_speaker = None

    # 4. Regex to detect lines that explicitly start with either speaker
    #    Examples: "Chris Dixon: text...", "CHRIS DIXON - text", etc.
    speaker_pattern = re.compile(
        r'^\s*(chris\s+dixon|david\s+george)\s*[:>\-–—]?\s*(.*)',
        re.IGNORECASE
    )

    # 5. Iterate through each line in the dataframe
    for _, row in df.iterrows():
        text_line = row["line"].strip()

        # Check if the line explicitly starts with a speaker name
        match = speaker_pattern.match(text_line)
        if match:
            # If there's a match, this line defines a new current speaker
            speaker_name = match.group(1).title()  # Normalize to title case
            if "chris" in speaker_name.lower():
                current_speaker = "Chris Dixon"
            else:
                current_speaker = "David George"

            # The rest of the text after the delimiter
            speaker_text = match.group(2).strip()
            # Accumulate the text for this speaker
            speakers[current_speaker].append(speaker_text)
        else:
            # If there's no match, it's either a continuation or an unattributed line
            if current_speaker:
                speakers[current_speaker].append(text_line)
            else:
                # Store unattributed lines separately instead of ignoring them
                speakers["Unattributed"].append(text_line)

    # 6. Join all lines for each speaker into one big string
    chris_text = " ".join(speakers["Chris Dixon"])
    david_text = " ".join(speakers["David George"])
    unattributed_text = " ".join(speakers["Unattributed"])

    # 7. Run spaCy Named Entity Recognition on each speaker's text
    chris_doc = nlp(chris_text)
    david_doc = nlp(david_text)

    # 8. Extract entities for each speaker
    chris_entities = [(ent.text, ent.label_) for ent in chris_doc.ents]
    david_entities = [(ent.text, ent.label_) for ent in david_doc.ents]

    # 9. Print the results
    print("=== Chris Dixon's Text ===")
    print(chris_text)
    print("\n=== Chris Dixon's Entities ===")
    for ent_text, ent_label in chris_entities:
        print(f"{ent_text} -> {ent_label}")

    print("\n" + "="*40 + "\n")

    print("=== David George's Text ===")
    print(david_text)
    print("\n=== David George's Entities ===")
    for ent_text, ent_label in david_entities:
        print(f"{ent_text} -> {ent_label}")

    # Output for unattributed text (if any)
    if unattributed_text:
        print("\n" + "="*40 + "\n")
        print("=== Unattributed Text ===")
        print(unattributed_text)


if __name__ == "__main__":
    main()
