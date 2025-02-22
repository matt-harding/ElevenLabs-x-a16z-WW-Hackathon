#!/usr/bin/env python3

import re
import spacy
from collections import defaultdict, Counter

def extract_frequent_nouns(text, nlp, top_n=5):
    """
    Returns a naive list of single-word 'topics' by:
      1. Processing the text with spaCy to get tokens.
      2. Checking if each token is:
         - A NOUN or PROPN (proper noun),
         - Not a stop word (like 'the', 'and', etc.),
         - Not punctuation.
      3. Lemmatising (converting to a base form) and counting frequency.
      4. Sorting by descending frequency and returning the top N.

    E.g., if text has many mentions of 'cryptocurrencies' and 'investments',
    those might appear as top topics.
    """
    doc = nlp(text)
    freq = defaultdict(int)
    
    for token in doc:
        # Only count if it's a NOUN/PROPN and not a stop word/punct.
        if token.pos_ in ("NOUN", "PROPN") and not token.is_stop and not token.is_punct:
            # Increase count of the token's lemma (lowercased).
            freq[token.lemma_.lower()] += 1

    # Sort our dictionary of {lemma: count} by count (descending).
    sorted_terms = sorted(freq.items(), key=lambda x: x[1], reverse=True)

    # Return just the lemma strings (not counts) up to top_n.
    return [term for term, _count in sorted_terms[:top_n]]

def extract_noun_chunk_phrases(text, nlp, top_n=5):
    """
    Extract frequent multi-word noun chunks from the text, e.g., 'cloud infrastructure'
    or 'machine learning model'. 
      1. spaCy's doc.noun_chunks gives us groups of tokens that form meaningful noun phrases.
      2. We convert them to lowercase, ignoring very short or trivial chunks.
      3. Use a Counter to track frequency and return the top N.

    This often yields more context than single-word topics alone.
    """
    doc = nlp(text)
    chunk_counter = Counter()

    for chunk in doc.noun_chunks:
        # The raw chunk text might have multiple words, e.g. "open source technology".
        chunk_text = chunk.text.strip().lower()
        
        # Skip empty or trivial chunks:
        if not chunk_text or len(chunk_text.split()) < 1:
            continue

        # Count how many times each chunk appears.
        chunk_counter[chunk_text] += 1

    # Get the most common chunk phrases, up to top_n.
    most_common = chunk_counter.most_common(top_n)

    # Return just the phrase portion, ignoring counts.
    return [phrase for (phrase, _count) in most_common]

def group_entities_by_label(doc):
    """
    From a spaCy Doc, collect all named entities into a dictionary:
      {
        "PERSON": ["Alice", "Bob"], 
        "ORG": ["OpenAI"],
        ...
      }
    The doc.ents property yields all recognized entities. We group them
    by their label (PERSON, ORG, GPE, etc.). Each label's value is a list of
    distinct entity strings (sorted alphabetically).
    """
    from collections import defaultdict
    entities_by_label = defaultdict(set)
    for ent in doc.ents:
        # ent.label_ might be 'PERSON', 'ORG', 'GPE', etc.
        entities_by_label[ent.label_].add(ent.text)
    
    # Convert each set to a sorted list for consistency.
    return {label: sorted(ents) for label, ents in entities_by_label.items()}

def extract_entity_subtopics(doc, nlp, top_n=3):
    """
    For each named entity in the Doc, gather subtopics from the sentence(s)
    that contain that entity. This lets us see what other words/nouns
    appear around the entity.

      1. For each entity in doc.ents:
         - Find all sentences where entity's start/end indices fall within that sentence.
         - Combine those sentences into one string 'context'.
      2. Run extract_frequent_nouns on that context to see which nouns
         or propn are mentioned around the entity.
      3. Return a dict like: { "OpenAI": ["research", "investment"], ... }
         indicating subtopics for each entity.
    """
    entity_subtopics = {}
    for ent in doc.ents:
        # We'll gather sentences that contain this entity.
        sentence_texts = []
        for sent in doc.sents:
            # If the entity start/end indices are within a sentence, 
            # that sentence is relevant context for the entity.
            if ent.start >= sent.start and ent.end <= sent.end:
                sentence_texts.append(sent.text)
        
        # Combine relevant sentences into one chunk.
        context = " ".join(sentence_texts).strip()
        if context:
            # Re-use our single-word topic extraction to get subtopics.
            subtopics = extract_frequent_nouns(context, nlp, top_n=top_n)
            entity_subtopics[ent.text] = subtopics
    
    return entity_subtopics

def main():
    """
    Orchestrates the analysis:
      1. Read transcript lines from 'transcript.txt'.
      2. Parse them into speaker turns (i.e., each speaker's chunk of text).
      3. For each turn:
         - Extract named entities
         - Extract single-word topics (frequent nouns)
         - Extract subtopics for each entity
         - Extract multi-word key noun phrases
      4. Print results in a structured way.
    
    This code uses a naive approach and is easily enhanced by 
    LLM-based summarization, TF-IDF, or more advanced methods.
    """

    # 1) Load spaCy's small English model. 
    # Make sure you've installed: python -m spacy download en_core_web_sm
    nlp = spacy.load("en_core_web_sm")

    # 2) Read the transcript lines manually 
    # to avoid issues with \n as a CSV separator.
    with open("transcript.txt", "r", encoding="utf-8") as f:
        lines = f.read().splitlines()

    # We'll identify a new speaker when we see something like "Alice: Hello..."
    speaker_pattern = re.compile(r'^(.+?):\s*(.*)')

    # We'll collect a list of speaker "turns", e.g.:
    # [
    #   { "speaker": "Alice", "text": "Hello world." },
    #   { "speaker": "Bob",   "text": "Hi Alice, what's up?" },
    #   ...
    # ]
    turns = []

    current_speaker = None
    current_lines = []

    # 3) Build a list of speaker turns.
    for line in lines:
        line = line.strip()
        match = speaker_pattern.match(line)

        if match:
            # If we're starting a new speaker, we first 
            # finalize the previous speaker's chunk.
            if current_speaker is not None:
                turns.append({
                    "speaker": current_speaker,
                    "text": " ".join(current_lines)
                })
            
            # Identify the new speaker and the text that follows on the same line.
            current_speaker = match.group(1).strip()  # e.g. "Alice"
            first_text = match.group(2).strip()       # e.g. "Hello world."
            current_lines = [first_text] if first_text else []
        else:
            # If no new speaker line, this is a continuation of the current speaker's text.
            if current_speaker is not None:
                current_lines.append(line)
            else:
                # If there's text before we even see a speaker, label it "Unknown".
                # This is a fallback to handle transcript formatting quirks.
                if len(turns) == 0 or turns[-1].get("speaker") != "Unknown":
                    turns.append({"speaker": "Unknown", "text": line})
                else:
                    turns[-1]["text"] += " " + line

    # After the loop, we might have an unfinished speaker chunk to save.
    if current_speaker is not None:
        turns.append({
            "speaker": current_speaker,
            "text": " ".join(current_lines)
        })

    # 4) For each turn, run our NLP analyses:
    #    - Named Entities
    #    - Single-word topics
    #    - Entity-based subtopics
    #    - Multi-word noun chunk phrases
    for turn in turns:
        speaker_name = turn["speaker"]
        chunk_text = turn["text"].strip()
        if not chunk_text:
            # Skip if this turn is empty text.
            continue

        doc = nlp(chunk_text)

        # Named Entities
        entities_dict = group_entities_by_label(doc)
        # Single-word topics
        chunk_topics = extract_frequent_nouns(chunk_text, nlp, top_n=5)
        # Entity subtopics (subtopics within the sentences that mention each entity)
        entity_subtopics = extract_entity_subtopics(doc, nlp, top_n=3)
        # Multi-word noun chunk phrases
        key_phrases = extract_noun_chunk_phrases(chunk_text, nlp, top_n=5)

        # Store all these in the turn dictionary 
        # so we can print or process them later.
        turn["entities"] = entities_dict
        turn["topics"] = chunk_topics
        turn["entity_subtopics"] = entity_subtopics
        turn["key_phrases"] = key_phrases

    # 5) Print the results in a readable way.
    for turn in turns:
        speaker_name = turn["speaker"]
        text = turn["text"].strip()
        if not text:
            continue
        
        print(f"=== Speaker: {speaker_name} ===")
        print(f"Text: {text}\n")

        # Named Entities
        entities = turn.get("entities", {})
        if entities:
            print("Named Entities (Entity Label -> Extracted Text):")
            for label, ent_list in entities.items():
                print(f"  {label}: {ent_list}")
        
        # Single-word topics
        topics = turn.get("topics", [])
        print(f"\nTop Single-Word Topics: {topics}")

        # Entity subtopics
        subtopics = turn.get("entity_subtopics", {})
        if subtopics:
            print("\nSubtopics by Entity:")
            for ent, st_list in subtopics.items():
                print(f"  {ent} -> {st_list}")

        # Multi-word noun chunk phrases
        key_phrases = turn.get("key_phrases", [])
        print(f"\nKey Multi-word Noun Phrases: {key_phrases}")

        print("\n" + "-"*50 + "\n")


if __name__ == "__main__":
    main()
