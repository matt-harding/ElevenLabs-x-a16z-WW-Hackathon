#!/usr/bin/env python3

import re
import spacy
from collections import defaultdict, Counter

def extract_frequent_nouns(text, nlp, top_n=5):
    """
    Returns a naive list of 'topics' by collecting frequent NOUN/PROPN lemmas.
    Excludes stop words and punctuation. Sort by frequency descending.
    """
    doc = nlp(text)
    freq = defaultdict(int)
    
    for token in doc:
        if token.pos_ in ("NOUN", "PROPN") and not token.is_stop and not token.is_punct:
            freq[token.lemma_.lower()] += 1

    # Sort by descending frequency
    sorted_terms = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    return [term for term, _count in sorted_terms[:top_n]]

def extract_noun_chunk_phrases(text, nlp, top_n=5):
    """
    Extract frequent noun chunks from the text (e.g., 'cloud infrastructure').
    We:
      - Use spaCy's 'noun_chunks' to capture multi-word phrases.
      - Clean them (exclude very short/stopword chunks).
      - Count frequencies, then return top_n.
    """
    doc = nlp(text)
    chunk_counter = Counter()

    for chunk in doc.noun_chunks:
        # Basic cleaning
        chunk_text = chunk.text.strip()
        if len(chunk_text.split()) < 1:
            continue
        # Optionally skip if chunk is just a stopword, etc.
        chunk_counter[chunk_text.lower()] += 1

    most_common = chunk_counter.most_common(top_n)
    return [phrase for (phrase, _count) in most_common]

def group_entities_by_label(doc):
    """
    From a spaCy Doc, returns a dict of entity_label -> list of entity_text.
    Example: { "PERSON": ["Alice", "Bob"], "ORG": ["OpenAI"] }
    """
    entities_by_label = defaultdict(set)
    for ent in doc.ents:
        entities_by_label[ent.label_].add(ent.text)
    
    # Convert sets to sorted lists for neatness
    return {label: sorted(list(ents)) for label, ents in entities_by_label.items()}

def extract_entity_subtopics(doc, nlp, top_n=3):
    """
    For each entity in the doc, gather subtopics from the sentence(s) in which
    that entity appears. Returns a dict: { entity_text: [topics] }
    """
    entity_subtopics = {}
    for ent in doc.ents:
        # We'll gather all sentences that contain this entity
        sentence_texts = []
        for sent in doc.sents:
            if ent.start >= sent.start and ent.end <= sent.end:
                # This sentence contains the entity
                sentence_texts.append(sent.text)
        
        # Combine the relevant sentences into one chunk
        context = " ".join(sentence_texts).strip()
        if context:
            subtopics = extract_frequent_nouns(context, nlp, top_n=top_n)
            entity_subtopics[ent.text] = subtopics
    
    return entity_subtopics

def main():
    """
    1. Read 'transcript.txt' line by line, splitting into speaker turns.
    2. For each turn:
       - Extract named entities (layer 1).
       - Extract overall chunk topics (layer 2).
       - Extract subtopics for each entity mention.
       - Extract key multi-word noun chunks for more relevant info.
    3. Print the final structure with speakers, text, entities, topics, 
       entity subtopics, and top noun chunk phrases.
    """

    # 1) Load spaCy model
    nlp = spacy.load("en_core_web_sm")

    # 2) Read transcript lines (bypassing CSV parser for the \n issue)
    with open("transcript.txt", "r", encoding="utf-8") as f:
        lines = f.read().splitlines()

    # 3) Collect speaker turns
    speaker_pattern = re.compile(r'^(.+?):\s*(.*)')
    turns = []
    current_speaker = None
    current_lines = []

    for line in lines:
        line = line.strip()
        match = speaker_pattern.match(line)

        if match:
            # Save previous chunk if any
            if current_speaker is not None:
                turns.append({
                    "speaker": current_speaker,
                    "text": " ".join(current_lines)
                })
            
            # Start new chunk
            current_speaker = match.group(1).strip()
            first_text = match.group(2).strip()
            current_lines = [first_text] if first_text else []
        else:
            # Continuation of current speaker's text
            if current_speaker is not None:
                current_lines.append(line)
            else:
                # If no speaker set yet, treat as 'Unknown'
                if len(turns) == 0 or turns[-1].get("speaker") != "Unknown":
                    turns.append({"speaker": "Unknown", "text": line})
                else:
                    turns[-1]["text"] += " " + line

    # Append the last chunk
    if current_speaker is not None:
        turns.append({
            "speaker": current_speaker,
            "text": " ".join(current_lines)
        })

    # 4) Analyze each turn
    for turn in turns:
        speaker_name = turn["speaker"]
        chunk_text = turn["text"].strip()
        if not chunk_text:
            continue

        doc = nlp(chunk_text)

        # (A) Named Entities
        entities_dict = group_entities_by_label(doc)

        # (B) Overall chunk topics (single-word approach)
        chunk_topics = extract_frequent_nouns(chunk_text, nlp, top_n=5)

        # (C) Subtopics for each entity mention
        entity_subtopics = extract_entity_subtopics(doc, nlp, top_n=3)

        # (D) Key multi-word noun chunk phrases
        key_phrases = extract_noun_chunk_phrases(chunk_text, nlp, top_n=5)

        # Store this analysis in the turn data structure
        turn["entities"] = entities_dict
        turn["topics"] = chunk_topics
        turn["entity_subtopics"] = entity_subtopics
        turn["key_phrases"] = key_phrases

    # 5) Print the results with improved context
    for turn in turns:
        speaker_name = turn["speaker"]
        text = turn["text"].strip()
        if not text:
            continue
        
        print(f"=== Speaker: {speaker_name} ===")
        print(f"Text: {text}\n")

        # Entities
        entities = turn.get("entities", {})
        if entities:
            print("Named Entities:")
            for label, ent_list in entities.items():
                print(f"  {label}: {ent_list}")
        
        # Overall single-word topics
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
        print(f"\nKey Multi-word Phrases: {key_phrases}")

        print("\n" + "-"*50 + "\n")


if __name__ == "__main__":
    main()
