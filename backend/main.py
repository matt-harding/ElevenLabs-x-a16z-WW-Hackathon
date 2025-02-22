#!/usr/bin/env python3

import re
import spacy
import json
from collections import defaultdict, Counter
from knowledge_extender import update_agent_with_file

def extract_frequent_nouns(text, nlp, top_n=5):
    """
    Returns a naive list of single-word 'topics' by:
      1. Processing the text with spaCy to get tokens.
      2. Checking if each token is a NOUN or PROPN, ignoring stopwords/punctuation.
      3. Counting frequency of lemma forms.
      4. Sorting by descending frequency, returning top N.
    """
    doc = nlp(text)
    freq = defaultdict(int)
    
    for token in doc:
        if token.pos_ in ("NOUN", "PROPN") and not token.is_stop and not token.is_punct:
            freq[token.lemma_.lower()] += 1

    sorted_terms = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    return [term for term, _count in sorted_terms[:top_n]]

def extract_noun_chunk_phrases(text, nlp, top_n=5):
    """
    Extract frequent multi-word noun chunks (e.g., 'cloud infrastructure').
    """
    doc = nlp(text)
    chunk_counter = Counter()

    for chunk in doc.noun_chunks:
        chunk_text = chunk.text.strip().lower()
        if not chunk_text or len(chunk_text.split()) < 1:
            continue
        chunk_counter[chunk_text] += 1

    most_common = chunk_counter.most_common(top_n)
    return [phrase for (phrase, _count) in most_common]

def group_entities_by_label(doc):
    """
    Collect named entities into a dict: { label: [entities], ... }
    """
    from collections import defaultdict
    entities_by_label = defaultdict(set)
    for ent in doc.ents:
        entities_by_label[ent.label_].add(ent.text)
    
    return {label: sorted(ents) for label, ents in entities_by_label.items()}

def extract_entity_subtopics(doc, nlp, top_n=3):
    """
    For each entity in the doc, gather subtopics from the sentence(s) containing it.
    Returns something like { "OpenAI": ["research", "investment"], ... }.
    """
    entity_subtopics = {}
    for ent in doc.ents:
        sentence_texts = []
        for sent in doc.sents:
            if ent.start >= sent.start and ent.end <= sent.end:
                sentence_texts.append(sent.text)
        
        context = " ".join(sentence_texts).strip()
        if context:
            subtopics = extract_frequent_nouns(context, nlp, top_n=top_n)
            entity_subtopics[ent.text] = subtopics
    
    return entity_subtopics

def main():
    """
    1. Read 'transcript.txt' line by line, parse into speaker turns.
    2. For each turn, run NLP analysis (entities, topics, subtopics, etc.).
    3. Print results and save them to 'analysis_output.json' for later usage.
    """
    # 1) Load spaCy model
    nlp = spacy.load("en_core_web_sm")

    # 2) Read transcript lines
    with open("transcript.txt", "r", encoding="utf-8") as f:
        lines = f.read().splitlines()

    speaker_pattern = re.compile(r'^(.+?):\s*(.*)')
    turns = []
    current_speaker = None
    current_lines = []

    # Build speaker turns
    for line in lines:
        line = line.strip()
        match = speaker_pattern.match(line)
        if match:
            # Save the previous chunk if any
            if current_speaker is not None:
                turns.append({
                    "speaker": current_speaker,
                    "text": " ".join(current_lines)
                })
            current_speaker = match.group(1).strip()
            first_text = match.group(2).strip()
            current_lines = [first_text] if first_text else []
        else:
            if current_speaker is not None:
                current_lines.append(line)
            else:
                # If we haven't seen a speaker yet, label as Unknown
                if not turns or turns[-1]["speaker"] != "Unknown":
                    turns.append({"speaker": "Unknown", "text": line})
                else:
                    turns[-1]["text"] += " " + line

    # Append the last chunk
    if current_speaker is not None:
        turns.append({
            "speaker": current_speaker,
            "text": " ".join(current_lines)
        })

    # Analyze each turn
    for turn in turns:
        chunk_text = turn["text"].strip()
        if not chunk_text:
            continue
        doc = nlp(chunk_text)
        turn["entities"] = group_entities_by_label(doc)
        turn["topics"] = extract_frequent_nouns(chunk_text, nlp, top_n=5)
        turn["entity_subtopics"] = extract_entity_subtopics(doc, nlp, top_n=3)
        turn["key_phrases"] = extract_noun_chunk_phrases(chunk_text, nlp, top_n=5)

    # Print results
    print("\n=== Transcript Analysis ===\n")
    for turn in turns:
        speaker = turn["speaker"]
        text = turn["text"].strip()
        if not text:
            continue
        
        print(f"--- Speaker: {speaker} ---")
        print(f"Text: {text}\n")
        print(f"Entities: {turn['entities']}")
        print(f"Topics (single-word): {turn['topics']}")
        print(f"Entity Subtopics: {turn['entity_subtopics']}")
        print(f"Key Phrases (multi-word): {turn['key_phrases']}")
        print("-" * 50)

    # Save the analysis output to JSON
    file_name = "analysis_output.json"
    analysis_data = {"turns": turns}
    with open(file_name, "w", encoding="utf-8") as outfile:
        json.dump(analysis_data, outfile, indent=2, ensure_ascii=False)

    print("\nAnalysis saved to 'analysis_output.json'.\n")


    update_agent_with_file(file_name)

if __name__ == "__main__":
    main()
