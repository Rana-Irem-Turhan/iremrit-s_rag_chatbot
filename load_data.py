"""
Load and clean SQL question-answer JSON data and save processed chunks.

This script:
- Loads 'sql_create_context_v4.json'
- Removes entries missing 'question', 'context', or 'answer'
- Strips whitespace
- Produces 'processed_chunks.json' containing {'text': "...", 'answer': "...'}
"""

import json
from typing import List, Dict

INPUT_FILE = "sql_create_context_v4.json"
OUTPUT_FILE = "processed_chunks.json"


def load_json(path: str) -> List[Dict]:
    """Load JSON file and return parsed data."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, data: List[Dict]) -> None:
    """Save data as pretty-printed JSON."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def clean_entries(data: List[Dict]) -> List[Dict]:
    """Return entries that contain question, context and answer fields."""
    cleaned: List[Dict] = [
        entry for entry in data
        if isinstance(entry, dict)
        and "question" in entry
        and "context" in entry
        and "answer" in entry
    ]

    for entry in cleaned:
        if isinstance(entry.get("question"), str):
            entry["question"] = entry["question"].strip()
        if isinstance(entry.get("context"), str):
            entry["context"] = entry["context"].strip()
        if isinstance(entry.get("answer"), str):
            entry["answer"] = entry["answer"].strip()

    return cleaned


def create_chunks(cleaned: List[Dict]) -> List[Dict]:
    """Combine context and question into text chunks."""
    chunks: List[Dict] = []
    for entry in cleaned:
        text_chunk = (
            f"Context: {entry['context']}\n"
            f"Question: {entry['question']}"
        )
        chunks.append({"text": text_chunk, "answer": entry["answer"]})
    return chunks


def process_data(data: List[Dict]) -> List[Dict]:
    """Clean and process the SQL data for chatbot usage."""
    cleaned = clean_entries(data)
    return create_chunks(cleaned)


def main() -> None:
    """Main execution function to load, clean, and save data."""
    data = load_json(INPUT_FILE)
    print(f"Total records: {len(data)}")

    if len(data) > 0 and isinstance(data[0], dict):
        print(f"First question: {data[0].get('question')}")
        print(f"The answer: {data[0].get('answer')}")
        print(f"Last question: {data[-1].get('question')}")
        print(f"The answer: {data[-1].get('answer')}")
    print(f"Type of data: {type(data)}")

    cleaned = clean_entries(data)
    print(f"Clean entries: {len(cleaned)} out of {len(data)}")

    for i, entry in enumerate(cleaned[:3]):
        print(f"Sample {i + 1} Question: {entry['question']}")
        print(f"Sample {i + 1} Context: {entry['context']}")
        print(f"Sample {i + 1} Answer: {entry['answer']}")
        print("------")

    chunks = create_chunks(cleaned)
    print(f"Total chunks created: {len(chunks)}")

    save_json(OUTPUT_FILE, chunks)
    print(f"Saved processed chunks to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
