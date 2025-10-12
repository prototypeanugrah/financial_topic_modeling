import json
import re
from typing import Any, Dict, List, Tuple


def parse_topic_file(topic_file_path: str) -> Dict[int, List[Tuple[float, str]]]:
    """
    Parse topic file to extract word probabilities for each topic.
    Returns a dictionary mapping topic_id to list of (probability, word) tuples.
    """
    topics = {}

    with open(topic_file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("Topic "):
                # Extract topic number
                topic_match = re.match(r"Topic (\d+):", line)
                if topic_match:
                    topic_id = int(topic_match.group(1))

                    # Extract word-probability pairs
                    word_prob_pairs = []
                    # Pattern to match: 0.027*"clinical" + 0.023*"drug" + ...
                    pattern = r'(\d+\.\d+)\*"([^"]+)"'
                    matches = re.findall(pattern, line)

                    for prob_str, word in matches:
                        probability = float(prob_str)
                        word_prob_pairs.append((probability, word))

                    # Sort by probability in descending order
                    word_prob_pairs.sort(key=lambda x: x[0], reverse=True)
                    topics[topic_id] = word_prob_pairs

    return topics


def parse_document_topics_file(
    doc_topics_file_path: str,
) -> List[List[Tuple[int, float]]]:
    """
    Parse document topics file to extract topic assignments for each document.
    Returns a list where each element is a list of (topic_id, probability) tuples for that document.
    """
    documents = []

    with open(doc_topics_file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("[") and line.endswith("]"):
                # Parse the list of tuples: [(0, 0.035390973), (5, 0.02996108), ...]
                # Remove brackets and split by '), ('
                content = line[1:-1]  # Remove outer brackets
                if content:  # If not empty
                    tuples = []
                    # Split by '), (' pattern
                    parts = content.split("), (")
                    for part in parts:
                        # Clean up the part
                        part = part.strip("()")
                        if part:
                            topic_id, prob = part.split(", ")
                            tuples.append((int(topic_id), float(prob)))

                    # Sort by probability in descending order
                    tuples.sort(key=lambda x: x[1], reverse=True)
                    documents.append(tuples)
                else:
                    documents.append([])

    return documents


def get_dominant_topic(document_topics: List[Tuple[int, float]]) -> Tuple[int, float]:
    """
    Get the topic with the highest probability for a document.
    """
    if not document_topics:
        return 0, 0.0
    topic_id, topic_prob = document_topics[0]
    return topic_id, topic_prob


def create_document_word_structure(
    analyst_doc_topics: List[List[Tuple[int, float]]],
    ipo_doc_topics: List[List[Tuple[int, float]]],
    analyst_topics: Dict[int, List[Tuple[float, str]]],
    ipo_topics: Dict[int, List[Tuple[float, str]]],
) -> Dict[str, Any]:
    """
    Create the JSON structure as requested.
    """
    result = {}

    # Ensure both lists have the same length
    min_length = min(len(analyst_doc_topics), len(ipo_doc_topics))

    for i in range(min_length):
        doc_key = f"document_{i + 1}"

        # Get dominant topics for both analyst and IPO
        analyst_dominant_topic, analyst_dominant_topic_prob = get_dominant_topic(
            analyst_doc_topics[i]
        )
        ipo_dominant_topic, ipo_dominant_topic_prob = get_dominant_topic(
            ipo_doc_topics[i]
        )

        # Get word distributions for dominant topics
        analyst_words = analyst_topics.get(analyst_dominant_topic, [])
        ipo_words = ipo_topics.get(ipo_dominant_topic, [])

        # Create the structure
        result[doc_key] = {
            "analyst_topic_num": analyst_dominant_topic,  # dominant analyst topic
            "analyst_topic_prob": analyst_dominant_topic_prob,  # dominant analyst topic probability
            "ipo_topic_num": ipo_dominant_topic,  # dominant ipo topic
            "ipo_topic_prob": ipo_dominant_topic_prob,  # dominant ipo topic probability
            "analyst_words_in_descending_prob": {
                f"word_{j + 1}": (prob, word)
                for j, (prob, word) in enumerate(analyst_words)
            },  # analyst words in descending probability
            "ipo_words_in_descending_prob": {
                f"word_{j + 1}": (prob, word)
                for j, (prob, word) in enumerate(ipo_words)
            },  # ipo words in descending probability
        }

    return result


def main():
    # File paths
    base_path = (
        "/Users/anugrah/Desktop/financial_topic_modeling/outputs/run_20251006_123540"
    )

    analyst_topics_file = f"{base_path}/analyst_topics.txt"
    analyst_doc_topics_file = f"{base_path}/analystdocument_topics_train.txt"
    ipo_topics_file = f"{base_path}/ipo_topics.txt"
    ipo_doc_topics_file = f"{base_path}/ipodocument_topics_train.txt"

    # Parse the files
    print("Parsing analyst topics...")
    analyst_topics = parse_topic_file(analyst_topics_file)

    print("Parsing IPO topics...")
    ipo_topics = parse_topic_file(ipo_topics_file)

    print("Parsing analyst document topics...")
    analyst_doc_topics = parse_document_topics_file(analyst_doc_topics_file)

    print("Parsing IPO document topics...")
    ipo_doc_topics = parse_document_topics_file(ipo_doc_topics_file)

    print(f"Found {len(analyst_topics)} analyst topics")
    print(f"Found {len(ipo_topics)} IPO topics")
    print(f"Found {len(analyst_doc_topics)} analyst documents")
    print(f"Found {len(ipo_doc_topics)} IPO documents")

    # Create the JSON structure
    print("Creating JSON structure...")
    result = create_document_word_structure(
        analyst_doc_topics, ipo_doc_topics, analyst_topics, ipo_topics
    )

    # Save to JSON file
    output_file = f"{base_path}/document_word_probabilities.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"JSON structure saved to: {output_file}")
    print(f"Total documents processed: {len(result)}")

    # Print a sample
    if result:
        sample_doc = list(result.keys())[0]
        print(f"\nSample document ({sample_doc}):")
        print(json.dumps(result[sample_doc], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
