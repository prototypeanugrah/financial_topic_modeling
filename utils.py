import logging

from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import multiprocessing as mp
import requests
from pathlib import Path
from typing import Tuple, Dict, Any, List
import yaml
import matplotlib.pyplot as plt
from collections import Counter
import re

# Setup logging - modify to only show INFO level
logging.basicConfig(
    level=logging.INFO,
    # format="%(message)s",  # Simplified format
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logging.getLogger("gensim").setLevel(logging.ERROR)  # For gensim
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def setup_logging(config: Dict) -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, config["logging"]["level"]),
        format=config["logging"]["format"],
    )


def setup_output_directory(config: Dict) -> Path:
    """Create timestamped output directory for results."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(config["output"]["base_dir"]) / f"run_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def fetch_single_document(url: str) -> Tuple[str, str]:
    """
    Fetch a single document from a URL.

    Args:
        url: URL to fetch document from

    Returns:
        Tuple of (url, document_text)
    """
    headers = {
        "User-Agent": "Sample Company Name AdminContact@company.com",
        "Host": "www.sec.gov",
    }

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return url, response.text
    except Exception as e:
        logger.error(f"Failed to fetch document from {url}: {str(e)}")
        return url, ""


def load_data(
    data_path: str,
    num_docs: int,
) -> List[str]:
    """
    Load and prepare text documents from URLs using multiprocessing.

    Args:
        data_path: Path to file containing URLs
        num_docs: Number of documents to load (default: 5)

    Returns:
        List of document texts
    """
    try:
        logger.info(f"Loading URLs from {data_path}")

        with open(data_path, "r", encoding="utf-8") as f:
            if num_docs > 0:
                urls = [line.strip() for line in f if line.strip()][:num_docs]
            elif num_docs == 0:
                urls = [line.strip() for line in f if line.strip()]
            else:
                urls = []
                logger.warning(
                    """Wrong number of documents entered. Enter
                    num_docs >=0"""
                )

        if len(urls) == 0:
            raise ValueError(f"No URLs found in {data_path}")

        logger.info(f"Found {len(urls)} URLs to process")
        documents = []

        # Determine number of processes (use max of 4 or number of CPU cores)
        num_processes = min(8, mp.cpu_count())
        logger.info(f"Using {num_processes} processes for parallel document fetching")

        # Use ProcessPoolExecutor for parallel processing
        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            # Submit all URL fetching tasks
            future_to_url = {
                executor.submit(fetch_single_document, url): url for url in urls
            }

            # Process completed tasks as they finish
            for future in as_completed(future_to_url):
                url = future_to_url[future]
                try:
                    url, doc_text = future.result()
                    if doc_text:  # Only append non-empty documents
                        documents.append(doc_text)
                        # logger.info(f"Successfully loaded document from {url}")
                except Exception as e:
                    logger.error(f"Exception processing {url}: {str(e)}")

        if not documents:
            raise ValueError("Failed to load any documents")

        logger.info(f"Successfully loaded {len(documents)} documents")
        return documents

    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise


def save_model_results(
    output_dir: Path,
    lda_model: Any,
    perf_metrics: Dict,
    config: Dict,
    model_type: str,
    vectorizer: Any = None,
    doc_term_matrix: Any = None,
    corpus: Any = None,
) -> None:
    """Save model results and metrics."""
    if config["output"]["save_metrics"]:
        if model_type == "gensim":
            metrics = {
                "perplexity": float(
                    perf_metrics["perplexity"]
                ),  # Convert to native Python float
                "coherence_metrics": {
                    k: float(v) for k, v in perf_metrics["coherence_metrics"].items()
                },  # Convert each metric
                "config": config,
            }
        elif model_type == "sklearn":
            metrics = {
                "perplexity": float(perf_metrics["perplexity"]),
                "log_likelihood": float(perf_metrics["log_likelihood"]),
            }
        else:
            metrics = {}
            logger.warning("Incorrect model type in save results function.")

        with open(output_dir / "metrics.yaml", "w", encoding="utf-8") as f:
            yaml.dump(metrics, f)

    if config["output"]["save_model"]:
        # Save model topics
        with open(output_dir / "topics.txt", "w", encoding="utf-8") as f:
            topics = (
                lda_model.print_topics()
                if model_type == "gensim"
                else (
                    lda_model_sklearn.print_topics(lda_model, vectorizer)
                    if model_type == "sklearn"
                    else []
                )
            )
            for topic in topics:
                f.write(f"{topic}\n")

        # Save document-topic assignments
        if doc_term_matrix is not None:
            if model_type == "sklearn":
                doc_topics = lda_model_sklearn.get_document_topics(
                    lda_model, doc_term_matrix
                )
                with open(
                    output_dir / "document_topics.txt",
                    "w",
                    encoding="utf-8",
                ) as f:
                    f.write("Document_ID,Topics\n")
                    for doc_idx, topic_dist in enumerate(doc_topics):
                        # Get top 3 topics for each document
                        top_topics = topic_dist.argsort()[-3:][::-1]
                        topic_str = "; ".join(
                            [f"Topic {t+1}: {topic_dist[t]:.4f}" for t in top_topics]
                        )
                        f.write(f"Document {doc_idx+1},{topic_str}\n")
            elif model_type == "gensim":
                # For gensim, we need to get document topics differently
                doc_topics = [lda_model.get_document_topics(doc) for doc in corpus]
                with open(
                    output_dir / "document_topics.txt", "w", encoding="utf-8"
                ) as f:
                    f.write("Document_ID,Topics\n")
                    for doc_idx, topics in enumerate(doc_topics):
                        # Get top 3 topics for each document
                        top_topics = sorted(topics, key=lambda x: x[1], reverse=True)[
                            :3
                        ]
                        topic_str = "; ".join(
                            [f"Topic {t[0]+1}: {t[1]:.4f}" for t in top_topics]
                        )
                        f.write(f"Document {doc_idx+1},{topic_str}\n")

        # Save model
        # lda_model.save(str(output_dir / "lda_model"))


def plot_perplexity_scores(
    topic_range: Dict,
    perplexity_scores: List[float],
    output_dir: Path,
):
    """Plot and save perplexity scores vs number of topics."""
    plt.figure(figsize=(10, 6))
    plt.plot(
        range(topic_range["start"], topic_range["limit"], topic_range["step"]),
        perplexity_scores,
        marker="o",
    )
    plt.xlabel("Number of Topics")
    plt.ylabel("Perplexity Score")
    plt.title("Perplexity Score vs Number of Topics")
    plt.grid(True)
    plt.savefig(output_dir / "perplexity_plot.png")
    plt.close()


def save_topic_perplexity_scores(
    topic_range: Dict,
    perplexity_scores: List[float],
    output_dir: Path,
):
    """Save topic numbers and their corresponding perplexity scores to a file."""
    topic_numbers = list(
        range(topic_range["start"], topic_range["limit"], topic_range["step"])
    )

    with open(output_dir / "topic_perplexity_scores.txt", "w", encoding="utf-8") as f:
        f.write("Number_of_Topics,Perplexity_Score\n")
        for topic_num, perplexity in zip(topic_numbers, perplexity_scores):
            f.write(f"{topic_num},{perplexity}\n")


def analyze_word_frequencies(file_path: Path, output_dir: Path) -> None:
    """Analyze word frequencies in topics.txt and create a visualization."""
    # Read the file
    with open(file_path, "r") as f:
        content = f.read()

    # Extract all words and convert to lowercase
    words = re.findall(r"\b\w+\b", content.lower())

    # Count word frequencies
    word_counts = Counter(words)

    # Sort by frequency in descending order
    sorted_counts = dict(sorted(word_counts.items(), key=lambda x: x[1], reverse=True))

    # Get top 20 words
    top_20_words = dict(list(sorted_counts.items())[:20])

    # Create the plot
    plt.figure(figsize=(15, 8))
    plt.bar(top_20_words.keys(), top_20_words.values())
    plt.xticks(rotation=45, ha="right")
    plt.title("Top 20 Most Frequent Words in Topics")
    plt.xlabel("Words")
    plt.ylabel("Frequency")
    plt.tight_layout()

    # Save the plot
    plt.savefig(output_dir / "word_frequency_plot.png")
    plt.close()

    # Print the word frequencies
    logger.info("\nTop 20 most frequent words:")
    for word, count in top_20_words.items():
        logger.info(f"{word}: {count}")
