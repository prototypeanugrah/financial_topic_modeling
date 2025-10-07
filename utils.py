"""
This file contains utility functions for the LDA topic modeling pipeline.

Returns:
    None
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple

import yaml
from gensim import corpora

# Setup logging - modify to only show INFO level
logging.basicConfig(
    level=logging.INFO,
    # format="%(message)s",  # Simplified format
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logging.getLogger("gensim").setLevel(logging.ERROR)  # For gensim
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to configuration file

    Returns:
        Dictionary containing configuration
    """
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def setup_output_directory(config: Dict) -> Path:
    """
    Create timestamped output directory for results.

    Args:
        config: Dictionary containing configuration

    Returns:
        Path to output directory
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(config["output"]["base_dir"]) / f"run_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def save_model_results(
    output_dir: Path,
    lda_model: Any,
    train_corpus: Any,
    test_corpus: Any,
    perf_metrics: Dict[str, Dict],
    config: Dict,
    prefix: str = "",
):
    """
    Save LDA model results to files with optional prefix.

    Args:
        output_dir: Directory to save results
        lda_model: Trained LDA model
        train_corpus: Training corpus
        test_corpus: Test corpus
        perf_metrics: Performance metrics dictionary
        config: Configuration dictionary
        prefix: Optional prefix for output files
    """
    logger = logging.getLogger(__name__)

    try:
        # Save model
        model_path = output_dir / f"{prefix}_lda_model"
        lda_model.save(str(model_path))
        logger.info(f"Model saved to {model_path}")

        # Save corpus
        corpus_path = output_dir / f"{prefix}_lda_model.bow_corpus.mm"
        corpora.MmCorpus.serialize(str(corpus_path), train_corpus)
        logger.info(f"Training corpus saved to {corpus_path}")

        # Save topics
        topics_path = output_dir / f"{prefix}_topics.txt"
        with open(topics_path, "w", encoding="utf-8") as f:
            for idx, topic in lda_model.print_topics(-1, num_words=50):
                f.write(f"Topic {idx}: {topic}\n")
        logger.info(f"Topics saved to {topics_path}")

        # Save metrics
        metrics_path = output_dir / f"{prefix}_metrics.yaml"
        with open(metrics_path, "w", encoding="utf-8") as f:
            yaml.dump(perf_metrics, f, default_flow_style=False)
        logger.info(f"Metrics saved to {metrics_path}")

    except Exception as e:
        logger.error(f"Failed to save model results: {e}")
        raise


def plot_perplexity_scores(
    topic_range: Dict[str, int],
    perplexity_scores: List[float],
    output_dir: Path,
    mode: str,
    prefix: str,
) -> None:
    """
    Plot and save perplexity scores with optional prefix.
    """
    logger = logging.getLogger(__name__)

    try:
        import matplotlib.pyplot as plt

        # Convert topic_range dict to list of actual topic numbers
        topic_numbers = list(
            range(topic_range["start"], topic_range["limit"], topic_range["step"])
        )

        plt.figure(figsize=(10, 6))
        plt.plot(topic_numbers, perplexity_scores, marker="o")
        plt.title(f"Perplexity Scores vs Number of Topics ({mode.title()} Set)")
        plt.xlabel("Number of Topics")
        plt.ylabel("Perplexity Score")
        plt.grid(True, alpha=0.3)

        plot_path = output_dir / f"{prefix}perplexity_plot_{mode}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Perplexity plot saved to {plot_path}")
    except Exception as e:
        logger.error(f"Failed to create perplexity plot: {e}")
        raise


def save_topic_perplexity_scores(
    topic_range: Dict[str, int],
    perplexity_scores: List[float],
    output_dir: Path,
    prefix: str,
    mode: str,
) -> None:
    """
    Save topic perplexity scores with optional prefix.

    Args:
        topic_range: Dictionary with start, limit, step for topic numbers
        perplexity_scores: List of perplexity scores
        output_dir: Path to output directory
        prefix: Optional prefix for output files
        mode: Mode of the perplexity scores
    Returns:
        None
    """
    logger = logging.getLogger(__name__)

    try:
        # Convert topic_range dict to list of actual topic numbers
        topic_numbers = list(
            range(topic_range["start"], topic_range["limit"], topic_range["step"])
        )

        scores_path = output_dir / f"{prefix}topic_perplexity_scores_{mode}.txt"
        with open(scores_path, "w", encoding="utf-8") as f:
            f.write("Topic Number\tPerplexity Score\n")
            for topic_num, score in zip(topic_numbers, perplexity_scores):
                f.write(f"{topic_num}\t{score:.4f}\n")
        logger.info(f"Topic perplexity scores saved to {scores_path}")
    except Exception as e:
        logger.error(f"Failed to save topic perplexity scores: {e}")
        raise


def analyze_word_frequencies(
    file_path: Path, output_dir: Path, prefix: str = ""
) -> None:
    """
    Analyze word frequencies from topics file with optional prefix.
    """
    logger = logging.getLogger(__name__)

    try:
        import re
        from collections import Counter

        import matplotlib.pyplot as plt

        # Read topics file
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Extract words and their weights
        word_pattern = r'(\d+\.\d+)\*"([^"]+)"'
        matches = re.findall(word_pattern, content)

        # Create word frequency dictionary
        word_freq = Counter()
        for weight, word in matches:
            word_freq[word] += float(weight)

        # Get top 20 words
        top_words = word_freq.most_common(20)

        # Create plot
        words, frequencies = zip(*top_words)

        plt.figure(figsize=(12, 8))
        plt.barh(range(len(words)), frequencies)
        plt.yticks(range(len(words)), words)
        plt.xlabel("Frequency Weight")
        plt.title("Top 20 Most Frequent Words Across All Topics")
        plt.gca().invert_yaxis()
        plt.tight_layout()

        # Save plot
        plot_path = output_dir / f"{prefix}word_frequency_plot.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Word frequency plot saved to {plot_path}")

    except Exception as e:
        logger.error(f"Failed to analyze word frequencies: {e}")
        raise


def load_documents_from_paths(
    file_paths: List[str],
    num_docs: int = 0,
) -> List[str]:
    """
    Load documents from a list of file paths.

    Args:
        file_paths: List of file paths to load
        num_docs: Number of documents to load (0 for all)

    Returns:
        List of document texts
    """
    logger = logging.getLogger(__name__)

    if num_docs > 0:
        file_paths = file_paths[:num_docs]
        logger.info("Using %d files for processing", len(file_paths))

    documents = []
    for file_path in file_paths:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                if content.strip():  # Only add non-empty documents
                    documents.append(content)
                else:
                    logger.warning("Empty file found: %s", file_path)
        except Exception as e:
            logger.error("Error loading %s: %s", file_path, str(e))

    logger.info("Loaded %d documents from file paths", len(documents))
    return documents


def load_files_in_batches_from_paths(
    file_paths: List[str],
    batch_size: int,
    test_perc: float,
    num_docs: int = 0,
) -> Tuple[Iterator[List[str]], Iterator[List[str]]]:
    """
    Generator function to load files in batches from file paths with train-test split

    Args:
        file_paths: List of file paths to load
        batch_size: Number of files to load in each batch
        test_perc: Percentage of files to use for testing (0.0 to 1.0)
        num_docs: Number of files to load (0 for all files)

    Returns:
        Tuple of (train_batches_iterator, test_batches_iterator)
    """
    logger = logging.getLogger(__name__)

    if not file_paths:
        logger.error("No file paths provided")
        return iter([]), iter([])

    if num_docs > 0:
        file_paths = file_paths[:num_docs]

    # Calculate split point
    split_idx = int(len(file_paths) * (1 - test_perc))
    train_files = file_paths[:split_idx]
    test_files = file_paths[split_idx:]

    logger.info(
        "Split into %d training and %d test files",
        len(train_files),
        len(test_files),
    )

    def batch_generator(file_list: List[str]) -> Iterator[List[str]]:
        for i in range(0, len(file_list), batch_size):
            batch_files = file_list[i : i + batch_size]
            batch_documents = []
            for file_path in batch_files:
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()
                        if content.strip():  # Only add non-empty documents
                            batch_documents.append(content)
                        else:
                            logger.warning("Empty file found: %s", file_path)
                except Exception as e:
                    logger.error("Error loading %s: %s", file_path, str(e))

            if batch_documents:  # Only yield batches with documents
                yield batch_documents
            else:
                logger.warning(
                    "No valid documents in batch %d-%d",
                    i,
                    i + batch_size,
                )
                # Yield an empty list to maintain the generator pattern
                yield []

    return batch_generator(train_files), batch_generator(test_files)
