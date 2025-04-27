"""
This file contains utility functions for the LDA topic modeling pipeline.

Returns:
    None
"""

from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Tuple, Dict, Any, List, Iterator
import logging
import multiprocessing as mp
import time
import yaml

import requests
import matplotlib.pyplot as plt
import re
from tqdm import tqdm

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


def setup_logging(config: Dict) -> None:
    """
    Setup logging configuration.

    Args:
        config: Dictionary containing configuration

    Returns:
        None
    """
    logging.basicConfig(
        level=getattr(logging, config["logging"]["level"]),
        format=config["logging"]["format"],
    )


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
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        return url, response.text
    except Exception as e:
        logger.error("Failed to fetch document from %s: %s", url, str(e))
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
        logger.info("Loading URLs from %s", data_path)

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
            raise ValueError("No URLs found in %s", data_path)

        logger.info("Found %s URLs to process", len(urls))
        documents = []

        # Determine number of processes (use max of 4 or number of CPU cores)
        num_processes = min(8, mp.cpu_count())
        logger.info("Using %s processes for parallel document fetching", num_processes)

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
                    logger.error("Exception processing %s: %s", url, str(e))

        if not documents:
            raise ValueError("Failed to load any documents")

        logger.info("Successfully loaded %s documents", len(documents))
        return documents

    except Exception as e:
        logger.error("Error loading data: %s", str(e))
        raise


def load_data_sequential(
    data_path: str,
    output_dir: str = "raw_data_files",
    delay: float = 0.5,  # delay between requests in seconds
) -> List[str]:
    """
    Load and prepare text documents from URLs sequentially, saving each to a file.

    Args:
        data_path: Path to file containing URLs
        output_dir: Directory to save the downloaded files (default: "raw_data_files")
        delay: Delay between requests in seconds (default: 0.5)

    Returns:
        List of paths to saved files
    """
    try:
        logger.info("Loading URLs from %s", data_path)

        # Create output directory if it doesn't exist
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Read URLs from file
        with open(data_path, "r", encoding="utf-8") as f:
            urls = [line.strip() for line in f if line.strip()]

        if len(urls) == 0:
            raise ValueError("No URLs found in %s", data_path)

        logger.info("Found %s URLs to process", len(urls))
        saved_files = []

        # Process URLs sequentially with progress bar
        for url in tqdm(urls, desc="Downloading documents", unit="file"):
            try:
                # Create a filename from the URL
                # Extract the last part of the URL and clean it
                url_filename = url.split("/")[-1]
                # Remove any non-alphanumeric characters except dots and dashes
                url_filename = re.sub(r"[^a-zA-Z0-9.-]", "_", url_filename)
                file_path = output_path / f"{url_filename}.txt"

                # Fetch the document
                _, doc_text = fetch_single_document(url)

                if doc_text:
                    # Save the document to a file
                    with open(file_path, "w", encoding="utf-8") as f:
                        f.write(doc_text)
                    saved_files.append(str(file_path))
                else:
                    logger.warning("No content retrieved from %s", url)

                # Add delay between requests
                time.sleep(delay)

            except Exception as e:
                logger.error("Error processing URL %s: %s", url, str(e))
                continue

        if not saved_files:
            raise ValueError("Failed to save any documents")

        logger.info("Successfully saved %s documents", len(saved_files))
        return saved_files

    except Exception as e:
        logger.error("Error in load_data_sequential: %s", str(e))
        raise


def save_model_results(
    output_dir: Path,
    lda_model: Any,
    perf_metrics: Dict,
    config: Dict,
) -> None:
    """Save model results and metrics."""
    if config["output"]["save_metrics"]:
        metrics = {
            "perplexity": float(
                perf_metrics["perplexity"]
            ),  # Convert to native Python float
            "coherence_metrics": {
                k: float(v) for k, v in perf_metrics["coherence_metrics"].items()
            },  # Convert each metric
            "config": config,
        }

        with open(output_dir / "metrics.yaml", "w", encoding="utf-8") as f:
            yaml.dump(metrics, f)

    if config["output"]["save_model"]:
        # Save model topics
        with open(output_dir / "topics.txt", "w", encoding="utf-8") as f:
            topics = lda_model.print_topics()
            for topic in topics:
                f.write(f"{topic}\n")

        # Save model
        # lda_model.save(str(output_dir / "lda_model"))


def plot_perplexity_scores(
    topic_range: Dict,
    perplexity_scores: List[float],
    output_dir: Path,
) -> None:
    """
    Plot and save perplexity scores vs number of topics.

    Args:
        topic_range: Dictionary containing topic range information
        perplexity_scores: List of perplexity scores
        output_dir: Path to output directory

    Returns:
        None
    """
    plt.figure(figsize=(10, 6))
    plt.plot(
        range(
            topic_range["start"],
            topic_range["limit"],
            topic_range["step"],
        ),
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
) -> None:
    """
    Save topic numbers and their corresponding perplexity scores to a file.

    Args:
        topic_range: Dictionary containing topic range information
        perplexity_scores: List of perplexity scores
        output_dir: Path to output directory

    Returns:
        None
    """
    topic_numbers = list(
        range(
            topic_range["start"],
            topic_range["limit"],
            topic_range["step"],
        )
    )

    with open(
        output_dir / "topic_perplexity_scores.txt",
        "w",
        encoding="utf-8",
    ) as f:
        f.write("Number_of_Topics,Perplexity_Score\n")
        for topic_num, perplexity in zip(topic_numbers, perplexity_scores):
            f.write(f"{topic_num},{perplexity}\n")


def analyze_word_frequencies(file_path: Path, output_dir: Path) -> None:
    """Analyze word frequencies in topics.txt and create a visualization."""
    # Read the file
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Extract all words and convert to lowercase
    words = re.findall(r"\b\w+\b", content.lower())

    # Count word frequencies
    word_counts = Counter(words)

    # Sort by frequency in descending order
    sorted_counts = dict(
        sorted(
            word_counts.items(),
            key=lambda x: x[1],
            reverse=True,
        ),
    )

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
    plt.savefig(output_dir / "word_frequency_plot.png", dpi=300)
    plt.close()

    # # Print the word frequencies
    # logger.info("\nTop 20 most frequent words:")
    # for word, count in top_20_words.items():
    #     logger.info("%s: %s", word, count)


def load_single_file(file_path: str) -> Tuple[str, str]:
    """
    Load a single file and return its contents.

    Args:
        file_path: Path to the file to load

    Returns:
        Tuple of (file_path, file_contents)
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        return file_path, content
    except Exception as e:
        logger.error("Failed to load file %s: %s", file_path, str(e))
        return file_path, ""


def load_files(
    input_dir: str = "raw_data_files",
    num_files: int = 0,
) -> List[str]:
    """
    Load and prepare text documents from saved files using multiprocessing.

    Args:
        input_dir: Directory containing the saved files (default: "raw_data_files")
        num_files: Number of files to load (0 for all files)

    Returns:
        List of document texts
    """
    try:
        input_path = Path(input_dir)
        if not input_path.exists():
            raise ValueError("Input directory %s does not exist", input_dir)

        # Get list of all .txt files in the directory
        files = list(input_path.glob("*.txt"))

        if num_files > 0:
            files = files[:num_files]

        if len(files) == 0:
            raise ValueError("No files found in %s", input_dir)

        logger.info("Found %s files to process", len(files))
        documents = []

        # Determine number of processes (use max of 8 or number of CPU cores)
        num_processes = min(8, mp.cpu_count())
        logger.info("Using %s processes for parallel file loading", num_processes)

        # Use ProcessPoolExecutor for parallel processing
        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            # Submit all file loading tasks
            future_to_file = {
                executor.submit(load_single_file, str(file_path)): file_path
                for file_path in files
            }

            # Process completed tasks as they finish
            for future in tqdm(
                as_completed(future_to_file),
                total=len(files),
                desc="Loading files",
                unit="file",
            ):
                file_path = future_to_file[future]
                try:
                    _, content = future.result()
                    if content:  # Only append non-empty documents
                        documents.append(content)
                except Exception as e:
                    logger.error("Exception processing %s: %s", file_path, str(e))

        if not documents:
            raise ValueError("Failed to load any documents")

        logger.info("Successfully loaded %s documents", len(documents))
        return documents

    except Exception as e:
        logger.error("Error loading files: %s", str(e))
        raise


def load_files_in_batches(
    batch_size: int,
    num_docs: int = 0,
    input_dir: str = "raw_data_files",
) -> Iterator[List[str]]:
    """
    Generator function to load files in batches

    Args:
        input_dir: Directory containing the saved files
        batch_size: Number of files to load in each batch

    Returns:
        Iterator of lists of document texts
    """
    input_path = Path(input_dir)
    files = list(input_path.glob("*.txt"))

    if not files:
        logger.error("No .txt files found in directory: %s", input_dir)
        # Yield an empty list to avoid errors downstream
        yield []
        return

    logger.info("Found %d .txt files in directory: %s", len(files), input_dir)

    if num_docs > 0:
        files = files[:num_docs]
        logger.info("Using %d files for processing", len(files))

    for i in range(0, len(files), batch_size):
        batch_files = files[i : i + batch_size]
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
            logger.warning("No valid documents in batch %d-%d", i, i + batch_size)
            # Yield an empty list to maintain the generator pattern
            yield []


def plot_ideal_topic_number(
    num_topics: List[int],
    mean_stabilities: List[float],
    coherences: List[float],
    ideal_topic_num: int,
) -> None:
    """
    Plot the ideal topic number and the metrics per number of topics.

    Args:
        num_topics (List[int]): List of topic numbers
        mean_stabilities (List[float]): List of mean stability scores
        coherences (List[float]): List of coherence scores
        ideal_topic_num (int): Ideal number of topics
    """
    plt.figure(figsize=(20, 10))
    ax = sns.lineplot(
        x=num_topics[:-1],
        y=mean_stabilities,
        label="Average Topic Overlap",
    )
    ax = sns.lineplot(
        x=num_topics[:-1],
        y=coherences,
        label="Topic Coherence",
    )

    ax.axvline(
        x=ideal_topic_num,
        label="Ideal Number of Topics",
        color="black",
    )
    ax.axvspan(
        xmin=ideal_topic_num - 1,
        xmax=ideal_topic_num + 1,
        alpha=0.5,
        facecolor="grey",
    )

    y_max = max(max(mean_stabilities), max(coherences)) + (
        0.10 * max(max(mean_stabilities), max(coherences))
    )
    ax.set_ylim([0, y_max])
    ax.set_xlim([1, num_topics[-1] - 1])

    ax.axes.set_title("Model Metrics per Number of Topics", fontsize=25)
    ax.set_ylabel("Metric Level", fontsize=20)
    ax.set_xlabel("Number of Topics", fontsize=20)
    plt.legend(fontsize=20)
    plt.show()
