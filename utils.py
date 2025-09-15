"""
This file contains utility functions for the LDA topic modeling pipeline.

Returns:
    None
"""

import logging
import multiprocessing as mp
import random
import re
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple

import matplotlib.pyplot as plt
import requests
import yaml
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
    perf_metrics: Dict[str, Dict],
    config: Dict,
) -> None:
    """Save model results and metrics."""
    if config["output"]["save_metrics"]:
        metrics = {
            "perplexity": {
                "train": float(perf_metrics["train"]["perplexity"]),
                "test": float(perf_metrics["test"]["perplexity"]),
            },
            "config": config,
        }

        with open(output_dir / "metrics.yaml", "w", encoding="utf-8") as f:
            yaml.dump(metrics, f)

    if config["output"]["save_model"]:
        # Save model topics
        with open(output_dir / "topics.txt", "w", encoding="utf-8") as f:
            # Get the number of topics from the model
            num_topics = lda_model.num_topics
            # Print all topics
            topics = lda_model.print_topics(num_topics=num_topics)
            for topic in topics:
                f.write(f"{topic}\n")

        # Save model
        lda_model.save(str(output_dir / "lda_model"))


def plot_perplexity_scores(
    topic_range: Dict,
    perplexity_scores: List[float],
    output_dir: Path,
    mode: str,
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
    plt.title(f"Perplexity Score vs Number of Topics ({mode})")
    plt.grid(True)
    plt.savefig(output_dir / f"perplexity_plot_{mode}.png")
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


def save_optimization_metrics(
    topic_range: Dict,
    all_metrics: Dict[int, Dict[str, float]],
    output_dir: Path,
) -> None:
    """
    Save comprehensive metrics from topic optimization.

    Args:
        topic_range: Topic range configuration
        all_metrics: Dictionary of all metrics per topic number
        output_dir: Output directory path
    """
    # Prepare data for saving
    results = []
    for num_topics in sorted(all_metrics.keys()):
        metrics = all_metrics[num_topics]
        row = {
            "num_topics": num_topics,
            "perplexity": metrics.get("perplexity"),
            "coherence_c_v": metrics.get("coherence_c_v"),
            "coherence_u_mass": metrics.get("coherence_u_mass"),
            "coherence_c_npmi": metrics.get("coherence_c_npmi"),
        }
        results.append(row)

    # Save as CSV
    import csv

    csv_path = output_dir / "optimization_metrics.csv"
    with open(csv_path, "w", newline="") as f:
        if results:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)

    logger.info(f"Saved optimization metrics to {csv_path}")

    # Also save as JSON for programmatic access
    json_path = output_dir / "optimization_metrics.json"
    with open(json_path, "w") as f:
        import json

        json.dump(all_metrics, f, indent=2)


def plot_metrics_comparison(
    topic_range: Dict,
    all_metrics: Dict[int, Dict[str, float]],
    output_dir: Path,
) -> None:
    """
    Create comprehensive visualization of all metrics.
    """
    import matplotlib.pyplot as plt

    topic_numbers = sorted(all_metrics.keys())

    # Extract metrics
    perplexity = [all_metrics[n].get("perplexity", float("inf")) for n in topic_numbers]
    coherence_cv = [all_metrics[n].get("coherence_c_v", 0) for n in topic_numbers]
    coherence_umass = [all_metrics[n].get("coherence_u_mass", 0) for n in topic_numbers]

    # Create figure with subplots
    fig, axes = plt.subplots(3, 1, figsize=(10, 12))

    # Plot perplexity
    axes[0].plot(topic_numbers, perplexity, "b-o")
    axes[0].set_xlabel("Number of Topics")
    axes[0].set_ylabel("Perplexity (lower is better)")
    axes[0].set_title("Perplexity vs Number of Topics")
    axes[0].grid(True, alpha=0.3)

    # Mark best perplexity
    valid_perplexity = [p for p in perplexity if p != float("inf")]
    if valid_perplexity:
        best_idx = perplexity.index(min(valid_perplexity))
        axes[0].plot(topic_numbers[best_idx], perplexity[best_idx], "r*", markersize=15)
        axes[0].annotate(
            f"Best: {topic_numbers[best_idx]} topics",
            xy=(topic_numbers[best_idx], perplexity[best_idx]),
            xytext=(10, 10),
            textcoords="offset points",
        )

    # Plot coherence C_V
    axes[1].plot(topic_numbers, coherence_cv, "g-s")
    axes[1].set_xlabel("Number of Topics")
    axes[1].set_ylabel("Coherence C_V (higher is better)")
    axes[1].set_title("Topic Coherence C_V vs Number of Topics")
    axes[1].grid(True, alpha=0.3)

    # Plot coherence U_Mass
    axes[2].plot(topic_numbers, coherence_umass, "m-^")
    axes[2].set_xlabel("Number of Topics")
    axes[2].set_ylabel("Coherence U_Mass (higher is better)")
    axes[2].set_title("Topic Coherence U_Mass vs Number of Topics")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "metrics_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(
        f"Saved metrics comparison plot to {output_dir / 'metrics_comparison.png'}"
    )


def analyze_word_frequencies(file_path: Path, output_dir: Path) -> None:
    """Analyze word frequencies in topics.txt and create a visualization."""
    # Read the file
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Extract only topic words between quotes (ignoring topic IDs and probabilities)
    words = re.findall(r'"([^"]+)"', content)

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
    test_perc: float,
    num_docs: int = 0,
    input_dir: str = "raw_data_files",
) -> Tuple[Iterator[List[str]], Iterator[List[str]]]:
    """
    Generator function to load files in batches with train-test split

    Args:
        batch_size: Number of files to load in each batch
        test_perc: Percentage of files to use for testing (0.0 to 1.0)
        num_docs: Number of files to load (0 for all files)
        input_dir: Directory containing the saved files

    Returns:
        Tuple of (train_batches_iterator, test_batches_iterator)
    """
    input_path = Path(input_dir)
    files = list(input_path.glob("*.txt"))

    if not files:
        logger.error("No .txt files found in directory: %s", input_dir)
        # Yield empty iterators to avoid errors downstream
        return iter([]), iter([])

    logger.info("Found %d .txt files in directory: %s", len(files), input_dir)

    if num_docs > 0:
        files = files[:num_docs]
        logger.info("Using %d files for processing", len(files))

    # Shuffle files for random train-test split
    random.shuffle(files)

    # Calculate split point
    split_idx = int(len(files) * (1 - test_perc))
    train_files = files[:split_idx]
    test_files = files[split_idx:]

    logger.info(
        "Split into %d training and %d test files",
        len(train_files),
        len(test_files),
    )

    def batch_generator(file_list: List[Path]) -> Iterator[List[str]]:
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


def load_all_documents(
    num_docs: int,
    input_dir: str = "data/raw_reports",
) -> List[str]:
    """
    Load all documents for cross-validation.

    Args:
        num_docs: Number of documents to load (0 for all)
        input_dir: Directory containing the documents

    Returns:
        List of document texts
    """
    logger = logging.getLogger(__name__)
    input_path = Path(input_dir)

    if not input_path.exists():
        raise FileNotFoundError(f"Input directory {input_dir} not found")

    # Get all text files
    files = list(input_path.glob("*.txt"))
    if not files:
        raise ValueError(f"No .txt files found in {input_dir}")

    logger.info(f"Found {len(files)} .txt files in directory: {input_dir}")

    # Limit number of files if specified
    if num_docs > 0:
        files = files[:num_docs]
        logger.info(f"Using {len(files)} files for processing")

    # Load all documents
    documents = []
    for file_path in tqdm(files, desc="Loading documents"):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if content:  # Only add non-empty documents
                    documents.append(content)
        except Exception as e:
            logger.warning(f"Failed to read {file_path}: {e}")

    logger.info(f"Successfully loaded {len(documents)} documents")
    return documents


def save_cv_results(cv_results: Dict[str, Any], output_dir: Path) -> None:
    """
    Save cross-validation results to files.

    Args:
        cv_results: CV results dictionary
        output_dir: Output directory
    """
    import json
    import pickle

    logger = logging.getLogger(__name__)

    try:
        # Save summary as JSON
        summary_file = output_dir / "cv_summary.json"
        with open(summary_file, "w") as f:
            json.dump(cv_results["summary"], f, indent=2, default=str)

        # Save full results as pickle (includes models if saved)
        results_file = output_dir / "cv_results.pkl"
        with open(results_file, "wb") as f:
            pickle.dump(cv_results, f)

        # Save readable summary
        summary_txt = output_dir / "cv_summary.txt"
        with open(summary_txt, "w") as f:
            f.write("CROSS-VALIDATION SUMMARY\n")
            f.write("=" * 60 + "\n")

            config = cv_results["config"]
            f.write("Configuration:\n")
            f.write(f"  Topics: {config['num_topics']}\n")
            f.write(f"  Folds: {config['n_splits']}\n")
            f.write(f"  Documents: {config['total_documents']}\n")
            f.write(
                f"  Successful folds: {cv_results['summary']['successful_folds']}\n\n"
            )

            summary = cv_results["summary"]
            for split in ["train", "test"]:
                if split in summary:
                    f.write(f"{split.upper()} SET METRICS:\n")
                    split_summary = summary[split]

                    # Perplexity
                    if "perplexity" in split_summary:
                        p = split_summary["perplexity"]
                        f.write(
                            f"  Perplexity: {p['mean']:.2f} ± {p['std']:.2f} "
                            f"[{p['min']:.2f}, {p['max']:.2f}] (n={p['count']})\n"
                        )

                    # Coherence scores
                    for coherence_type in ["c_v", "u_mass", "c_npmi"]:
                        metric_name = f"coherence_{coherence_type}"
                        if metric_name in split_summary:
                            c = split_summary[metric_name]
                            f.write(
                                f"  {metric_name}: {c['mean']:.3f} ± {c['std']:.3f} "
                                f"[{c['min']:.3f}, {c['max']:.3f}] (n={c['count']})\n"
                            )
                    f.write("\n")

        logger.info(f"CV results saved to {output_dir}")

    except Exception as e:
        logger.error(f"Failed to save CV results: {e}")
        raise
