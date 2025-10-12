import logging
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Set, Tuple

import yaml

# Setup logging - modify to only show INFO level
logging.basicConfig(
    level=logging.INFO,
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


class BatchStream(Iterator[List[Tuple[str, str]]]):
    """
    Iterator wrapper that yields document batches and exposes the total number of batches.
    """

    def __init__(self, file_list: List[str], batch_size: int):
        self._file_list = file_list
        self._batch_size = batch_size
        self._cursor = 0
        self._total_batches = (
            math.ceil(len(file_list) / batch_size) if file_list and batch_size else 0
        )

    def __len__(self) -> int:
        return self._total_batches

    def __iter__(self) -> "BatchStream":
        return self

    @property
    def file_paths(self) -> List[str]:
        """Return the underlying file paths as a new list."""
        return list(self._file_list)

    def __next__(self) -> List[Tuple[str, str]]:
        if self._cursor >= len(self._file_list):
            raise StopIteration

        start_idx = self._cursor
        end_idx = min(self._cursor + self._batch_size, len(self._file_list))
        self._cursor = end_idx

        batch_files = self._file_list[start_idx:end_idx]
        batch_documents: List[Tuple[str, str]] = []

        for file_path in batch_files:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    if content.strip():
                        batch_documents.append((content, file_path))
                    else:
                        logger.warning("Empty file found: %s", file_path)
            except Exception as e:
                logger.error("Error loading %s: %s", file_path, str(e))

        if not batch_documents:
            logger.warning(
                "No valid documents in batch %d-%d",
                start_idx,
                end_idx,
            )
            return []

        return batch_documents

    def reset(self) -> None:
        """Reset cursor to allow re-iteration if needed."""
        self._cursor = 0


def extract_zero_indices(zero_docs: List[Tuple[str, str]]) -> Set[int]:
    indices: Set[int] = set()
    for doc_label, _ in zero_docs:
        try:
            _, raw_index = doc_label.split("_", maxsplit=1)
            indices.add(int(raw_index))
        except (ValueError, IndexError):
            logger.warning("Unable to parse document index from label '%s'", doc_label)
    return indices


def load_splits_from_paths(
    file_paths: List[str],
    batch_size: int,
    test_perc: float,
    num_docs: int,
    report_type: str,
) -> Tuple[
    Iterator[List[Tuple[str, str]]],
    Iterator[List[Tuple[str, str]]],
    Iterator[List[Tuple[str, str]]],
]:
    """
    Generator function to load files in batches from file paths with train-test split

    Args:
        file_paths: List of file paths to load
        batch_size: Number of files to load in each batch
        test_perc: Percentage of files to use for testing (0.0 to 1.0)
        num_docs: Number of files to load (0 for all files)
        report_type: Type of report ('ipo' or 'analyst')
    Returns:
        Tuple of (train_batches_iterator, val_batches_iterator, test_batches_iterator)
    """
    if not file_paths:
        logger.error("No file paths provided")
        return iter([]), iter([]), iter([])

    if num_docs > 0:
        file_paths = file_paths[:num_docs]

    # Calculate split point
    # Example: If test_perc is 0.1, then split_idx will be 0.9 of the (training + validation) files
    # Keep val_perc = test_perc. The validation files are generated from the (training + validation) files.
    # So the split will be like this:
    # Training/validation/test = 0.8/0.1/0.1

    if num_docs > 0:
        file_paths = file_paths[:num_docs]

    total_files = len(file_paths)

    # First split into (train + validation) vs test
    test_count = int(total_files * test_perc)
    train_val_count = total_files - test_count

    test_files = file_paths[train_val_count:]
    train_val_files = file_paths[:train_val_count]

    # Derive validation portion from the (train + validation) pool
    val_count = int(train_val_count * test_perc)
    train_count = train_val_count - val_count

    train_files = train_val_files[:train_count]
    val_files = train_val_files[train_count:]

    if total_files:
        logger.info(
            "%s - Split into %d train (%.1f%%), %d val (%.1f%%) and %d test (%.1f%%) files",
            report_type,
            len(train_files),
            (len(train_files) / total_files) * 100,
            len(val_files),
            (len(val_files) / total_files) * 100,
            len(test_files),
            (len(test_files) / total_files) * 100,
        )

    return (
        BatchStream(train_files, batch_size),
        BatchStream(val_files, batch_size),
        BatchStream(test_files, batch_size),
    )


def plot_perplexity_scores(
    topic_range: Dict[str, int],
    perplexity_scores: List[float],
    output_dir: Path,
    mode: str,
) -> None:
    """
    Plot and save perplexity scores with optional prefix.
    """
    logger = logging.getLogger(__name__)

    try:
        import matplotlib.pyplot as plt

        # Convert topic_range dict to list of actual topic numbers
        topic_numbers = list(
            range(
                topic_range["start"],
                topic_range["limit"] + 1,  # +1 because range is exclusive
                topic_range["step"],
            )
        )

        plt.figure(figsize=(10, 6))
        plt.plot(topic_numbers, perplexity_scores, marker="o")
        plt.title(f"Perplexity Scores vs Number of Topics ({mode.title()} Set)")
        plt.xlabel("Number of Topics")
        plt.ylabel("Perplexity Score")
        plt.grid(True, alpha=0.3)

        plot_path = output_dir / f"perplexity_plot_{mode}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()

    except Exception as e:
        logger.error(f"Failed to create perplexity plot: {e}")
        raise
