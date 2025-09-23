"""
This file contains the functions for training the LDA model using Gensim.
"""

import logging
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numexpr as ne
import numpy as np
from gensim import models
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel
from gensim.models.callbacks import PerplexityMetric
from tqdm import tqdm

logger = logging.getLogger(__name__)


def model_training(
    topic_num: int,
    train_corpus: List[List[Tuple[int, int]]],
    test_corpus: List[List[Tuple[int, int]]],
    id2word: Dictionary,
    model_params: Dict[str, Any] = None,
) -> models.LdaModel:
    """
    Train LDA model with configured parameters.

    Args:
        topic_num: Number of topics
        corpus: Document corpus in bow format
        id2word: Dictionary mapping word IDs to words
        model_params: Model parameters from config

    Returns:
        ldamodel.LdaModel: Trained LDA model

    Raises:
        ValueError: If topic_num is not positive or corpus is empty
    """
    if topic_num <= 0:
        raise ValueError("topic_num must be positive")
    if not train_corpus:
        raise ValueError("corpus cannot be empty")

    if model_params is None:
        model_params = {}

    # Create a copy of model_params to avoid modifying the original dict
    params = model_params.copy()

    # Override num_topics with topic_num
    params["num_topics"] = topic_num

    if test_corpus is not None:
        perplexity_callback = PerplexityMetric(
            corpus=test_corpus,
            logger="shell",
        )

    try:
        lda_model = models.LdaModel(
            corpus=train_corpus,
            callbacks=[perplexity_callback] if test_corpus is not None else None,
            id2word=id2word,
            eval_every=1,
            **params,
        )
        return lda_model
    except Exception as e:
        logger.error("Failed to train LDA model: %s", str(e))
        raise


def performance_metrics(
    model: models.LdaModel,
    corpus: List[List[Tuple[int, int]]],
    texts: List[List[str]],
    id2word: Dictionary,
    compute_coherence: bool = True,
    coherence_types: List[str] = None,
) -> Dict[str, float]:
    """
    Calculate model performance metrics.

    Args:
        model: Trained LDA model
        corpus: Document corpus in bow format
        texts: List of tokenized documents
        id2word: Dictionary mapping word IDs to words
        compute_coherence: Whether to compute coherence metrics
        coherence_types: List of coherence types to compute

    Returns:
        Dictionary containing all metrics

    Raises:
        ValueError: If corpus or texts are empty
    """
    metrics = {}

    # Compute Perplexity (primary metric)
    try:
        metrics["perplexity"] = np.exp2(-model.log_perplexity(corpus))
    except Exception as e:
        logger.error(f"Failed to compute perplexity: {e}")
        metrics["perplexity"] = float("inf")

    # Compute Coherence Scores (secondary metrics for reporting)
    if compute_coherence and texts:
        if coherence_types is None:
            coherence_types = ["c_v"]

        for coherence_type in coherence_types:
            try:
                if coherence_type == "u_mass":
                    # u_mass uses corpus, not texts
                    coherence_model = CoherenceModel(
                        model=model,
                        corpus=corpus,
                        dictionary=id2word,
                        coherence=coherence_type,
                    )
                else:
                    # c_v, c_npmi use texts
                    coherence_model = CoherenceModel(
                        model=model,
                        texts=texts,
                        dictionary=id2word,
                        coherence=coherence_type,
                    )
                coherence_score = coherence_model.get_coherence()
                # Check for invalid values (NaN, inf, or extremely low values)
                if (
                    coherence_score is None
                    or not isinstance(coherence_score, (int, float))
                    or coherence_score != coherence_score
                    or abs(coherence_score) == float("inf")
                ):
                    logger.warning(
                        f"Invalid {coherence_type} coherence score: {coherence_score}"
                    )
                    metrics[f"coherence_{coherence_type}"] = None
                else:
                    metrics[f"coherence_{coherence_type}"] = coherence_score
            except (ZeroDivisionError, RuntimeWarning, ValueError) as e:
                logger.warning(
                    f"Failed to compute {coherence_type} coherence due to sparse vocabulary: {e}"
                )
                metrics[f"coherence_{coherence_type}"] = None
            except Exception as e:
                logger.warning(
                    f"Unexpected error computing {coherence_type} coherence: {e}"
                )
                metrics[f"coherence_{coherence_type}"] = None

    return metrics


def _train_single_model(
    args: Tuple[
        int,
        List[List[Tuple[int, int]]],
        Dictionary,
        Dict[str, Any],
        List[List[str]],
        List[List[Tuple[int, int]]],  # train_corpus
        List[List[Tuple[int, int]]],  # test_corpus
        bool,  # compute_coherence flag
        str,  # Optional save path
    ],
) -> Tuple[models.LdaModel, Dict[str, float], int]:
    """
    Helper function to train a single LDA model with given parameters.
    This needs to be at module level for multiprocessing to work.

    Args:
        args: Tuple containing (num_topics, train_corpus, id2word, model_params, texts, test_corpus, compute_coherence, save_path)

    Returns:
        Tuple of (trained model or None, metrics dict, num_topics)
    """
    (
        num_topics,
        train_corpus,
        id2word,
        model_params,
        texts,
        train_corpus,
        test_corpus,
        compute_coherence,
        save_path,
    ) = args

    try:
        # Train model
        model = model_training(
            topic_num=num_topics,
            train_corpus=train_corpus,
            test_corpus=test_corpus,
            id2word=id2word,
            model_params=model_params,
        )

        # Compute metrics on test set
        metrics = performance_metrics(
            model=model,
            corpus=test_corpus,
            texts=texts,
            id2word=id2word,
            compute_coherence=compute_coherence,
        )

        # # Save model to disk if path provided (memory optimization)
        # if save_path:
        #     from pathlib import Path

        #     save_dir = Path(save_path)
        #     save_dir.mkdir(parents=True, exist_ok=True)
        #     model_file = save_dir / f"lda_model_{num_topics}_topics.gz"
        #     model.save(str(model_file))
        #     logger.debug(f"Saved model with {num_topics} topics to {model_file}")
        #     # Return None instead of model to save memory
        #     return None, metrics, num_topics

        return model, metrics, num_topics

    except Exception as e:
        logger.error(f"Failed to train model with {num_topics} topics: {e}")
        return None, {"perplexity": float("inf")}, num_topics


def optimize_topic_number(
    train_corpus: List[List[Tuple[int, int]]],
    test_corpus: List[List[Tuple[int, int]]],
    id2word: Dictionary,
    texts: List[List[str]],
    topic_range: Dict[str, int],
    num_cores: int,
    model_params: Dict[str, Any] = None,
    save_models: bool = False,
    save_dir: str = None,
    compute_coherence: bool = True,
) -> Tuple[models.LdaModel, Dict[int, Dict[str, float]], int]:
    """
    Find optimal number of topics using perplexity scores with memory optimization.

    Args:
        train_corpus: Training document corpus
        id2word: Dictionary mapping
        texts: Tokenized documents
        topic_range: Topic range parameters
        num_cores: Number of CPU cores
        model_params: Model parameters
        test_corpus: Test corpus for evaluation
        save_models: Whether to save models to disk
        save_dir: Directory to save models
        compute_coherence: Whether to compute coherence metrics

    Returns:
        Tuple of (best_model, all_metrics, best_num_topics)

    Raises:
        ValueError: If topic_range parameters are invalid
    """
    if not all(k in topic_range for k in ["start", "limit", "step"]):
        raise ValueError("topic_range must contain 'start', 'limit', and 'step'")
    if topic_range["start"] <= 0 or topic_range["limit"] <= topic_range["start"]:
        raise ValueError("Invalid topic range parameters")

    # Setup save directory if needed
    if save_models and save_dir:
        from pathlib import Path

        Path(save_dir).mkdir(parents=True, exist_ok=True)

    topic_numbers = list(
        range(
            topic_range["start"],
            topic_range["limit"],
            topic_range["step"],
        )
    )

    # Prepare arguments for parallel processing
    train_args = [
        (
            n,
            train_corpus,
            id2word,
            model_params,
            texts,
            train_corpus,
            test_corpus,
            compute_coherence,
            save_dir if save_models else None,
        )
        for n in topic_numbers
    ]

    # Determine number of processes (cap at 8 for memory)
    num_processes = min(num_cores, mp.cpu_count(), 8)

    # Set NumExpr thread count to match our process limit
    ne.set_num_threads(min(num_processes, 8))

    # Track results
    all_metrics = {}
    best_model = None
    best_perplexity = float("inf")
    best_num_topics = topic_numbers[0]

    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        # Submit all training tasks
        future_to_topic = {
            executor.submit(_train_single_model, args): args[0] for args in train_args
        }

        # Process results as they complete
        for future in tqdm(
            as_completed(future_to_topic),
            total=len(topic_numbers),
            desc="Optimizing number of topics",
        ):
            try:
                model, metrics, num_topics = future.result()

                # Store metrics
                all_metrics[num_topics] = metrics
                current_perplexity = metrics.get("perplexity", float("inf"))

                # Track best model (based on perplexity only)
                if current_perplexity < best_perplexity:
                    # Delete previous best model from memory
                    if best_model is not None:
                        del best_model

                    # Load or keep new best model
                    if save_models and model is None:
                        # Model was saved to disk, load it
                        from pathlib import Path

                        model_file = (
                            Path(save_dir) / f"lda_model_{num_topics}_topics.gz"
                        )
                        best_model = models.LdaModel.load(str(model_file))
                    else:
                        best_model = model

                    best_perplexity = current_perplexity
                    best_num_topics = num_topics

                    logger.info(
                        f"New best model: {num_topics} topics, "
                        f"perplexity={current_perplexity:.2f}, "
                        f"coherence_c_v={metrics.get('coherence_c_v', 'N/A')}"
                    )
                elif model is not None:
                    # Not the best model, free memory
                    del model

            except Exception as e:
                logger.error(f"Model training failed: {e}")

    if best_model is None:
        raise ValueError("Failed to train any models successfully")

    logger.info(
        f"Optimization complete. Best: {best_num_topics} topics "
        f"(perplexity={best_perplexity:.2f})"
    )

    return best_model, all_metrics, best_num_topics


def document_topic_distribution(
    model: models.LdaModel,
    train_corpus: List[List[Tuple[int, int]]],
    test_corpus: List[List[Tuple[int, int]]],
    output_dir: Path,
) -> None:
    """
    Calculate document topic distribution.

    Args:
        model: Trained LDA model
        corpus: Document corpus in bow format

    Returns:
        None
    """
    document_topics_train = model.get_document_topics(train_corpus)
    document_topics_test = model.get_document_topics(test_corpus)
    with open(output_dir / "document_topics_train.txt", "w", encoding="utf-8") as f:
        for document_topic in document_topics_train:
            f.write(f"{document_topic}\n")
    with open(output_dir / "document_topics_test.txt", "w", encoding="utf-8") as f:
        for document_topic in document_topics_test:
            f.write(f"{document_topic}\n")
    return document_topics_train, document_topics_test
