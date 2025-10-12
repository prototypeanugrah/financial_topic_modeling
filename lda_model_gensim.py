"""
This file contains the functions for training the LDA model using Gensim.
"""

import logging
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numexpr as ne
import numpy as np
from gensim import models
from gensim.corpora import Dictionary

# from gensim.models.callbacks import PerplexityMetric  # Available but not used in current implementation
from tqdm import tqdm

logger = logging.getLogger(__name__)


def model_training(
    topic_num: int,
    train_corpus: List[List[Tuple[int, int]]],
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

    try:
        lda_model = models.LdaModel(
            corpus=train_corpus,
            id2word=id2word,
            eval_every=None,
            **params,
        )
        return lda_model
    except Exception as e:
        logger.error("Failed to train LDA model: %s", str(e))
        raise


def performance_metrics(
    model: models.LdaModel,
    corpus: List[List[Tuple[int, int]]],
) -> np.float64:
    """
    Calculate model performance metrics.

    Args:
        model: Trained LDA model
        corpus: Document corpus in bow format

    Returns:
        Perplexity score as a numpy float64

    Raises:
        ValueError: If corpus is empty
    """

    # Compute Perplexity (primary metric)
    try:
        perplexity = np.exp2(-model.log_perplexity(corpus))
    except Exception as e:
        logger.error(f"Failed to compute perplexity: {e}")
        perplexity = np.inf

    return perplexity


def per_doc_log_per_word(
    model: models.LdaModel,
    corpus: List[List[Tuple[int, int]]],
) -> List[float]:
    """
    Return a list of per-word log-likelihood values (one per document).
    Uses gensim's log_perplexity on single-doc iterables.
    """
    if not corpus:
        return []
    return [model.log_perplexity([bow]) for bow in corpus]


def perplexity_macro(
    model: models.LdaModel,
    corpora_by_type: Dict[str, List[List[Tuple[int, int]]]],
    base: str = "2",
) -> Tuple[float, Dict[str, float], Dict[str, float]]:
    """
    Compute a type-balanced (macro) perplexity over multiple corpora (e.g., IPO vs Analyst).

    Args:
        model: Trained LDA model
        corpora_by_type: Mapping from a type label (e.g., "IPO", "Analyst")
                         to its corpus (list of BoW docs)
        base: "2" for perplexity = 2^(-avg log p_w); "e" for exp(-avg log p_w)

    Returns:
        (ppx_macro, per_type_ppx, per_type_log_per_word)
        - ppx_macro: macro perplexity across types (unweighted mean over types)
        - per_type_ppx: dict of perplexity per type
        - per_type_log_per_word: dict of mean per-word log-likelihood per type
    """
    if base not in {"2", "e"}:
        raise ValueError("base must be '2' or 'e'")
    to_ppx = np.exp2 if base == "2" else np.exp

    per_type_lpw: Dict[str, float] = {}
    per_type_ppx: Dict[str, float] = {}

    for name, c in corpora_by_type.items():
        if not c:
            continue
        lpws = per_doc_log_per_word(model, c)
        mean_lpw = float(np.mean(lpws)) if lpws else float("-inf")
        per_type_lpw[name] = mean_lpw
        per_type_ppx[name] = float(to_ppx(-mean_lpw))

    if not per_type_lpw:
        return float("inf"), {}, {}

    lp_macro = float(np.mean(list(per_type_lpw.values())))
    ppx_macro = float(to_ppx(-lp_macro))
    return ppx_macro, per_type_ppx, per_type_lpw


def _train_single_model(
    args: Tuple[
        int,  # num_topics
        List[List[Tuple[int, int]]],  # train_corpus
        List[List[Tuple[int, int]]],  # val_corpus
        Optional[
            Dict[str, List[List[Tuple[int, int]]]]
        ],  # val_corpora_by_type (macro eval)
        Dictionary,  # id2word
        Dict[str, Any],  # model_params
        List[int],  # random seeds
    ],
) -> Tuple[
    models.LdaModel,
    np.float64,
    np.float64,
    int,
    List[Dict[str, Any]],
]:
    """
    Helper function to train a single LDA model with given parameters.
    This needs to be at module level for multiprocessing to work.

    Args:
        args: Tuple containing (num_topics, train_corpus, test_corpus,
        id2word, model_params, random_seeds)

    Returns:
        Tuple containing the best seed model, averaged train perplexity,
        averaged test perplexity, num_topics, and per-seed metrics
    """
    (
        num_topics,
        train_corpus,
        val_corpus,
        val_corpora_by_type,
        id2word,
        model_params,
        random_seeds,
    ) = args

    try:
        seed_metrics: List[Dict[str, Any]] = []
        train_scores: List[float] = []
        val_scores: List[float] = []
        best_seed_model: Optional[models.LdaModel] = None
        best_seed_val_perplexity = np.inf

        for seed in random_seeds:
            params = (model_params or {}).copy()
            params["random_state"] = seed

            model = model_training(
                topic_num=num_topics,
                train_corpus=train_corpus,
                id2word=id2word,
                model_params=params,
            )

            train_perplexity = performance_metrics(
                model=model,
                corpus=train_corpus,
            )
            # Prefer macro perplexity if per-type corpora are provided; else fallback to micro.
            per_type_ppx: Dict[str, float] = {}
            if val_corpora_by_type:
                val_perplexity, per_type_ppx, _ = perplexity_macro(
                    model=model,
                    corpora_by_type=val_corpora_by_type,
                    base="2",
                )
            else:
                val_perplexity = performance_metrics(
                    model=model,
                    corpus=val_corpus,
                )

            seed_metric: Dict[str, Any] = {
                "seed": seed,
                "train_perplexity": float(train_perplexity),
                "val_perplexity": float(val_perplexity),
            }
            if per_type_ppx:
                seed_metric["val_perplexity_by_type"] = {
                    name: float(ppx) for name, ppx in per_type_ppx.items()
                }
            seed_metrics.append(seed_metric)
            train_scores.append(float(train_perplexity))
            val_scores.append(float(val_perplexity))

            if val_perplexity < best_seed_val_perplexity:
                if best_seed_model is not None:
                    del best_seed_model
                best_seed_model = model
                best_seed_val_perplexity = val_perplexity
            else:
                del model

        if not seed_metrics:
            raise RuntimeError("Failed to train any models for the provided seeds.")

        avg_train_perplexity = float(np.min(train_scores))
        avg_val_perplexity = float(np.min(val_scores))

        # # Save model to disk if path provided (memory optimization)
        # if save_path:
        #     from pathlib import Path
        #
        #     save_dir = Path(save_path)
        #     save_dir.mkdir(parents=True, exist_ok=True)
        #     model_file = save_dir / f"lda_model_{num_topics}_topics.gz"
        #     model.save(str(model_file))
        #     logger.debug(f"Saved model with {num_topics} topics to {model_file}")
        #     # Return None instead of model to save memory
        #     return None, metrics, num_topics

        return (
            best_seed_model,
            avg_train_perplexity,
            avg_val_perplexity,
            num_topics,
            seed_metrics,
        )
    except Exception as e:
        logger.error(f"Failed to train model with {num_topics} topics: {e}")
        return (
            None,
            np.inf,
            np.inf,
            num_topics,
            [],
        )


def optimize_topic_number(
    train_corpus: List[List[Tuple[int, int]]],
    val_corpus: List[List[Tuple[int, int]]],
    id2word: Dictionary,
    topic_range: Dict[str, int],
    num_cores: int,
    model_params: Dict[str, Any] = None,
    random_seeds: Optional[List[int]] = None,
    save_models: bool = False,
    save_dir: str = None,
    val_corpora_by_type: Optional[Dict[str, List[List[Tuple[int, int]]]]] = None,
) -> Tuple[models.LdaModel, Dict[int, Dict[str, Any]], int]:
    """
    Find optimal number of topics using perplexity scores with memory optimization.

    Args:
        train_corpus: Training document corpus
        id2word: Dictionary mapping
        topic_range: Topic range parameters
        num_cores: Number of CPU cores
        model_params: Model parameters
        random_seeds: List of seeds to average over for random_state
        val_corpus: Validation corpus for evaluation
        save_models: Whether to save models to disk
        save_dir: Directory to save models

    Returns:
        Tuple of (best_model, all_metrics, best_num_topics) where all_metrics maps
        each topic count to averaged/per-seed train and val perplexities.

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
            topic_range["limit"] + 1,  # +1 because range is exclusive
            topic_range["step"],
        )
    )

    if not random_seeds:
        random_seeds = [42, 100, 3]
    else:
        random_seeds = list(random_seeds)

    # Prepare arguments for parallel processing
    train_args = [
        (
            n,
            train_corpus,
            val_corpus,
            val_corpora_by_type,
            id2word,
            model_params,
            random_seeds,
        )
        for n in topic_numbers
    ]

    # Determine number of cores (cap at 8 for memory)
    num_cores = min(num_cores, mp.cpu_count()) if num_cores else mp.cpu_count()

    # Set NumExpr thread count to match our process limit
    ne.set_num_threads(min(num_cores, mp.cpu_count()))

    # Track results
    all_metrics = {}
    best_model = None
    best_perplexity = np.inf
    best_num_topics = topic_numbers[0]

    with ProcessPoolExecutor(max_workers=num_cores) as executor:
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
                (
                    model,
                    train_metrics,
                    val_metrics,
                    num_topics,
                    seed_metrics,
                ) = future.result()

                # Store metrics - initialize the dict for this num_topics
                # Dict structure:
                #   {
                #       num_topics: {
                #           "train": {"average": avg, "per_seed": {seed: value}},
                #           "test": {"average": avg, "per_seed": {seed: value}},
                #       }
                #   }
                per_seed_train = {
                    metric["seed"]: metric["train_perplexity"]
                    for metric in seed_metrics
                }
                per_seed_val = {
                    metric["seed"]: metric["val_perplexity"] for metric in seed_metrics
                }
                per_type_avg: Dict[str, float] = {}
                if val_corpora_by_type:
                    per_type_collect: Dict[str, List[float]] = {}
                    for metric in seed_metrics:
                        for type_name, ppx in (
                            metric.get("val_perplexity_by_type", {}) or {}
                        ).items():
                            per_type_collect.setdefault(type_name, []).append(ppx)
                    per_type_avg = {
                        name: float(np.mean(values))
                        for name, values in per_type_collect.items()
                    }
                all_metrics[num_topics] = {
                    "train": {
                        "average": train_metrics,
                        "per_seed": per_seed_train,
                    },
                    "val": {
                        "average": val_metrics,
                        "per_seed": per_seed_val,
                        "per_type_average": per_type_avg,
                    },
                }
                current_train_perplexity = train_metrics
                current_val_perplexity = val_metrics

                seed_val_details = ", ".join(
                    f"seed {seed}: {perplexity}"
                    for seed, perplexity in sorted(per_seed_val.items())
                )
                per_type_details = ""
                if per_type_avg:
                    per_type_details = " | Val Perplexity by type: " + ", ".join(
                        f"{type_name}: {ppx}"
                        for type_name, ppx in sorted(per_type_avg.items())
                    )

                print(
                    f"Number of topics: {num_topics}, "
                    f"Train Perplexity (avg): {current_train_perplexity}, "
                    f"Val Perplexity (avg): {current_val_perplexity}"
                    + (
                        f" | Val Perplexity by seed: {seed_val_details}"
                        if seed_val_details
                        else ""
                    )
                    + per_type_details
                )

                # Track best model (based on test perplexity)
                # Update best model if test perplexity is better by at least 10% than best so far
                if current_val_perplexity < best_perplexity:
                    # Delete previous best model from memory
                    if best_model is not None:
                        del best_model

                    # Load or keep new best model
                    if save_models and model is None:
                        # Model was saved to disk, load it
                        from pathlib import Path

                        model_file = Path(save_dir) / f"lda_model_{num_topics}_topics"
                        best_model = models.LdaModel.load(str(model_file))
                    else:
                        best_model = model

                    best_perplexity = current_val_perplexity
                    best_num_topics = num_topics

                    logger.info(
                        f"New best model: {num_topics} topics, "
                        f"avg_train_perplexity={current_train_perplexity:.2f}, "
                        f"avg_val_perplexity={current_val_perplexity:.2f}, "
                    )
                else:
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
    corpus: List[List[Tuple[int, int]]],
    output_dir: Path,
    prefix: str,
    mode: str,
) -> None:
    """
    Calculate document topic distribution with optional prefix.

    Args:
        model: Trained LDA model
        corpus: Corpus in bow format
        output_dir: Output directory
        prefix: Optional prefix for output files

    Returns:
        None
    """
    document_topics = model.get_document_topics(corpus)

    with open(
        output_dir / f"{prefix}_document_topics_{mode}.txt",
        "w",
        encoding="utf-8",
    ) as f:
        for document_topic in document_topics:
            f.write(f"{document_topic}\n")
