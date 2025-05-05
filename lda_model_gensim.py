"""
This file contains the functions for training the LDA model using Gensim.
"""

from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from typing import Tuple, Dict, List, Any
import logging
import multiprocessing as mp
import numpy as np
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel
from gensim import models

logger = logging.getLogger(__name__)

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)


def model_training(
    topic_num: int,
    corpus: List[List[Tuple[int, int]]],
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
    if not corpus:
        raise ValueError("corpus cannot be empty")

    if model_params is None:
        model_params = {}

    # Create a copy of model_params to avoid modifying the original dict
    params = model_params.copy()

    # Override num_topics with topic_num
    params["num_topics"] = topic_num

    try:
        lda_model = models.LdaModel(
            corpus=corpus,
            id2word=id2word,
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
) -> Tuple[float, Dict[str, float]]:
    """
    Calculate model performance metrics.

    Args:
        model: Trained LDA model
        corpus: Document corpus in bow format
        texts: List of tokenized documents
        id2word: Dictionary mapping word IDs to words

    Returns:
        Tuple containing:
        - perplexity (float): Model perplexity score
        - coherence_metrics (Dict[str, float]): Dictionary of coherence scores

    Raises:
        ValueError: If corpus or texts are empty
    """

    # Compute Perplexity
    perplexity = np.exp2(
        -model.log_perplexity(corpus)
    )  # https://github.com/piskvorky/gensim/blob/develop/gensim/models/ldamodel.py

    # # Compute Coherence Score
    # coherence_metrics = {
    #     # "u_mass": None,
    #     # "c_uci": None,
    #     # "c_npmi": None,
    #     "c_v": None,
    # }

    # for metric in coherence_metrics:
    #     try:
    #         coherence_model = CoherenceModel(
    #             model=model,
    #             texts=texts,
    #             dictionary=id2word,
    #             coherence=metric,
    #         )
    #         coherence_metrics[metric] = coherence_model.get_coherence()
    #     except Exception as e:
    #         logger.warning(
    #             "Failed to compute %s coherence: %s",
    #             metric,
    #             str(e),
    #         )
    #         coherence_metrics[metric] = None

    # return perplexity, coherence_metrics
    return perplexity


def _train_single_model(
    args: Tuple[
        int,
        List[List[Tuple[int, int]]],
        Dictionary,
        Dict[str, Any],
        List[List[str]],
        List[List[Tuple[int, int]]],  # test_corpus
    ],
) -> Tuple[models.LdaModel, float, int]:
    """
    Helper function to train a single LDA model with given parameters.
    This needs to be at module level for multiprocessing to work.

    Args:
        args: Tuple containing (num_topics, train_corpus, id2word, model_params, texts, test_corpus)

    Returns:
        Tuple of (trained model, perplexity on test set, num_topics)
    """
    num_topics, train_corpus, id2word, model_params, texts, test_corpus = args
    model = model_training(num_topics, train_corpus, id2word, model_params)
    perplexity = performance_metrics(model, test_corpus, texts, id2word)
    return model, perplexity, num_topics


def optimize_topic_number(
    train_corpus: List[List[Tuple[int, int]]],
    id2word: Dictionary,
    texts: List[List[str]],
    topic_range: Dict[str, int],
    num_cores: int,
    model_params: Dict[str, Any] = None,
    test_corpus: List[List[Tuple[int, int]]] = None,
) -> Tuple[List[models.LdaModel], List[float]]:
    """
    Find optimal number of topics using perplexity scores.

    Args:
        train_corpus: Training document corpus in bow format
        id2word: Dictionary mapping word IDs to words
        texts: List of tokenized documents
        topic_range: Dictionary with start, limit, and step for topic numbers
        model_params: Additional model parameters
        test_corpus: Test document corpus in bow format for evaluation

    Returns:
        Tuple containing:
        - List of trained LDA models
        - List of perplexity scores on test set

    Raises:
        ValueError: If topic_range parameters are invalid
    """
    if not all(k in topic_range for k in ["start", "limit", "step"]):
        raise ValueError("topic_range must contain 'start', 'limit', and 'step'")
    if topic_range["start"] <= 0 or topic_range["limit"] <= topic_range["start"]:
        raise ValueError("Invalid topic range parameters")

    perplexity_scores = []
    models_perplexity_scores = []

    topic_numbers = range(
        topic_range["start"],
        topic_range["limit"],
        topic_range["step"],
    )

    # Prepare arguments for parallel processing
    train_args = [
        (
            n,
            train_corpus,
            id2word,
            model_params,
            texts,
            test_corpus,
        )
        for n in topic_numbers
    ]

    # Determine number of processes (use max of 8 or number of CPU cores)
    num_processes = min(num_cores, mp.cpu_count())

    # Use ProcessPoolExecutor for parallel processing
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        # Submit all training tasks
        future_to_topic = {
            executor.submit(
                _train_single_model,
                args,
            ): args[0]
            for args in train_args
        }

        # Process results as they complete
        results = []
        for future in tqdm(
            as_completed(future_to_topic),
            total=len(topic_numbers),
            desc="Optimizing number of topics",
        ):
            try:
                model, perplexity, num_topics = future.result()
                results.append((num_topics, model, perplexity))
            except Exception as e:
                logger.error(
                    "Model training failed for %s topics: %s",
                    future_to_topic[future],
                    str(e),
                )

    # Sort results by number of topics to maintain order
    results.sort(key=lambda x: x[0])

    # Unpack sorted results
    models = [r[1] for r in results]
    perplexity_scores = [r[2] for r in results]

    if not models:
        raise ValueError("Failed to train any models")

    return models, perplexity_scores
