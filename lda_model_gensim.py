# Description: This file is used to test the LDA model.

from typing import Tuple, Dict, List, Any, Union
from gensim.models import CoherenceModel
from gensim.models import ldamodel
from gensim.corpora import Dictionary
from tqdm import tqdm
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

logger = logging.getLogger(__name__)


def model_training(
    topic_num: int,
    corpus: List[List[Tuple[int, int]]],
    id2word: Dictionary,
    model_params: Dict[str, Any] = None,
) -> ldamodel.LdaModel:
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

    # Override n_components with topic_num
    params["num_topics"] = topic_num

    try:
        lda_model = ldamodel.LdaModel(
            corpus=corpus,
            id2word=id2word,
            **params,
        )
        return lda_model
    except Exception as e:
        logger.error(f"Failed to train LDA model: {str(e)}")
        raise


def performance_metrics(
    model: ldamodel.LdaModel,
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
    if not corpus or not texts:
        raise ValueError("corpus and texts cannot be empty")

    # Compute Perplexity
    perplexity = model.log_perplexity(corpus)

    # Compute Coherence Score
    coherence_metrics = {
        "u_mass": None,
        "c_v": None,
        "c_uci": None,
        "c_npmi": None,
    }

    for metric in coherence_metrics:
        try:
            coherence_model = CoherenceModel(
                model=model,
                texts=texts,
                dictionary=id2word,
                coherence=metric,
            )
            coherence_metrics[metric] = coherence_model.get_coherence()
        except Exception as e:
            logger.warning(f"Failed to compute {metric} coherence: {str(e)}")
            coherence_metrics[metric] = None

    return perplexity, coherence_metrics


def _train_single_model(
    args: Tuple[
        int,
        List[List[Tuple[int, int]]],
        Dictionary,
        Dict[str, Any],
        List[List[str]],
    ],
) -> Tuple[ldamodel.LdaModel, float, int]:
    """
    Helper function to train a single LDA model with given parameters.
    This needs to be at module level for multiprocessing to work.

    Args:
        args: Tuple containing (num_topics, corpus, id2word, model_params, texts)

    Returns:
        Tuple of (trained model, perplexity, num_topics)
    """
    num_topics, corpus, id2word, model_params, texts = args
    model = model_training(num_topics, corpus, id2word, model_params)
    perplexity, _ = performance_metrics(model, corpus, texts, id2word)
    return model, perplexity, num_topics


def optimize_topic_number(
    corpus: List[List[Tuple[int, int]]],
    id2word: Dictionary,
    texts: List[List[str]],
    topic_range: Dict[str, int],
    model_params: Dict[str, Any] = None,
) -> Tuple[List[ldamodel.LdaModel], List[float]]:
    """
    Find optimal number of topics using perplexity scores.

    Args:
        corpus: Document corpus in bow format
        id2word: Dictionary mapping word IDs to words
        texts: List of tokenized documents
        topic_range: Dictionary with start, limit, and step for topic numbers
        model_params: Additional model parameters

    Returns:
        Tuple containing:
        - List of trained LDA models
        - List of perplexity scores

    Raises:
        ValueError: If topic_range parameters are invalid
    """
    if not all(k in topic_range for k in ["start", "limit", "step"]):
        raise ValueError("topic_range must contain 'start', 'limit', and 'step'")
    if topic_range["start"] <= 0 or topic_range["limit"] <= topic_range["start"]:
        raise ValueError("Invalid topic range parameters")

    perplexity_scores = []
    models = []

    topic_numbers = range(
        topic_range["start"],
        topic_range["limit"],
        topic_range["step"],
    )

    # Prepare arguments for parallel processing
    train_args = [(n, corpus, id2word, model_params, texts) for n in topic_numbers]

    # Determine number of processes (use max of 8 or number of CPU cores)
    num_processes = min(8, mp.cpu_count())

    # Use ProcessPoolExecutor for parallel processing
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        # Submit all training tasks
        future_to_topic = {
            executor.submit(_train_single_model, args): args[0] for args in train_args
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
                    f"Model training failed for {future_to_topic[future]} topics: {str(e)}"
                )

    # Sort results by number of topics to maintain order
    results.sort(key=lambda x: x[0])

    # Unpack sorted results
    models = [r[1] for r in results]
    perplexity_scores = [r[2] for r in results]

    if not models:
        raise ValueError("Failed to train any models")

    return models, perplexity_scores


if __name__ == "__main__":
    pass
