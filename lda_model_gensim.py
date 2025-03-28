# Description: This file is used to test the LDA model.

from typing import Tuple, Dict, List, Any, Union
from gensim.models import CoherenceModel
from gensim.models import ldamodel
from gensim.corpora import Dictionary
from tqdm import tqdm
import logging

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
    coherence_type: str = "u_mass",
) -> Tuple[float, Dict[str, float]]:
    """
    Calculate model performance metrics.

    Args:
        model: Trained LDA model
        corpus: Document corpus in bow format
        texts: List of tokenized documents
        id2word: Dictionary mapping word IDs to words
        coherence_type: Type of coherence metric to use

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

    for num_topics in tqdm(topic_numbers, desc="Optimizing number of topics"):
        try:
            model = model_training(num_topics, corpus, id2word, model_params)
            perplexity = model.log_perplexity(corpus)
            perplexity_scores.append(perplexity)
            models.append(model)
        except Exception as e:
            logger.error(f"Failed to train model with {num_topics} topics: {str(e)}")
            continue

    if not models:
        raise ValueError("Failed to train any models")

    return models, perplexity_scores


if __name__ == "__main__":
    pass
