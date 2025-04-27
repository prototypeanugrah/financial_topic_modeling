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

# from gensim.models import ldamodel
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

    # Override n_components with topic_num
    params["num_topics"] = topic_num

    try:
        lda_model = models.LdaModel(
            corpus=corpus,
            id2word=id2word,
            # eval_every=1,
            **params,
        )
        return lda_model
    except Exception as e:
        logger.error("Failed to train LDA model: %s", str(e))
        raise


def performance_metrics(
    # model: ldamodel.LdaModel,
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
    if not corpus or not texts:
        raise ValueError("corpus and texts cannot be empty")

    # Compute Perplexity
    perplexity = model.log_perplexity(corpus)

    # Compute Coherence Score
    coherence_metrics = {
        # "u_mass": None,
        # "c_uci": None,
        # "c_npmi": None,
        "c_v": None,
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
            logger.warning(
                "Failed to compute %s coherence: %s",
                metric,
                str(e),
            )
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
) -> Tuple[models.LdaModel, float, int]:
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
) -> Tuple[List[models.LdaModel], List[float]]:
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
    train_args = [
        (
            n,
            corpus,
            id2word,
            model_params,
            texts,
        )
        for n in topic_numbers
    ]

    # Determine number of processes (use max of 8 or number of CPU cores)
    num_processes = min(16, mp.cpu_count())

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


def jaccard_similarity(
    topic_1: List[str],
    topic_2: List[str],
) -> float:
    """
    Derives the Jaccard similarity of two topics

    Jaccard similarity:
    - A statistic used for comparing the similarity and diversity of sample sets
    - J(A,B) = (A ∩ B)/(A ∪ B)
    - Goal is low Jaccard scores for coverage of the diverse elements
    """
    intersection = set(topic_1).intersection(set(topic_2))
    union = set(topic_1).union(set(topic_2))

    return float(len(intersection)) / float(len(union))


def _train_single_lda_model(
    args: Tuple[
        int, List[List[Tuple[int, float]]], Dict[int, str], Dict[str, Any], int
    ],
) -> Tuple[models.LdaModel, List[List[str]]]:
    """
    Helper function to train a single LDA model with given parameters.
    This needs to be at module level for multiprocessing to work.

    Args:
        args: Tuple containing (num_topics, corpus, id2word, model_params, num_keywords)

    Returns:
        Tuple of (trained model, list of topic keywords)
    """
    num_topics, corpus, id2word, model_params, num_keywords = args
    model = model_training(num_topics, corpus, id2word, model_params)
    shown_topics = model.show_topics(
        num_topics=num_topics, num_words=num_keywords, formatted=False
    )
    topics = [[word[0] for word in topic[1]] for topic in shown_topics]
    return model, topics


def _calculate_single_coherence(
    args: Tuple[models.LdaModel, List[List[str]], Dict[int, str]],
) -> float:
    """
    Helper function to calculate coherence for a single model.
    This needs to be at module level for multiprocessing to work.

    Args:
        args: Tuple containing (model, texts, id2word)

    Returns:
        Coherence score for the model
    """
    model, texts, id2word = args
    coherence = CoherenceModel(
        model=model,
        texts=texts,
        dictionary=id2word,
        coherence="c_v",
    ).get_coherence()
    return coherence


def train_multiple_lda_models(
    num_topics: List[int],
    corpus: List[List[Tuple[int, float]]],
    id2word: Dict[int, str],
    model_params: Dict[str, Any],
    num_cores: int,
    num_keywords: int,
) -> Tuple[Dict[int, models.LdaModel], Dict[int, List[List[str]]]]:
    """
    Train multiple LDA models with different numbers of topics and extract their topics.

    Args:
        num_topics: List of different topic numbers to try
        corpus: TF-IDF transformed document corpus
        id2word: Dictionary mapping word IDs to words
        model_params: Model parameters from config
        num_cores: Number of cores to use for parallel processing
        num_keywords: Number of keywords to extract per topic (default: 10)

    Returns:
        Tuple containing:
        - Dictionary mapping number of topics to trained LDA models
        - Dictionary mapping number of topics to list of topic keywords
    """
    try:
        if isinstance(num_topics, int):
            num_topics = list(range(num_topics))[1:]
    except Exception as e:
        logger.error("Error converting num_topics to list: %s", str(e))
        raise

    # Check if corpus is empty
    if not corpus:
        logger.error("Empty corpus provided. Cannot train LDA models.")
        return {}, {}

    # Check if id2word is empty
    if not id2word:
        logger.error("Empty id2word dictionary provided. Cannot train LDA models.")
        return {}, {}

    models = {}
    topics = {}

    # Prepare arguments for parallel processing
    train_args = [
        (
            n,
            corpus,
            id2word,
            model_params,
            num_keywords,
        )
        for n in num_topics
        if n > 0
    ]

    # Determine number of processes (use max of 8 or number of CPU cores)
    num_processes = min(num_cores, mp.cpu_count())

    # Use ProcessPoolExecutor for parallel processing
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        # Submit all training tasks
        future_to_topic = {
            executor.submit(_train_single_lda_model, args): args[0]
            for args in train_args
        }

        # Process results as they complete
        results = []
        for future in tqdm(
            as_completed(future_to_topic),
            total=len(num_topics),
            desc="Training LDA models",
        ):
            try:
                model, topic_keywords = future.result()
                results.append((future_to_topic[future], model, topic_keywords))
            except Exception as e:
                logger.error(
                    "Model training failed for %s topics: %s",
                    future_to_topic[future],
                    str(e),
                )

    # Sort results by number of topics to maintain order
    results.sort(key=lambda x: x[0])

    # Unpack sorted results
    for topic_num, model, topic_keywords in results:
        models[topic_num] = model
        topics[topic_num] = topic_keywords

    return models, topics


def calculate_lda_model_stability(
    topics: Dict[int, List[List[str]]],
    num_topics: List[int],
) -> Tuple[Dict[int, List[List[float]]], List[float]]:
    """
    Calculate stability between consecutive LDA models using Jaccard similarity.

    This function compares topics between models with consecutive numbers of topics
    to measure how stable the topic structure remains as the number of topics changes.

    Args:
        topics: Dictionary mapping number of topics to list of topic keywords
                   Each topic is a list of strings (keywords)
        num_topics: List of topic numbers in ascending order

    Returns:
        Tuple containing:
        - Dictionary mapping number of topics to matrix of Jaccard similarities
          with the next model's topics
        - List of mean stability scores for each number of topics (except the last)

    Example:
        If num_topics = [5, 10, 15], the function will compare:
        - 5-topic model with 10-topic model
        - 10-topic model with 15-topic model
        And return stability scores for models with 5 and 10 topics.
    """
    # Check if topics dictionary is empty
    if not topics:
        logger.error("Empty topics dictionary provided. Cannot calculate stability.")
        return {}, []

    # Check if num_topics list is empty or has less than 2 elements
    if not num_topics or len(num_topics) < 2:
        logger.error(
            "Invalid num_topics list. Need at least 2 topic numbers to calculate stability."
        )
        return {}, []

    stability = {}

    # Compare each model with the next one
    for i in range(0, len(num_topics) - 1):
        jaccard_sims = []

        # Check if both current and next topic numbers exist in the topics dictionary
        if num_topics[i] not in topics or num_topics[i + 1] not in topics:
            logger.warning(
                f"Missing topics for {num_topics[i]} or {num_topics[i+1]}. Skipping comparison."
            )
            continue

        # Compare each topic in current model
        for t1, topic1 in enumerate(topics[num_topics[i]]):
            # Calculate similarities with all topics in next model
            sims = []
            for t2, topic2 in enumerate(topics[num_topics[i + 1]]):
                sims.append(jaccard_similarity(topic1, topic2))

            jaccard_sims.append(sims)

        # Store similarity matrix for this number of topics
        stability[num_topics[i]] = jaccard_sims

    # Calculate mean stability for each number of topics
    mean_stabilities = [
        np.array(stability[i]).mean() for i in num_topics[:-1] if i in stability
    ]

    return stability, mean_stabilities


def calculate_coherence_multiple_lda_models(
    models: Dict[int, models.LdaModel],
    corpus: List[List[Tuple[int, float]]],
    id2word: Dict[int, str],
    num_topics: List[int],
    num_cores: int,
) -> List[float]:
    """
    Calculate coherence scores for multiple LDA models.

    Args:
        models: Dictionary mapping number of topics to trained LDA models
        corpus: TF-IDF transformed document corpus
        id2word: Dictionary mapping word IDs to words
        num_topics: List of topic numbers in ascending order

    Returns:
        List of coherence scores for each number of topics
    """
    # Prepare arguments for parallel processing
    coherence_args = [
        (models[n], corpus, id2word)
        for n in num_topics[:-1]  # Exclude last topic number as per original code
    ]

    # Determine number of processes (use max of 8 or number of CPU cores)
    num_processes = min(num_cores, mp.cpu_count())

    # Use ProcessPoolExecutor for parallel processing
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        # Submit all coherence calculation tasks
        future_to_topic = {
            executor.submit(_calculate_single_coherence, args): i
            for i, args in enumerate(coherence_args)
        }

        # Process results as they complete
        results = []
        for future in tqdm(
            as_completed(future_to_topic),
            total=len(coherence_args),
            desc="Calculating coherence scores",
        ):
            try:
                coherence = future.result()
                results.append((future_to_topic[future], coherence))
            except Exception as e:
                logger.error(
                    "Coherence calculation failed for model %d: %s",
                    future_to_topic[future],
                    str(e),
                )

    # Sort results by model index to maintain order
    results.sort(key=lambda x: x[0])

    # Extract coherence scores in order
    coherences = [r[1] for r in results]

    return coherences


def calculate_stats_plot(
    coherences: List[float],
    mean_stabilities: List[float],
    num_topics: List[int],
    num_keywords: int,
) -> int:
    coh_sta_diffs = [
        coherences[i] - mean_stabilities[i] for i in range(num_keywords)[:-1]
    ]  # limit topic numbers to the number of keywords
    coh_sta_max = max(coh_sta_diffs)
    coh_sta_max_idxs = [i for i, j in enumerate(coh_sta_diffs) if j == coh_sta_max]
    ideal_topic_num_index = coh_sta_max_idxs[
        0
    ]  # choose less topics in case there's more than one max
    ideal_topic_num = num_topics[ideal_topic_num_index]

    return ideal_topic_num


if __name__ == "__main__":
    pass
