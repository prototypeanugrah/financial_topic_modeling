# Description: This file implements LDA topic modeling using scikit-learn.

from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Tuple, Dict, List, Any, Union
import multiprocessing as mp

from sklearn.decomposition import LatentDirichletAllocation
from tqdm import tqdm
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer


def model_training(
    topic_num: int,
    doc_term_matrix,
    model_params: Dict = None,
) -> LatentDirichletAllocation:
    """
    Train LDA model with configured parameters using scikit-learn.

    Args:
        topic_num: Number of topics
        doc_term_matrix: Document-term matrix (sparse matrix)
        model_params: Model parameters from config

    Returns:
        Trained LDA model
    """
    if model_params is None:
        model_params = {}

    # Create a copy of model_params to avoid modifying the original dict
    params = model_params.copy()

    # Override n_components with topic_num
    params["n_components"] = topic_num

    # Convert string 'None' to Python None for n_jobs parameter
    if "n_jobs" in params:
        if params["n_jobs"] == "None":
            params["n_jobs"] = None
        else:
            params["n_jobs"] = int(params["n_jobs"])

    # Create and train the model
    lda_model = LatentDirichletAllocation(**params)
    lda_model.fit(doc_term_matrix)

    return lda_model


def _train_single_model(
    args: Tuple[int, Union[csr_matrix, np.ndarray], Dict[str, Any]],
) -> Tuple[LatentDirichletAllocation, float, int]:
    """
    Helper function to train a single LDA model with given parameters.
    This needs to be at module level for multiprocessing to work.

    Args:
        args: Tuple containing (num_topics, doc_term_matrix, model_params)

    Returns:
        Tuple of (trained model, perplexity, num_topics)
    """
    num_topics, doc_term_matrix, model_params = args
    model = model_training(num_topics, doc_term_matrix, model_params)
    perplexity, _ = performance_metrics_sklearn(model, doc_term_matrix)
    return model, perplexity, num_topics


def performance_metrics_sklearn(
    model: LatentDirichletAllocation,
    doc_term_matrix,
) -> Tuple[float, Dict[str, float]]:
    """
    Calculate model performance metrics for sklearn LDA.

    Args:
        model (LatentDirichletAllocation): Trained LDA model
        doc_term_matrix: Document-term matrix (sparse matrix)

    Returns:
        perplexity (float): Perplexity of the model
        metrics (dict): Dictionary containing log-likelihood and perplexity
    """
    # Calculate log-likelihood
    log_likelihood = model.score(doc_term_matrix)

    # Calculate perplexity (exp(-1. * log_likelihood per word))
    # manual_perplexity = np.exp(-1.0 * log_likelihood / doc_term_matrix.sum())
    perplexity = np.exp(-1.0 * log_likelihood / doc_term_matrix.sum())
    # sklearn_perplexity = model.perplexity(doc_term_matrix)

    metrics = {
        "log_likelihood": float(log_likelihood),
        # "manual_perplexity": float(manual_perplexity),
        "perplexity": float(perplexity),
    }

    return perplexity, metrics


def optimize_topic_number(
    doc_term_matrix,
    topic_range: Dict,
    model_params: Dict = None,
) -> Tuple[List[LatentDirichletAllocation], List[float]]:
    """
    Find optimal number of topics using perplexity scores.

    Args:
        doc_term_matrix: Document-term matrix
        topic_range: Dictionary with start, limit, and step for topic numbers
        model_params: Additional model parameters

    Returns:
        Tuple of (list of models, list of perplexity scores)
    """
    perplexity_scores = []
    models = []

    topic_numbers = range(
        topic_range["start"],
        topic_range["limit"],
        topic_range["step"],
    )

    # Prepare arguments for parallel processing
    train_args = [(n, doc_term_matrix, model_params) for n in topic_numbers]

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
                print(
                    f"Model training failed for {future_to_topic[future]} topics: {str(e)}"
                )

    # Sort results by number of topics to maintain order
    results.sort(key=lambda x: x[0])

    # Unpack sorted results
    models = [r[1] for r in results]
    perplexity_scores = [r[2] for r in results]

    return models, perplexity_scores

    # for num_topics in tqdm(topic_numbers, desc="Optimizing number of topics"):
    #     # Train model with current number of topics
    #     model = model_training(num_topics, doc_term_matrix, model_params)

    #     # Calculate perplexity
    #     perplexity, _ = performance_metrics_sklearn(model, doc_term_matrix)

    #     perplexity_scores.append(perplexity)
    #     models.append(model)

    # return models, perplexity_scores


def print_topics(
    model: LatentDirichletAllocation,
    vectorizer: TfidfVectorizer,
    n_top_words: int = 10,
) -> List[List[str]]:
    """
    Print and return the top words for each topic.

    Args:
        model: Trained LDA model
        vectorizer: TfidfVectorizer used to create the document-term matrix
        n_top_words: Number of top words to show per topic

    Returns:
        List of lists containing top words for each topic
    """
    feature_names = vectorizer.get_feature_names_out()
    topics = []

    for _, topic in enumerate(model.components_):
        top_words = [feature_names[i] for i in topic.argsort()[: -n_top_words - 1 : -1]]
        topics.append(top_words)

    return topics


def get_document_topics(
    model: LatentDirichletAllocation,
    doc_term_matrix: Union[csr_matrix, np.ndarray],
) -> np.ndarray:
    """
    Get topic distribution for each document.

    Args:
        model: Trained LDA model
        doc_term_matrix: Document-term matrix (sparse matrix or numpy array)

    Returns:
        np.ndarray: Array of topic distributions for each document
    """
    return model.transform(doc_term_matrix)


if __name__ == "__main__":
    pass
