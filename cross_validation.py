"""
Cross-validation functionality for LDA topic modeling.
"""

import logging
import numpy as np
from sklearn.model_selection import KFold
from typing import List, Dict, Tuple, Any
import lda_model_gensim
from data_preprocessing import lda_preprocessing as data_preprocessing

logger = logging.getLogger(__name__)


def cross_validate_lda(
    documents: List[str],
    config: Dict[str, Any],
    num_topics: int,
    num_cores: int = 8,
    n_splits: int = 5,
    compute_coherence: bool = True,
    random_state: int = 100,
) -> Dict[str, Any]:
    """
    Perform k-fold cross-validation for LDA model evaluation.

    Args:
        documents: List of document texts
        config: Configuration dictionary
        num_topics: Number of topics for LDA model
        num_cores: Number of CPU cores to use
        n_splits: Number of CV folds
        compute_coherence: Whether to compute coherence metrics
        random_state: Random seed for reproducibility

    Returns:
        Dictionary containing CV results and statistics
    """
    logger.info(f"Starting {n_splits}-fold cross-validation with {len(documents)} documents")

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    cv_results = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(documents)):
        logger.info(f"Processing fold {fold + 1}/{n_splits}")

        # Split documents
        train_docs = [documents[i] for i in train_idx]
        test_docs = [documents[i] for i in test_idx]

        logger.info(f"Fold {fold + 1}: {len(train_docs)} train, {len(test_docs)} test documents")

        try:
            # Preprocess training data
            def train_doc_generator():
                yield train_docs

            train_dict, train_bow_corpus, train_tfidf_corpus, train_texts = (
                data_preprocessing.pre_processing_gensim(
                    documents_generator=train_doc_generator(),
                    config=config["preprocessing"],
                    num_cores=num_cores,
                    mode="train",
                )
            )

            # Preprocess test data
            def test_doc_generator():
                yield test_docs

            test_texts = data_preprocessing.pre_processing_gensim(
                documents_generator=test_doc_generator(),
                config=config["preprocessing"],
                num_cores=num_cores,
                mode="test",
            )[3]

            # Filter test corpus using training dictionary
            test_bow_corpus, test_tfidf_corpus = data_preprocessing.test_corpus_filtering(
                dic=train_dict,
                test_texts=test_texts,
            )

            # Train LDA model
            model = lda_model_gensim.model_training(
                topic_num=num_topics,
                corpus=train_bow_corpus,
                id2word=train_dict,
                model_params=config["lda"]["gensim"],
            )

            # Evaluate model
            train_metrics = lda_model_gensim.performance_metrics(
                model=model,
                corpus=train_bow_corpus,
                texts=train_texts,
                id2word=train_dict,
                compute_coherence=compute_coherence,
            )

            test_metrics = lda_model_gensim.performance_metrics(
                model=model,
                corpus=test_bow_corpus,
                texts=train_texts,  # Use train texts for coherence consistency
                id2word=train_dict,
                compute_coherence=compute_coherence,
            )

            fold_result = {
                "fold": fold + 1,
                "train_docs": len(train_docs),
                "test_docs": len(test_docs),
                "vocab_size": len(train_dict),
                "train_metrics": train_metrics,
                "test_metrics": test_metrics,
            }

            cv_results.append(fold_result)

            logger.info(f"Fold {fold + 1} completed - "
                       f"Test perplexity: {test_metrics.get('perplexity', 'N/A'):.2f}, "
                       f"Train coherence_c_v: {train_metrics.get('coherence_c_v', 'N/A')}")

        except Exception as e:
            logger.error(f"Fold {fold + 1} failed: {e}")
            continue

    # Calculate summary statistics
    summary = calculate_cv_summary(cv_results)

    return {
        "cv_results": cv_results,
        "summary": summary,
        "config": {
            "n_splits": n_splits,
            "num_topics": num_topics,
            "total_documents": len(documents),
            "random_state": random_state,
        }
    }


def calculate_cv_summary(cv_results: List[Dict]) -> Dict[str, Any]:
    """Calculate summary statistics from cross-validation results."""
    if not cv_results:
        return {"error": "No successful CV folds"}

    summary = {}

    # Extract metrics from all folds
    train_perplexities = [r["train_metrics"].get("perplexity") for r in cv_results
                         if r["train_metrics"].get("perplexity") is not None]
    test_perplexities = [r["test_metrics"].get("perplexity") for r in cv_results
                        if r["test_metrics"].get("perplexity") is not None]

    # Coherence metrics
    coherence_types = ["c_v", "u_mass", "c_npmi"]

    for split in ["train", "test"]:
        summary[split] = {}

        # Perplexity statistics
        perplexities = train_perplexities if split == "train" else test_perplexities
        if perplexities:
            summary[split]["perplexity"] = {
                "mean": np.mean(perplexities),
                "std": np.std(perplexities),
                "min": np.min(perplexities),
                "max": np.max(perplexities),
                "count": len(perplexities),
            }

        # Coherence statistics
        for coherence_type in coherence_types:
            metric_name = f"coherence_{coherence_type}"
            coherence_scores = [r[f"{split}_metrics"].get(metric_name) for r in cv_results
                              if r[f"{split}_metrics"].get(metric_name) is not None]

            if coherence_scores:
                summary[split][metric_name] = {
                    "mean": np.mean(coherence_scores),
                    "std": np.std(coherence_scores),
                    "min": np.min(coherence_scores),
                    "max": np.max(coherence_scores),
                    "count": len(coherence_scores),
                }

    summary["successful_folds"] = len(cv_results)

    return summary


def print_cv_summary(cv_summary: Dict[str, Any]) -> None:
    """Print formatted cross-validation summary."""
    summary = cv_summary["summary"]
    config = cv_summary["config"]

    print(f"\n{'='*60}")
    print(f"CROSS-VALIDATION SUMMARY")
    print(f"{'='*60}")
    print(f"Configuration:")
    print(f"  Topics: {config['num_topics']}")
    print(f"  Folds: {config['n_splits']}")
    print(f"  Documents: {config['total_documents']}")
    print(f"  Successful folds: {summary['successful_folds']}")

    for split in ["train", "test"]:
        if split in summary:
            print(f"\n{split.upper()} SET METRICS:")
            split_summary = summary[split]

            # Perplexity
            if "perplexity" in split_summary:
                p = split_summary["perplexity"]
                print(f"  Perplexity: {p['mean']:.2f} ± {p['std']:.2f} "
                      f"[{p['min']:.2f}, {p['max']:.2f}] (n={p['count']})")

            # Coherence scores
            for coherence_type in ["c_v", "u_mass", "c_npmi"]:
                metric_name = f"coherence_{coherence_type}"
                if metric_name in split_summary:
                    c = split_summary[metric_name]
                    print(f"  {metric_name}: {c['mean']:.3f} ± {c['std']:.3f} "
                          f"[{c['min']:.3f}, {c['max']:.3f}] (n={c['count']})")

    print(f"{'='*60}\n")