"""
Utilities for training LDA models with Gensim focused on train/test evaluation.
"""

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from gensim import models
from gensim.corpora import Dictionary

logger = logging.getLogger(__name__)


def model_training(
    topic_num: int,
    train_corpus: List[List[Tuple[int, int]]],
    id2word: Dictionary,
    model_params: Optional[Dict[str, Any]] = None,
) -> models.LdaModel:
    """
    Train an LDA model on the provided corpus.
    """
    if topic_num <= 0:
        raise ValueError("topic_num must be positive")
    if not train_corpus:
        raise ValueError("train_corpus cannot be empty")

    params = (model_params or {}).copy()
    params["num_topics"] = topic_num

    try:
        lda_model = models.LdaModel(
            corpus=train_corpus,
            id2word=id2word,
            eval_every=None,
            **params,
        )
        return lda_model
    except Exception as exc:
        logger.error("Failed to train LDA model: %s", exc)
        raise


def performance_metrics(
    model: models.LdaModel,
    corpus: List[List[Tuple[int, int]]],
) -> float:
    """
    Compute perplexity for a corpus using a trained model.
    """
    if not corpus:
        raise ValueError("corpus cannot be empty when evaluating perplexity")

    try:
        return float(np.exp2(-model.log_perplexity(corpus)))
    except Exception as exc:
        logger.error("Failed to compute perplexity: %s", exc)
        return float("inf")


def train_best_seed_for_fold(
    num_topics: int,
    train_corpus: List[List[Tuple[int, int]]],
    test_corpus: Optional[List[List[Tuple[int, int]]]],
    id2word: Dictionary,
    model_params: Optional[Dict[str, Any]],
    random_seeds: List[int],
) -> Dict[str, Any]:
    """
    Train multiple LDA models (one per seed) and retain the model with the lowest
    training perplexity. Evaluate the retained model on the provided test corpus.
    """
    if not train_corpus:
        raise ValueError("Training corpus is empty; cannot fit LDA model.")
    if not random_seeds:
        random_seeds = [42]

    per_seed_train: Dict[int, float] = {}
    best_seed: Optional[int] = None
    best_train_ppx: float = float("inf")
    best_model: Optional[models.LdaModel] = None

    for seed in random_seeds:
        params = (model_params or {}).copy()
        params["random_state"] = seed

        model = model_training(
            topic_num=num_topics,
            train_corpus=train_corpus,
            id2word=id2word,
            model_params=params,
        )

        train_ppx = float(performance_metrics(model=model, corpus=train_corpus))
        per_seed_train[seed] = train_ppx

        is_better = train_ppx < best_train_ppx or (
            math.isclose(train_ppx, best_train_ppx)
            and best_seed is not None
            and seed < best_seed
        )

        if is_better or best_seed is None:
            if best_model is not None:
                del best_model
            best_model = model
            best_seed = seed
            best_train_ppx = train_ppx
        else:
            del model

    if best_model is None or best_seed is None:
        raise RuntimeError("Failed to train any LDA models for the provided seeds.")

    test_ppx: Optional[float] = None
    if test_corpus:
        non_empty_test = [doc for doc in test_corpus if doc]
        if non_empty_test:
            test_ppx = float(
                performance_metrics(model=best_model, corpus=non_empty_test)
            )
        else:
            logger.warning("Test corpus is empty after filtering; skipping perplexity.")

    return {
        "model": best_model,
        "best_seed": best_seed,
        "train_perplexity": best_train_ppx,
        "test_perplexity": test_ppx,
        "per_seed_train_perplexity": per_seed_train,
    }
