import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from gensim import corpora

import lda_model_gensim
import test_lda_preprocessing
import utils

# Setup logging - modify to only show INFO level
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logging.getLogger("gensim").setLevel(logging.ERROR)  # For gensim
logger = logging.getLogger(__name__)


def arguments_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Cross-Validated Topic Modeling for IPO and Analyst Reports",
    )
    parser.add_argument(
        "-c",
        "--config",
        default="config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "-k",
        "--num_topics",
        type=int,
        required=False,
        help=(
            "Number of topics to train. "
            "If omitted, the topic range from the config is used."
        ),
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=5,
        help="Number of cross-validation folds.",
    )
    parser.add_argument(
        "--cv_random_state",
        type=int,
        default=42,
        help="Random state used when shuffling folds for cross-validation.",
    )
    parser.add_argument(
        "--no_ngrams",
        action="store_true",
        help="Disable bigram/trigram modeling; use unigram tokens only.",
    )
    parser.add_argument(
        "--output_subdir",
        type=str,
        default=None,
        help=(
            "Optional subdirectory under the configured output base directory "
            "for saving run artefacts (e.g., 'comparison/with_ngrams')."
        ),
    )
    return parser.parse_args()


def _determine_topic_numbers(
    args: argparse.Namespace,
    config: Dict[str, Dict],
) -> List[int]:
    if args.num_topics:
        return [args.num_topics]
    topic_cfg = config["lda"]["topic_range"]
    return list(range(topic_cfg["start"], topic_cfg["limit"] + 1, topic_cfg["step"]))


def _aggregate_metrics(
    results: List[Dict[str, Optional[float]]],
    topic_numbers: Sequence[int],
) -> Dict[int, Dict[str, Optional[float]]]:
    aggregated: Dict[int, Dict[str, Optional[float]]] = {}
    for topic in topic_numbers:
        topic_results = [record for record in results if record["num_topics"] == topic]
        train_vals = [
            record["train_perplexity"]
            for record in topic_results
            if record["train_perplexity"] is not None
        ]
        test_vals = [
            record["test_perplexity"]
            for record in topic_results
            if record["test_perplexity"] is not None
        ]

        def _stats(values: List[float]) -> Tuple[Optional[float], Optional[float]]:
            if not values:
                return None, None
            arr = np.array(values, dtype=float)
            return float(arr.mean()), float(arr.std(ddof=1)) if len(arr) > 1 else 0.0

        train_mean, train_std = _stats(train_vals)
        test_mean, test_std = _stats(test_vals)

        aggregated[topic] = {
            "train_mean": train_mean,
            "train_std": train_std,
            "test_mean": test_mean,
            "test_std": test_std,
            "folds_contributed": len(topic_results),
        }
    return aggregated


def _serialize_cv_results(
    output_dir: Path,
    metrics: Dict[str, Any],
    csv_rows: List[Dict[str, Any]],
) -> None:
    json_path = output_dir / "cross_validation_metrics.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    csv_path = output_dir / "cross_validation_metrics.csv"
    if csv_rows:
        fieldnames = list(csv_rows[0].keys())
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in csv_rows:
                writer.writerow(row)


def _train_full_model_and_export(
    pairs: List[test_lda_preprocessing.PreprocessedPair],
    combined_order_path: Path,
    best_topic: int,
    use_ngrams: bool,
    config: Dict[str, Any],
    output_dir: Path,
) -> None:
    try:
        with open(combined_order_path, "r", encoding="utf-8") as f:
            order_dict = json.load(f)
    except FileNotFoundError:
        logger.warning(
            "Combined order dictionary not found at %s; skipping full-model export.",
            combined_order_path,
        )
        return

    sorted_labels = [
        label
        for _, label in sorted(
            order_dict.items(), key=lambda item: int(item[0].replace("doc", ""))
        )
    ]

    ipo_tokens_map = {pair.pair_id: pair.ipo_tokens for pair in pairs}
    analyst_tokens_map = {pair.pair_id: pair.analyst_tokens for pair in pairs}

    texts: List[List[str]] = []
    labels: List[str] = []

    for label in sorted_labels:
        parts = label.split("_")
        if len(parts) < 2:
            continue
        try:
            pair_id = int(parts[1])
        except ValueError:
            continue

        if label.lower().startswith("ipo"):
            tokens = ipo_tokens_map.get(pair_id)
        else:
            tokens = analyst_tokens_map.get(pair_id)

        if tokens is None:
            continue

        texts.append(list(tokens))
        labels.append(label)

    if not texts:
        logger.warning(
            "No documents available for full-model training; skipping export."
        )
        return

    if use_ngrams:
        bigram_mod, trigram_mod = test_lda_preprocessing.train_ngram_models(texts=texts)
        texts = test_lda_preprocessing.apply_ngram_models(
            texts=texts,
            bigram_mod=bigram_mod,
            trigram_mod=trigram_mod,
        )

    dictionary = corpora.Dictionary(texts)
    filter_params = config.get("preprocessing", {}).get("filter_extremes")
    if filter_params:
        dictionary.filter_extremes(
            no_below=filter_params.get("no_below"),
            no_above=filter_params.get("no_above"),
            keep_n=filter_params.get("keep_n"),
        )
        if len(dictionary) == 0:
            dictionary = corpora.Dictionary(texts)

    corpus = [dictionary.doc2bow(doc) for doc in texts]
    if not corpus:
        logger.warning(
            "Full corpus is empty after dictionary creation; skipping export."
        )
        return

    lda_params = config["lda"]["gensim"]["params"].copy()
    random_seeds = config["lda"]["gensim"].get("random_seeds", [42])
    lda_params["random_state"] = random_seeds[0]

    final_model = lda_model_gensim.model_training(
        topic_num=best_topic,
        train_corpus=corpus,
        id2word=dictionary,
        model_params=lda_params,
    )

    topic_word_probs = final_model.get_topics()
    header = ["shared_dict"] + [
        dictionary[token_id] for token_id in range(len(dictionary))
    ]

    rows: List[List[str]] = []

    export_dir = output_dir / "full_model_export"
    export_dir.mkdir(parents=True, exist_ok=True)

    topic_word_distribution = {
        f"topic_{topic_idx}": [
            [float(prob), dictionary[token_id]]
            for token_id, prob in enumerate(word_probs)
        ]
        for topic_idx, word_probs in enumerate(topic_word_probs)
    }

    doc_topic_distribution = {}
    for label, bow in zip(labels, corpus):
        doc_topics = final_model.get_document_topics(bow, minimum_probability=0.0)
        doc_topic_distribution[label] = [
            [float(prob), int(topic_id)] for topic_id, prob in doc_topics
        ]

    with open(
        export_dir / f"topic_word_distribution_{best_topic}_topics.json",
        "w",
        encoding="utf-8",
    ) as fh:
        json.dump(topic_word_distribution, fh, indent=2)

    with open(
        export_dir / f"doc_topic_distribution_{best_topic}_topics.json",
        "w",
        encoding="utf-8",
    ) as fh:
        json.dump(doc_topic_distribution, fh, indent=2)

    for label, bow in zip(labels, corpus):
        if not bow:
            continue
        doc_topics = final_model.get_document_topics(bow, minimum_probability=0.0)
        if not doc_topics:
            continue
        dominant_topic = max(doc_topics, key=lambda item: item[1])[0]
        word_probs = topic_word_probs[dominant_topic]
        formatted_probs = [f"{prob:.6f}" for prob in word_probs]
        rows.append([label] + formatted_probs)

    csv_path = export_dir / f"dominant_topic_word_probs_{best_topic}_topics.csv"

    with open(csv_path, "w", encoding="utf-8", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(header)
        writer.writerows(rows)

    logger.info(
        "Full dataset model trained with %d topics; dominant topic CSV saved to %s",
        best_topic,
        csv_path,
    )


def run_cross_validation(
    args: argparse.Namespace,
    config: Dict[str, Dict],
) -> None:
    try:
        from sklearn.model_selection import KFold
    except ImportError as exc:
        raise ImportError(
            "scikit-learn is required for cross-validation. "
            "Install scikit-learn or run without --cross_validate."
        ) from exc

    topic_numbers = _determine_topic_numbers(args, config)
    lda_params = config["lda"]["gensim"]["params"]
    random_seeds = config["lda"]["gensim"].get("random_seeds", [42, 100, 3])
    use_ngrams = not args.no_ngrams

    preprocessing_cfg = config.get("preprocessing", {})
    batch_size = preprocessing_cfg.get("batch_size", 200)
    num_cores = preprocessing_cfg.get("num_cores_preprocessing", 2)
    num_docs_limit = preprocessing_cfg.get("num_docs", 0)
    pair_metadata_path = preprocessing_cfg.get(
        "pair_metadata_path",
        "data/final_analyst_reports_for_latest_s1_filings_extracted.csv",
    )

    output_dir = utils.setup_output_directory(config, subdir=args.output_subdir)
    model_save_dir = output_dir / "models_cv"
    model_save_dir.mkdir(parents=True, exist_ok=True)

    pair_records = utils.load_pair_records(
        pair_metadata_path,
        limit=num_docs_limit if num_docs_limit else 0,
    )
    preprocessed_pairs = test_lda_preprocessing.preprocess_pairs_for_cv(
        pair_records=pair_records,
        config=config,
        batch_size=batch_size,
        num_cores=num_cores,
        output_dir=output_dir,
        mode="cv",
    )
    pairs = preprocessed_pairs.pairs
    if not pairs:
        raise ValueError("No valid IPO/analyst pairs available after preprocessing.")
    if len(pairs) < args.folds:
        raise ValueError(
            f"Not enough pairs ({len(pairs)}) to perform {args.folds}-fold CV."
        )

    indices = np.arange(len(pairs))

    splitter = KFold(
        n_splits=args.folds,
        shuffle=True,
        random_state=args.cv_random_state,
    )
    fold_splits = list(splitter.split(indices))

    logger.info(
        "Running %d-fold cross-validation over %d pairs (n-grams: %s)",
        args.folds,
        len(pairs),
        "enabled" if use_ngrams else "disabled",
    )

    filter_params = preprocessing_cfg.get("filter_extremes")

    fold_results: List[Dict[str, Any]] = []
    csv_rows: List[Dict[str, Any]] = []

    for fold_idx, (train_indices, test_indices) in enumerate(fold_splits, start=1):
        train_pairs = [pairs[i] for i in train_indices]
        test_pairs = [pairs[i] for i in test_indices]

        logger.info(
            "Fold %d: %d train pairs (%d docs) | %d test pairs (%d docs)",
            fold_idx,
            len(train_pairs),
            len(train_pairs) * 2,
            len(test_pairs),
            len(test_pairs) * 2,
        )

        train_tokens: List[List[str]] = []
        for pair in train_pairs:
            train_tokens.append(list(pair.ipo_tokens))
            train_tokens.append(list(pair.analyst_tokens))

        if not train_tokens:
            raise ValueError(f"Fold {fold_idx} produced an empty training set.")

        if use_ngrams:
            bigram_mod, trigram_mod = test_lda_preprocessing.train_ngram_models(
                texts=train_tokens,
            )
            train_tokens = test_lda_preprocessing.apply_ngram_models(
                texts=train_tokens,
                bigram_mod=bigram_mod,
                trigram_mod=trigram_mod,
            )

        test_tokens: List[List[str]] = []
        for pair in test_pairs:
            test_tokens.append(list(pair.ipo_tokens))
            test_tokens.append(list(pair.analyst_tokens))

        if use_ngrams:
            test_tokens = test_lda_preprocessing.apply_ngram_models(
                texts=test_tokens,
                bigram_mod=bigram_mod,
                trigram_mod=trigram_mod,
            )

        dictionary, train_corpus = test_lda_preprocessing.create_dictionary(
            texts=train_tokens,
            output_dir=output_dir,
            mode=f"cv_fold{fold_idx}_train",
            filter_params=filter_params,
        )
        test_corpus = test_lda_preprocessing.corpus_filtering_from_dictionary(
            dic=dictionary,
            texts=test_tokens,
        )

        train_corpus_non_empty = [doc for doc in train_corpus if doc]
        test_corpus_non_empty = [doc for doc in test_corpus if doc]
        empty_train_docs = len(train_corpus) - len(train_corpus_non_empty)
        empty_test_docs = len(test_corpus) - len(test_corpus_non_empty)

        if empty_train_docs:
            logger.warning(
                "Fold %d: %d training documents empty after dictionary filtering.",
                fold_idx,
                empty_train_docs,
            )
        if empty_test_docs:
            logger.warning(
                "Fold %d: %d test documents empty after dictionary filtering.",
                fold_idx,
                empty_test_docs,
            )

        if not train_corpus_non_empty:
            raise ValueError(
                f"Fold {fold_idx}: dictionary filtering removed all tokens from the training corpus."
            )

        fold_prefix = f"cv_fold{fold_idx}"
        fold_preprocessed_dir = output_dir / "preprocessed"
        fold_preprocessed_dir.mkdir(parents=True, exist_ok=True)
        dictionary.save(
            str(fold_preprocessed_dir / f"{fold_prefix}_dictionary.id2word")
        )
        corpora.MmCorpus.serialize(
            str(fold_preprocessed_dir / f"{fold_prefix}_train_bow_corpus.mm"),
            train_corpus_non_empty,
        )
        if test_corpus_non_empty:
            corpora.MmCorpus.serialize(
                str(fold_preprocessed_dir / f"{fold_prefix}_test_bow_corpus.mm"),
                test_corpus_non_empty,
            )

        for num_topics in topic_numbers:
            try:
                result = lda_model_gensim.train_best_seed_for_fold(
                    num_topics=num_topics,
                    train_corpus=train_corpus_non_empty,
                    test_corpus=test_corpus_non_empty,
                    id2word=dictionary,
                    model_params=lda_params,
                    random_seeds=random_seeds,
                )
            except Exception as exc:
                logger.error(
                    "Fold %d | topics %d failed: %s", fold_idx, num_topics, exc
                )
                raise

            best_model = result.pop("model")
            best_seed = result["best_seed"]
            train_perplexity = result["train_perplexity"]
            test_perplexity = result["test_perplexity"]
            seed_train_details = ", ".join(
                f"seed {seed}: {perplexity:.4f}"
                for seed, perplexity in sorted(
                    result["per_seed_train_perplexity"].items()
                )
            )

            if config.get("output", {}).get("save_model"):
                model_path = model_save_dir / (
                    f"fold{fold_idx}_topics{num_topics}_seed{best_seed}.model"
                )
                model_path.parent.mkdir(parents=True, exist_ok=True)
                best_model.save(str(model_path))

            del best_model

            logger.info(
                "Fold %d | topics=%d | best seed=%d | train perplexity=%.4f | "
                "test perplexity=%s | train per seed: %s",
                fold_idx,
                num_topics,
                best_seed,
                train_perplexity,
                f"{test_perplexity:.4f}" if test_perplexity is not None else "N/A",
                seed_train_details,
            )

            record = {
                "fold": fold_idx,
                "num_topics": num_topics,
                "best_seed": best_seed,
                "train_perplexity": train_perplexity,
                "test_perplexity": test_perplexity,
                "dictionary_size": len(dictionary),
                "train_docs": len(train_corpus),
                "train_docs_effective": len(train_corpus_non_empty),
                "test_docs": len(test_corpus),
                "test_docs_effective": len(test_corpus_non_empty),
                "empty_train_docs": empty_train_docs,
                "empty_test_docs": empty_test_docs,
                "per_seed_train_perplexity": result["per_seed_train_perplexity"],
            }
            fold_results.append(record)

            csv_rows.append(
                {
                    "fold": fold_idx,
                    "num_topics": num_topics,
                    "best_seed": best_seed,
                    "train_perplexity": train_perplexity,
                    "test_perplexity": test_perplexity,
                    "dictionary_size": len(dictionary),
                    "train_docs": len(train_corpus),
                    "train_docs_effective": len(train_corpus_non_empty),
                    "test_docs": len(test_corpus),
                    "test_docs_effective": len(test_corpus_non_empty),
                    "empty_train_docs": empty_train_docs,
                    "empty_test_docs": empty_test_docs,
                    "per_seed_train_perplexity": json.dumps(
                        result["per_seed_train_perplexity"]
                    ),
                }
            )

    aggregated_metrics = _aggregate_metrics(fold_results, topic_numbers)
    metrics_payload = {
        "configuration": {
            "folds": args.folds,
            "topic_numbers": topic_numbers,
            "random_seeds": random_seeds,
        },
        "fold_results": fold_results,
        "aggregated": aggregated_metrics,
    }

    _serialize_cv_results(output_dir, metrics_payload, csv_rows)

    best_topic: Optional[int] = None
    best_topic_mean = float("inf")
    for topic in topic_numbers:
        topic_metrics = aggregated_metrics.get(topic)
        if not topic_metrics:
            continue
        mean_value = topic_metrics.get("test_mean")
        if mean_value is None:
            continue
        if mean_value < best_topic_mean:
            best_topic_mean = mean_value
            best_topic = topic

    if best_topic is not None and np.isfinite(best_topic_mean):
        logger.info(
            "Best topic count by average test perplexity: %d (%.4f)",
            best_topic,
            best_topic_mean,
        )
        _train_full_model_and_export(
            pairs=pairs,
            combined_order_path=preprocessed_pairs.order_metadata_path,
            best_topic=best_topic,
            use_ngrams=use_ngrams,
            config=config,
            output_dir=output_dir,
        )
    else:
        logger.warning(
            "Unable to determine best topic count from aggregated metrics; skipping full-model export."
        )

    if config.get("output", {}).get("save_visualizations"):
        test_topics: List[int] = []
        test_means: List[float] = []
        for topic in topic_numbers:
            topic_metrics = aggregated_metrics.get(topic)
            if not topic_metrics:
                continue
            mean_value = topic_metrics.get("test_mean")
            if mean_value is None:
                continue
            test_topics.append(topic)
            test_means.append(mean_value)
        if test_topics:
            utils.plot_cv_perplexity_scores(
                topic_numbers=test_topics,
                perplexity_scores=test_means,
                output_dir=output_dir,
                mode="test",
            )

    logger.info("Cross-validation complete. Metrics saved to %s", output_dir)
    for topic in topic_numbers:
        aggregates = aggregated_metrics.get(topic, {})
        logger.info(
            "Topics %d -> train mean %.4f (std %.4f), test mean %s (std %s)",
            topic,
            aggregates.get("train_mean") or float("nan"),
            aggregates.get("train_std") or float("nan"),
            (
                f"{aggregates.get('test_mean'):.4f}"
                if aggregates.get("test_mean") is not None
                else "N/A"
            ),
            (
                f"{aggregates.get('test_std'):.4f}"
                if aggregates.get("test_std") is not None
                else "N/A"
            ),
        )


def main() -> None:
    args = arguments_parser()
    config = utils.load_config(args.config)

    run_cross_validation(args, config)


if __name__ == "__main__":
    main()
