import argparse
import json
import logging
import os
from pathlib import Path

import gensim

import lda_model_gensim
import utils

# Setup logging - modify to only show INFO level
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logging.getLogger("gensim").setLevel(logging.ERROR)  # For gensim
logger = logging.getLogger(__name__)


def arguments_parser():
    parser = argparse.ArgumentParser(
        description="Run Topic Modeling Pipeline for Both IPO and Analyst Reports"
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
        help="Number of topics to run the topic modeling on. If not provided, the number of topics will be determined automatically.",
    )
    parser.add_argument(
        "-nc",
        "--num_cores_lda",
        type=int,
        required=False,
        default=2,
        help="Minimum number of cores to run the topic modeling on. If not provided, the number of cores will be determined automatically.",
    )
    parser.add_argument(
        "--optimize_topics",
        action="store_true",
        help="Whether to optimize the number of topics. If not provided, the number of topics will not be optimized.",
    )

    return parser.parse_args()


def main():
    # Parse arguments
    args = arguments_parser()

    # Load configuration
    config = utils.load_config(args.config)

    # load the train and val data from the preprocessed directory
    preprocessed_dir = Path("outputs/run_20251011_153357/preprocessed")
    model_save_dir = Path("outputs/run_20251011_153357/models_macro")

    # with open(preprocessed_dir / "train_texts.txt", "r") as f:
    #     train_texts = f.readlines()
    # with open(preprocessed_dir / "val_texts.txt", "r") as f:
    #     val_texts = f.readlines()
    # with open(preprocessed_dir / "test_texts.txt", "r") as f:
    #     test_texts = f.readlines()
    # with open(preprocessed_dir / "train_val_texts.txt", "r") as f:
    #     train_val_texts = f.readlines()

    train_dictionary = gensim.corpora.Dictionary.load(
        str(preprocessed_dir / "train_dictionary.id2word")
    )
    train_bow_corpus = gensim.corpora.MmCorpus(
        str(preprocessed_dir / "train_bow_corpus.mm")
    )
    val_bow_corpus = gensim.corpora.MmCorpus(
        str(preprocessed_dir / "val_bow_corpus.mm")
    )
    test_bow_corpus = gensim.corpora.MmCorpus(
        str(preprocessed_dir / "test_bow_corpus.mm")
    )
    train_val_bow_corpus = gensim.corpora.MmCorpus(
        str(preprocessed_dir / "train_val_bow_corpus.mm")
    )

    print(f"Loaded train corpus: {len(train_bow_corpus)}")
    print(f"Loaded val corpus: {len(val_bow_corpus)}")
    print(f"Loaded test corpus: {len(test_bow_corpus)}")
    print(f"Loaded train_val corpus: {len(train_val_bow_corpus)}")
    print(f"Loaded train_dictionary with tokens: {len(train_dictionary)}")

    # Convert validation corpus to an in-memory list for per-type slicing
    val_bow_docs = list(val_bow_corpus)
    with open(preprocessed_dir / "val_combined_order_dict.json", "r") as f:
        val_order_dict = json.load(f)
    val_doc_items = sorted(
        val_order_dict.items(),
        key=lambda item: int(item[0].replace("doc", "")),
    )
    if len(val_doc_items) != len(val_bow_docs):
        raise ValueError(
            "Mismatch between validation corpus size and order dictionary entries."
        )

    val_ipo_bow = []
    val_analyst_bow = []
    for (doc_key, label), bow in zip(val_doc_items, val_bow_docs):
        label_lower = label.lower()
        if label_lower.startswith("ipo_"):
            val_ipo_bow.append(bow)
        elif label_lower.startswith("analyst_"):
            val_analyst_bow.append(bow)
        else:
            raise ValueError(
                f"Unknown label '{label}' in validation order dictionary at {doc_key}."
            )

    print(
        f"Validation corpus split -> IPO docs: {len(val_ipo_bow)}, "
        f"Analyst docs: {len(val_analyst_bow)}"
    )

    val_bow_corpus = val_bow_docs

    if args.optimize_topics and not args.num_topics:
        topic_model, metrics, optimal_num_topics = (
            lda_model_gensim.optimize_topic_number(
                train_corpus=train_bow_corpus,
                val_corpus=val_bow_corpus,
                id2word=train_dictionary,
                topic_range=config["lda"]["topic_range"],
                num_cores=config["lda"]["gensim"]["num_cores_lda"],
                model_params=config["lda"]["gensim"]["params"],
                random_seeds=config["lda"]["gensim"]["random_seeds"],
                save_models=config["lda"]["save_intermediate_models"],
                save_dir=str(model_save_dir),
                # val_corpora_by_type={
                #     "IPO": val_ipo_bow,
                #     "Analyst": val_analyst_bow,
                # },
            )
        )

        print(f"Optimized number of topics: {optimal_num_topics}")

        # Extract perplexity scores for all topic numbers (for plotting)
        val_perplexity_scores = [
            metrics[n]["val"]["average"] for n in sorted(metrics.keys())
        ]
        train_perplexity_scores = [
            metrics[n]["train"]["average"] for n in sorted(metrics.keys())
        ]

        # Save the topic model
        topic_model.save(
            str(model_save_dir / f"lda_model_{optimal_num_topics}_optimized")
        )

    elif args.num_topics and not args.optimize_topics:
        # Load the model if it exists
        if os.path.exists(
            str(model_save_dir / f"lda_model_{args.num_topics}_optimized")
        ):
            topic_model = gensim.models.LdaModel.load(
                str(model_save_dir / f"lda_model_{args.num_topics}_optimized")
            )
        else:
            topic_model = lda_model_gensim.model_training(
                topic_num=args.num_topics,
                train_corpus=train_val_bow_corpus,
                id2word=train_dictionary,
                model_params=config["lda"]["gensim"]["params"],
            )

    else:
        raise ValueError("Provide either a topic number or optimize_topics flag")

    if args.optimize_topics:
        # Plot perplexity scores
        utils.plot_perplexity_scores(
            topic_range=config["lda"]["topic_range"],
            perplexity_scores=train_perplexity_scores,
            output_dir=model_save_dir,
            mode="train",
        )
        utils.plot_perplexity_scores(
            topic_range=config["lda"]["topic_range"],
            perplexity_scores=val_perplexity_scores,
            output_dir=model_save_dir,
            mode="val",
        )

        # Save all the metrics
        with open(str(model_save_dir / "metrics.json"), "w") as f:
            json.dump(metrics, f)

    # plot the metrics


if __name__ == "__main__":
    main()
