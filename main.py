import argparse
import logging
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

from visualizing_wordcloud import visualize_wordcloud
import data_preprocessing
import lda_model_gensim
import utils

# Setup logging - modify to only show INFO level
logging.basicConfig(
    level=logging.INFO,
    # format="%(message)s",  # Simplified format
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logging.getLogger("gensim").setLevel(logging.ERROR)  # For gensim
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run Topic Modeling Pipeline")
    parser.add_argument(
        "-c",
        "--config",
        default="config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "-d",
        "--data",
        required=False,
        default="data_url_csv.txt",
        help="Path to input text document",
    )
    parser.add_argument(
        "-n",
        "--num_docs",
        type=int,
        required=True,
        help="""How many documents to run the topic modeling on. If running for
        less documents, mention the exact number. If want to run for all
        documents, enter 0""",
    )
    # parser.add_argument(
    #     "-k",
    #     "--num_topics",
    #     type=int,
    #     required=False,
    #     help="Number of topics to run the topic modeling on. If not provided, the number of topics will be determined automatically.",
    # )
    parser.add_argument(
        "-nc",
        "--num_cores",
        type=int,
        required=False,
        default=16,
        help="Minimum number of cores to run the topic modeling on. If not provided, the number of cores will be determined automatically.",
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        type=int,
        required=False,
        default=100,
        help="Number of documents to process in each batch. If not provided, the batch size will be 15.",
    )
    args = parser.parse_args()

    # Load configuration
    config = utils.load_config(args.config)

    try:
        # Initialize variables
        dictionary = None
        bow_corpus = None
        tfidf_corpus = None
        topic_model = None
        batch_size = args.batch_size

        # Setup output directory
        output_dir = utils.setup_output_directory(config)
        logger.info("Results will be saved to %s", output_dir)

        # Load data
        # documents = utils.load_data(args.data, args.num_docs)
        # documents = utils.load_files(num_files=args.num_docs)
        documents_generator = utils.load_files_in_batches(
            batch_size=batch_size,
            num_docs=args.num_docs,
        )

        # documents = [document]

        # -------- Preprocess data --------
        logger.info("Starting preprocessing")
        dictionary, bow_corpus, tfidf_corpus, texts = (
            data_preprocessing.pre_processing_gensim(
                documents_generator=documents_generator,
                config=config["preprocessing"],
                num_cores=args.num_cores,
            )
        )

        # -------- Train model if optimize_topics is True --------
        if config["lda"]["optimize_topics"]:
            logger.info("Starting topic optimization")
            models, perplexity_scores = lda_model_gensim.optimize_topic_number(
                tfidf_corpus,
                dictionary,
                bow_corpus,
                config["lda"]["topic_range"],
                config["lda"]["gensim"],
            )
            best_idx = perplexity_scores.index(min(perplexity_scores))
            best_num_topics = config["lda"]["topic_range"]["start"] + (
                best_idx * config["lda"]["topic_range"]["step"]
            )
            topic_model = models[best_idx]
            logger.info("Best number of topics: %s", best_num_topics)
            # Save topic numbers and perplexity scores
            utils.save_topic_perplexity_scores(
                config["lda"]["topic_range"],
                perplexity_scores,
                output_dir,
            )

            # Create and save the perplexity plot
            utils.plot_perplexity_scores(
                config["lda"]["topic_range"],
                perplexity_scores,
                output_dir,
            )

        else:
            logger.info(
                f"Training LDA model with gensim with {config['lda']['gensim']['num_topics']} topics"
            )
            topic_model = lda_model_gensim.model_training(
                config["lda"]["gensim"]["num_topics"],
                # tfidf_corpus,
                bow_corpus,
                dictionary,
                config["lda"]["gensim"],
            )

        # -------- Compute metrics --------
        perf_metrics = {}
        logger.info("Computing model performance metrics using gensim")
        perplexity, coherence_metrics = lda_model_gensim.performance_metrics(
            topic_model,
            # tfidf_corpus,
            bow_corpus,
            texts,
            dictionary,
        )
        perf_metrics = {
            "perplexity": perplexity,
            "coherence_metrics": coherence_metrics,
        }

        # Save results
        logger.info("Saving results")
        utils.save_model_results(
            output_dir,
            topic_model,
            perf_metrics,
            config,
        )

        # Generate visualizations
        if config["output"]["save_visualizations"]:
            logger.info("Generating visualizations")

            visualize_wordcloud(
                topic_model,
                output_dir / "wordcloud.png",
                config["visualization"]["wordcloud"],
            )

        # Analyze word frequencies
        logger.info("Analyzing word frequencies")
        utils.analyze_word_frequencies(output_dir / "topics.txt", output_dir)

        logger.info("Pipeline completed successfully")

    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
