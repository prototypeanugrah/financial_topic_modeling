import argparse
import logging


import lda_model_gensim
import lda_model_sklearn
from visualizing_wordcloud import visualize_wordcloud
import data_preprocessing
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
    parser = argparse.ArgumentParser(description="Run LDA Topic Modeling Pipeline")
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--data",
        required=True,
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
    parser.add_argument(
        "-m",
        "--model_type",
        required=True,
        help="Whether ot select gensim or sklearn for lda topic modeling",
    )
    args = parser.parse_args()

    # Load configuration
    config = utils.load_config(args.config)

    try:
        # Initialize variables
        tfidf_vectorizer = None
        tfidf_matrix = None
        id2word = None
        texts = None
        tfidf_corpus = None
        lda_model = None

        # Setup output directory
        output_dir = utils.setup_output_directory(config)
        logger.info(f"Results will be saved to {output_dir}")

        # Load data
        documents = utils.load_data(args.data, args.num_docs)

        # documents = [document]

        # Preprocess data
        logger.info("Starting preprocessing")
        if args.model_type == "gensim":
            id2word, texts, _, tfidf_corpus = data_preprocessing.pre_processing_gensim(
                documents,
                config=config["preprocessing"],
            )
        elif args.model_type == "sklearn":
            _, tfidf_vectorizer, tfidf_matrix = (
                data_preprocessing.pre_processing_sklearn(
                    documents,
                    config=config["preprocessing"],
                )
            )
        else:
            tfidf_vectorizer = None
            tfidf_matrix = None
            id2word = None
            texts = None
            tfidf_corpus = None

        # Train model
        if config["lda"]["optimize_topics"]:
            if args.model_type == "gensim":
                logger.info("Starting topic optimization")
                models, perplexity_scores = lda_model_gensim.optimize_topic_number(
                    tfidf_corpus,
                    id2word,
                    texts,
                    config["lda"]["topic_range"],
                    config["lda"]["gensim"],
                )
                best_idx = perplexity_scores.index(min(perplexity_scores))
                best_num_topics = config["lda"]["topic_range"]["start"] + (
                    best_idx * config["lda"]["topic_range"]["step"]
                )
                lda_model = models[best_idx]
                logger.info(f"Best number of topics: {best_num_topics}")
                # Save topic numbers and perplexity scores
                utils.save_topic_perplexity_scores(
                    config["lda"]["topic_range"],
                    perplexity_scores,
                    output_dir,
                )

            elif args.model_type == "sklearn":
                models, perplexity_scores = lda_model_sklearn.optimize_topic_number(
                    tfidf_matrix,
                    config["lda"]["topic_range"],
                    config["lda"]["sklearn"],
                )
                best_idx = perplexity_scores.index(min(perplexity_scores))
                best_num_topics = config["lda"]["topic_range"]["start"] + (
                    best_idx * config["lda"]["topic_range"]["step"]
                )
                lda_model = models[best_idx]
                logger.info(f"Best number of topics: {best_num_topics}")
                # Save topic numbers and perplexity scores
                utils.save_topic_perplexity_scores(
                    config["lda"]["topic_range"],
                    perplexity_scores,
                    output_dir,
                )
            else:
                lda_model = None

            # Create and save the perplexity plot
            utils.plot_perplexity_scores(
                config["lda"]["topic_range"],
                perplexity_scores,
                output_dir,
            )

        else:
            if args.model_type == "gensim":
                logger.info("Training LDA model with gensim")
                lda_model = lda_model_gensim.model_training(
                    config["lda"]["gensim"]["num_topics"],
                    tfidf_corpus,
                    id2word,
                    config["lda"]["gensim"],
                )
            elif args.model_type == "sklearn":
                logger.info("Training LDA model with sklearn")
                lda_model = lda_model_sklearn.model_training(
                    config["lda"]["sklearn"]["n_components"],
                    tfidf_matrix,
                    config["lda"]["sklearn"],
                )
            else:
                lda_model = None

        # Compute metrics
        perf_metrics = {}
        if args.model_type == "gensim":
            logger.info("Computing model performance metrics using gensim")
            perplexity, coherence_metrics = lda_model_gensim.performance_metrics(
                lda_model,
                tfidf_corpus,
                texts,
                id2word,
            )
            perf_metrics = {
                "perplexity": perplexity,
                "coherence_metrics": coherence_metrics,
            }
        elif args.model_type == "sklearn":
            logger.info("Computing model performance metrics using sklearn")
            perplexity, metrics = lda_model_sklearn.performance_metrics_sklearn(
                lda_model,
                tfidf_matrix,
            )
            perf_metrics = metrics
        else:
            perf_metrics = {}

        # Save results
        logger.info("Saving results")
        utils.save_model_results(
            output_dir,
            lda_model,
            perf_metrics,
            config,
            model_type=args.model_type,
            vectorizer=tfidf_vectorizer if args.model_type == "sklearn" else None,
            doc_term_matrix=tfidf_matrix if args.model_type == "sklearn" else None,
            corpus=tfidf_corpus if args.model_type == "gensim" else None,
        )

        # Generate visualizations
        if config["output"]["save_visualizations"]:
            logger.info("Generating visualizations")
            if args.model_type == "sklearn":
                # Add feature names to visualization config for sklearn
                config["visualization"]["wordcloud"][
                    "feature_names"
                ] = tfidf_vectorizer.get_feature_names_out()
            visualize_wordcloud(
                lda_model,
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
