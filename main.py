import argparse
import logging

from visualizing_wordcloud import visualize_wordcloud
import data_preprocessing
import lda_model_gensim
import utils

# Setup logging - modify to only show INFO level
logging.basicConfig(
    level=logging.INFO,
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
        "-n",
        "--num_docs",
        type=int,
        required=True,
        help="""How many documents to run the topic modeling on. If running for
        less documents, mention the exact number. If want to run for all
        documents, enter 0""",
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
    parser.add_argument(
        "-t",
        "--test_perc",
        type=float,
        required=False,
        default=0.1,
        help="Percentage of documents to use for testing. If not provided, the test percentage will be 20%.",
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
        train_documents_generator, test_documents_generator = (
            utils.load_files_in_batches(
                batch_size=batch_size,
                num_docs=args.num_docs,
                test_perc=args.test_perc,
            )
        )

        # documents = [document]

        # -------- Preprocess data --------
        logger.info("Starting preprocessing")
        train_dictionary, train_bow_corpus, train_tfidf_corpus, train_texts = (
            data_preprocessing.pre_processing_gensim(
                documents_generator=train_documents_generator,
                config=config["preprocessing"],
                num_cores=args.num_cores,
                mode="train",
            )
        )
        _, _, _, test_texts = data_preprocessing.pre_processing_gensim(
            documents_generator=test_documents_generator,
            config=config["preprocessing"],
            num_cores=args.num_cores,
            mode="test",
        )

        (
            test_bow_corpus,
            test_tfidf_corpus,
        ) = data_preprocessing.test_corpus_filtering(
            dic=train_dictionary,
            test_texts=test_texts,
        )

        # -------- Train model if optimize_topics is True --------
        if config["lda"]["optimize_topics"]:
            logger.info("Starting topic optimization")
            models, perplexity_scores = lda_model_gensim.optimize_topic_number(
                # train_corpus=train_tfidf_corpus,
                train_corpus=train_bow_corpus,
                id2word=train_dictionary,
                texts=train_texts,
                topic_range=config["lda"]["topic_range"],
                model_params=config["lda"]["gensim"],
                # test_corpus=test_tfidf_corpus,
                test_corpus=test_bow_corpus,
                num_cores=args.num_cores,
            )
            best_topic_num_idx = perplexity_scores.index(min(perplexity_scores))
            best_topic_num = config["lda"]["topic_range"]["start"] + (
                best_topic_num_idx * config["lda"]["topic_range"]["step"]
            )
            topic_model = models[best_topic_num_idx]
            logger.info("Best number of topics: %s", best_topic_num)
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
                mode="test",
            )

        elif args.num_topics:
            logger.info(
                "Training LDA model with gensim with %s topics",
                args.num_topics,
            )
            topic_model = lda_model_gensim.model_training(
                topic_num=args.num_topics,
                corpus=train_tfidf_corpus,
                # corpus=train_bow_corpus,
                id2word=train_dictionary,
                model_params=config["lda"]["gensim"],
            )
        else:
            raise ValueError("No topic number provided")

        # -------- Compute metrics --------
        perf_metrics = {}
        logger.info("Computing model performance metrics for train set")
        perplexity = lda_model_gensim.performance_metrics(
            model=topic_model,
            # corpus=train_tfidf_corpus,
            corpus=train_bow_corpus,
            texts=train_texts,
            id2word=train_dictionary,
        )
        perf_metrics["train"] = {
            "perplexity": perplexity,
        }

        logger.info("Computing model performance metrics for test set")
        perplexity = lda_model_gensim.performance_metrics(
            model=topic_model,
            # corpus=test_tfidf_corpus,
            corpus=test_bow_corpus,
            texts=test_texts,
            id2word=train_dictionary,
        )
        perf_metrics["test"] = {
            "perplexity": perplexity,
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
