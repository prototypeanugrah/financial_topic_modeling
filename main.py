import argparse
import logging

import cross_validation
import lda_model_gensim
import utils
from data_preprocessing import lda_preprocessing as data_preprocessing
from visualizing_wordcloud import visualize_wordcloud

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
        "-i",
        "--input_dir",
        default="data/raw_reports_v3",
        help="Path to input directory",
    )
    parser.add_argument(
        "-n",
        "--num_docs",
        type=int,
        required=True,
        help="How many documents to run the topic modeling on. If running for less documents, mention the exact number. If want to run for all documents, enter 0",
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
        help="Percentage of documents to use for testing. If not provided, the test percentage will be 20 percent.",
    )

    parser.add_argument(
        "--compute_coherence",
        action="store_true",
        help="Whether to compute coherence metrics. If not provided, the coherence metrics will not be computed. If provided, the coherence metrics will be computed.",
    )

    parser.add_argument(
        "--optimize_topics",
        action="store_true",
        help="Whether to optimize the number of topics. If not provided, the number of topics will not be optimized.",
    )

    parser.add_argument(
        "--cross_validate",
        action="store_true",
        help="Perform k-fold cross-validation instead of train/test split evaluation.",
    )

    parser.add_argument(
        "--cv_folds",
        type=int,
        default=5,
        help="Number of folds for cross-validation (default: 5).",
    )

    args = parser.parse_args()

    # Load configuration
    config = utils.load_config(args.config)

    try:
        # Setup output directory
        output_dir = utils.setup_output_directory(config)
        logger.info("Results will be saved to %s", output_dir)

        # Check if cross-validation mode
        if args.cross_validate:
            if not args.num_topics:
                raise ValueError(
                    "Cross-validation requires specifying number of topics with -k/--num_topics"
                )

            logger.info("Running in cross-validation mode")

            # Load all documents for CV
            all_documents = utils.load_all_documents(
                num_docs=args.num_docs,
                input_dir=args.input_dir,
            )

            # Run cross-validation
            cv_results = cross_validation.cross_validate_lda(
                documents=all_documents,
                config=config,
                num_topics=args.num_topics,
                num_cores=args.num_cores,
                n_splits=args.cv_folds,
                compute_coherence=args.compute_coherence,
            )

            # Print and save results
            cross_validation.print_cv_summary(cv_results)
            utils.save_cv_results(cv_results, output_dir)

            logger.info("Cross-validation completed successfully")
            return

        # Load data for standard train/test split
        train_documents_generator, test_documents_generator = (
            utils.load_files_in_batches(
                batch_size=args.batch_size,
                num_docs=args.num_docs,
                test_perc=args.test_perc,
                input_dir=args.input_dir,
            )
        )

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
        test_texts = data_preprocessing.pre_processing_gensim(
            documents_generator=test_documents_generator,
            config=config["preprocessing"],
            num_cores=args.num_cores,
            mode="test",
        )[3]

        (
            test_bow_corpus,
            test_tfidf_corpus,
        ) = data_preprocessing.test_corpus_filtering(
            dic=train_dictionary,
            test_texts=test_texts,
        )

        # -------- Train model if optimize_topics is True --------
        if args.optimize_topics and not args.num_topics:
            logger.info("Starting topic optimization")
            # Add configuration for optimization
            save_models_to_disk = config.get("lda", {}).get(
                "save_intermediate_models", True
            )
            model_save_dir = (
                output_dir / "intermediate_models" if save_models_to_disk else None
            )

            # Run optimization
            topic_model, all_metrics, best_topic_num = (
                lda_model_gensim.optimize_topic_number(
                    train_corpus=train_bow_corpus,
                    test_corpus=test_bow_corpus,
                    id2word=train_dictionary,
                    texts=train_texts,
                    topic_range=config["lda"]["topic_range"],
                    model_params=config["lda"]["gensim"],
                    num_cores=args.num_cores,
                    save_models=save_models_to_disk,
                    save_dir=str(model_save_dir) if model_save_dir else None,
                    compute_coherence=args.compute_coherence,  # Always compute for reporting
                )
            )

            # Extract perplexity scores for plotting
            perplexity_scores = [
                metrics.get("perplexity", float("inf"))
                for num_topics, metrics in sorted(all_metrics.items())
            ]

            logger.info("Best number of topics: %s", best_topic_num)
            # Save comprehensive metrics
            utils.save_optimization_metrics(
                config["lda"]["topic_range"], all_metrics, output_dir
            )

            # Save topic numbers and perplexity scores (for backward compatibility)
            utils.save_topic_perplexity_scores(
                config["lda"]["topic_range"],
                perplexity_scores,
                output_dir,
            )

            # Create enhanced visualization
            utils.plot_metrics_comparison(
                config["lda"]["topic_range"], all_metrics, output_dir
            )

            # Create and save the perplexity plot (for backward compatibility)
            utils.plot_perplexity_scores(
                config["lda"]["topic_range"],
                perplexity_scores,
                output_dir,
                mode="test",
            )

        elif args.num_topics and not args.optimize_topics:
            logger.info(
                "Training LDA model with gensim with %s topics",
                args.num_topics,
            )
            topic_model = lda_model_gensim.model_training(
                topic_num=args.num_topics,
                train_corpus=train_bow_corpus,
                test_corpus=test_bow_corpus,
                id2word=train_dictionary,
                model_params=config["lda"]["gensim"],
            )

        else:
            raise ValueError("No topic number provided")

        # -------- Compute metrics --------
        perf_metrics = {}
        logger.info("Computing model performance metrics for train set")
        train_metrics = lda_model_gensim.performance_metrics(
            model=topic_model,
            corpus=train_bow_corpus,
            texts=train_texts,
            id2word=train_dictionary,
            compute_coherence=args.compute_coherence,
        )
        perf_metrics["train"] = train_metrics

        logger.info("Computing model performance metrics for test set")
        test_metrics = lda_model_gensim.performance_metrics(
            model=topic_model,
            corpus=test_bow_corpus,
            texts=train_texts,  # Use train texts for coherence to avoid vocabulary mismatch
            id2word=train_dictionary,
            compute_coherence=args.compute_coherence,
        )
        perf_metrics["test"] = test_metrics

        # Save results
        logger.info("Saving results")
        utils.save_model_results(
            output_dir=output_dir,
            lda_model=topic_model,
            corpus=train_bow_corpus,
            perf_metrics=perf_metrics,
            config=config,
        )

        # Plot perplexity scores
        if args.optimize_topics:
            utils.plot_perplexity_scores(
                topic_range=config["lda"]["topic_range"],
                perplexity_scores=perplexity_scores,
                output_dir=output_dir,
                mode="test",
            )

        # Generate visualizations
        if config["output"]["save_visualizations"]:
            logger.info("Generating visualizations")

            visualize_wordcloud(
                lda_model=topic_model,
                output_path=output_dir / "wordcloud.png",
                config=config["visualization"]["wordcloud"],
            )

        # Analyze word frequencies
        logger.info("Analyzing word frequencies")
        utils.analyze_word_frequencies(
            file_path=output_dir / "topics.txt",
            output_dir=output_dir,
        )

        # Save document topic distribution
        logger.info("Saving document topic distribution")
        lda_model_gensim.document_topic_distribution(
            model=topic_model,
            train_corpus=train_bow_corpus,
            test_corpus=test_bow_corpus,
            output_dir=output_dir,
        )

        logger.info("Pipeline completed successfully")

    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
