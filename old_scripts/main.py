import argparse
import logging

import numpy as np
import pandas as pd

import lda_model_gensim
import utils
from data_preprocessing import lda_preprocessing as data_preprocessing

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
        "--csv_file",
        default="data/final_analyst_reports_for_latest_s1_filings_extracted.csv",
        help="Path to CSV file containing report paths",
    )
    parser.add_argument(
        "--report_type",
        choices=["analyst", "ipo", "both"],
        required=True,
        help="Type of reports to process: 'analyst' for analyst reports, 'ipo' for S1 filings",
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

    return parser.parse_args()


def process_report_type(
    report_type: str,
    file_paths: list,
    args: argparse.Namespace,
    config: dict,
    output_dir: str,
    prefix: str,
    shared_dictionary,
):
    """
    Process a specific report type (ipo or analyst) through the full pipeline.

    Args:
        report_type: Type of report ('ipo' or 'analyst')
        file_paths: List of file paths for this report type
        args: Command line arguments
        config: Configuration dictionary
        output_dir: Output directory path
        prefix: Prefix for output files
        shared_dictionary: Optional shared dictionary for both report types (default: None)
    """
    logger.info(
        f"Starting processing for {report_type} reports with {len(file_paths)} files"
    )

    # Load data for standard train/test split
    train_documents_generator, test_documents_generator = (
        utils.load_files_in_batches_from_paths(
            file_paths=file_paths,
            batch_size=args.batch_size,
            num_docs=args.num_docs,
            test_perc=args.test_perc,
        )
    )

    # -------- Preprocess data --------

    if shared_dictionary is not None:
        # Use shared dictionary for preprocessing
        train_bow_corpus, train_texts = (
            data_preprocessing.preprocess_documents_with_dictionary(
                documents_generator=train_documents_generator,
                shared_dictionary=shared_dictionary,
                config=config["preprocessing"],
                num_cores=args.num_cores,
                mode="train",
            )
        )

        test_bow_corpus, test_texts = (
            data_preprocessing.preprocess_documents_with_dictionary(
                documents_generator=test_documents_generator,
                shared_dictionary=shared_dictionary,
                config=config["preprocessing"],
                num_cores=args.num_cores,
                mode="test",
            )
        )

        train_dictionary = shared_dictionary

    else:
        # Original workflow: create separate dictionary for each report type
        train_dictionary, train_bow_corpus, train_texts = (
            data_preprocessing.preprocess_documents_build_dictionary(
                documents_generator=train_documents_generator,
                config=config["preprocessing"],
                num_cores=args.num_cores,
                mode="train",
            )
        )

        test_bow_corpus, test_texts = (
            data_preprocessing.preprocess_documents_with_dictionary(
                documents_generator=test_documents_generator,
                shared_dictionary=train_dictionary,
                config=config["preprocessing"],
                num_cores=args.num_cores,
                mode="test",
            )
        )

    # Initialize variables for perplexity scores
    test_perplexity_scores = []
    train_perplexity_scores = []

    # -------- Train model if optimize_topics is True --------
    if args.optimize_topics and not args.num_topics:
        logger.info(f"Starting topic optimization for {report_type} reports")
        # Add configuration for optimization
        save_models_to_disk = config.get("lda", {}).get(
            "save_intermediate_models", True
        )
        model_save_dir = (
            output_dir / f"{prefix}_intermediate_models"
            if save_models_to_disk
            else None
        )

        # Run optimization
        topic_model, all_metrics, best_topic_num = (
            lda_model_gensim.optimize_topic_number(
                train_corpus=train_bow_corpus,
                test_corpus=test_bow_corpus,
                id2word=train_dictionary,
                train_texts=train_texts,
                test_texts=test_texts,
                topic_range=config["lda"]["topic_range"],
                model_params=config["lda"]["gensim"][f"{prefix}_params"],
                num_cores=1,
                save_models=save_models_to_disk,
                save_dir=str(model_save_dir) if model_save_dir else None,
                compute_coherence=args.compute_coherence,  # Always compute for reporting
            )
        )

        # Extract perplexity scores for plotting (use test perplexity)
        test_perplexity_scores = [
            metrics.get("test", {}).get("perplexity", np.inf)
            for _, metrics in sorted(all_metrics.items())
        ]

        # Extract train perplexity scores for plotting (currently unused but available for future use)
        train_perplexity_scores = [
            metrics.get("train", {}).get("perplexity", np.inf)
            for _, metrics in sorted(all_metrics.items())
        ]

        logger.info(f"Best number of topics for {report_type}: %s", best_topic_num)
        # Save comprehensive metrics
        # utils.save_optimization_metrics(
        #     config["lda"]["topic_range"],
        #     all_metrics,
        #     output_dir,
        #     prefix=prefix,
        # )

    elif args.num_topics and not args.optimize_topics:
        logger.info(
            f"Training LDA model for {report_type} reports with %s topics",
            args.num_topics,
        )
        topic_model = lda_model_gensim.model_training(
            topic_num=args.num_topics,
            train_corpus=train_bow_corpus,
            test_corpus=test_bow_corpus,
            id2word=train_dictionary,
            model_params=config["lda"]["gensim"][f"{prefix}_params"],
        )

    else:
        raise ValueError("Provide either a topic number or optimize_topics flag")

    # -------- Compute metrics --------
    logger.info(f"Computing model performance metrics for {report_type} train set")
    train_perplexity = lda_model_gensim.performance_metrics(
        model=topic_model,
        corpus=train_bow_corpus,
    )
    test_perplexity = lda_model_gensim.performance_metrics(
        model=topic_model,
        corpus=test_bow_corpus,
    )

    # Save results
    logger.info(f"Saving results for {report_type} reports")

    # Save train results
    utils.save_model_results(
        output_dir=output_dir,
        lda_model=topic_model,
        corpus=train_bow_corpus,
        perplexity=train_perplexity,
        prefix=prefix,
    )
    # Save test results
    utils.save_model_results(
        output_dir=output_dir,
        lda_model=topic_model,
        corpus=test_bow_corpus,
        perplexity=test_perplexity,
        prefix=prefix,
    )

    # Plot perplexity scores
    if args.optimize_topics:
        utils.plot_perplexity_scores(
            topic_range=config["lda"]["topic_range"],
            perplexity_scores=train_perplexity_scores,
            output_dir=output_dir,
            prefix=prefix,
            mode="train",
        )
        utils.plot_perplexity_scores(
            topic_range=config["lda"]["topic_range"],
            perplexity_scores=test_perplexity_scores,
            output_dir=output_dir,
            prefix=prefix,
            mode="test",
        )

        # Save topic numbers and perplexity scores (for backward compatibility)
        utils.save_topic_perplexity_scores(
            topic_range=config["lda"]["topic_range"],
            perplexity_scores=train_perplexity_scores,
            output_dir=output_dir,
            prefix=prefix,
            mode="train",
        )
        utils.save_topic_perplexity_scores(
            topic_range=config["lda"]["topic_range"],
            perplexity_scores=test_perplexity_scores,
            output_dir=output_dir,
            prefix=prefix,
            mode="test",
        )

    # Analyze word frequencies
    logger.info(f"Analyzing word frequencies for {report_type} reports")
    utils.analyze_word_frequencies(
        file_path=output_dir / f"{prefix}_topics.txt",
        output_dir=output_dir,
        prefix=prefix,
    )

    # Save document topic distribution
    logger.info(f"Saving document topic distribution for {report_type} reports")
    lda_model_gensim.document_topic_distribution(
        model=topic_model,
        corpus=train_bow_corpus,
        output_dir=output_dir,
        prefix=prefix,
        mode="train",
    )
    lda_model_gensim.document_topic_distribution(
        model=topic_model,
        corpus=test_bow_corpus,
        output_dir=output_dir,
        prefix=prefix,
        mode="test",
    )

    logger.info(f"Processing for {report_type} reports completed successfully")


def main():
    # Parse arguments
    args = arguments_parser()

    # Load configuration
    config = utils.load_config(args.config)

    # Load CSV and extract file paths for both report types
    df = pd.read_csv(args.csv_file)

    # Extract file paths for both report types
    ipo_file_paths = df["s1_path"].dropna().tolist()
    analyst_file_paths = df["analyst_report_path"].dropna().tolist()

    try:
        # Setup output directory
        output_dir = utils.setup_output_directory(config)

        if args.report_type == "both":
            # Create shared dictionary from both report types
            logger.info("=" * 60)
            logger.info("CREATING SHARED DICTIONARY FROM BOTH REPORT TYPES")
            logger.info("=" * 60)

            # Load all documents for shared dictionary creation
            ipo_all_docs_generator, _ = utils.load_files_in_batches_from_paths(
                file_paths=ipo_file_paths,
                batch_size=args.batch_size,
                num_docs=args.num_docs,
                test_perc=0.0,  # Load all as training for dictionary
            )

            analyst_all_docs_generator, _ = utils.load_files_in_batches_from_paths(
                file_paths=analyst_file_paths,
                batch_size=args.batch_size,
                num_docs=args.num_docs,
                test_perc=0.0,  # Load all as training for dictionary
            )

            # Create shared dictionary from both document types
            shared_dictionary = data_preprocessing.build_dictionary_from_generators(
                documents_generators=[
                    ipo_all_docs_generator,
                    analyst_all_docs_generator,
                ],
                num_cores=args.num_cores,
                config=config["preprocessing"],
                source_labels=["ipo", "analyst"],
            )

            # logger.info(
            #     f"Shared dictionary created with {len(shared_dictionary)} unique tokens"
            # )

            # Save shared dictionary
            shared_dict_path = output_dir / "shared_dictionary.id2word"
            shared_dictionary.save(str(shared_dict_path))

            # Process IPO reports first with shared dictionary
            logger.info("=" * 60)
            logger.info("STARTING IPO REPORTS PROCESSING WITH SHARED DICTIONARY")
            logger.info("=" * 60)
            process_report_type(
                report_type="ipo",
                file_paths=ipo_file_paths,
                args=args,
                config=config,
                output_dir=output_dir,
                prefix="ipo",
                shared_dictionary=shared_dictionary,
            )

            # Process analyst reports second with shared dictionary
            logger.info("=" * 60)
            logger.info("STARTING ANALYST REPORTS PROCESSING WITH SHARED DICTIONARY")
            logger.info("=" * 60)
            process_report_type(
                report_type="analyst",
                file_paths=analyst_file_paths,
                args=args,
                config=config,
                output_dir=output_dir,
                prefix="analyst",
                shared_dictionary=shared_dictionary,
            )

        elif args.report_type == "ipo":
            logger.info("=" * 60)
            logger.info("STARTING IPO REPORTS PROCESSING (SINGLE REPORT TYPE MODE)")
            logger.info("=" * 60)
            process_report_type(
                report_type="ipo",
                file_paths=ipo_file_paths,
                args=args,
                config=config,
                output_dir=output_dir,
                prefix="ipo",
                shared_dictionary=None,
            )

        elif args.report_type == "analyst":
            logger.info("=" * 60)
            logger.info("STARTING ANALYST REPORTS PROCESSING (SINGLE REPORT TYPE MODE)")
            logger.info("=" * 60)
            process_report_type(
                report_type="analyst",
                file_paths=analyst_file_paths,
                args=args,
                config=config,
                output_dir=output_dir,
                prefix="analyst",
                shared_dictionary=None,
            )

        else:
            raise ValueError("Invalid report type")

        logger.info("=" * 60)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
