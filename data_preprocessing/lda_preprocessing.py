"""
This script is used to preprocess the text data for the LDA topic modeling.
"""

import json
import logging
import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple, Union, cast

import numpy as np
import pandas as pd
import regex as re
from gensim import corpora, models
from gensim.utils import simple_preprocess
from nltk.corpus import stopwords
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

import utils

logger = logging.getLogger(__name__)

# Global worker variables for efficient spaCy initialization
worker_nlp = None
worker_allowed_postags = None


# Preprocess the text data
def basic_preprocessing(text: str) -> str:
    """
    Basic preprocessing of text data to remove unwanted characters and metadata.
    This includes:
    - Removing everything before <SEC-DOCUMENT> tag
    - Removing SEC header metadata block
    - Keeping only the S-1 and S-1/A document section
    - Removing all HTML tags (content between < and >)
    - Removing extra spaces, emails, apostrophes, and non-alphabet characters
    - Converting text to lowercase

    Args:
        text (str): Input text to preprocess

    Returns:
        str: Preprocessed text
    """

    # # Remove everything before <SEC-DOCUMENT> tag
    # text = re.sub(
    #     r"^.*?<SEC-DOCUMENT>", "<SEC-DOCUMENT>", text, flags=re.DOTALL | re.MULTILINE
    # )

    # # Remove SEC header starting from <SEC-HEADER> metadata block until </SEC-HEADER>
    # text = re.sub(
    #     r"<SEC-HEADER>.*?</SEC-HEADER>", "", text, flags=re.DOTALL | re.MULTILINE
    # )

    # # First find and keep only the S-1 and S-1/A document section
    # s1_pattern = r"<DOCUMENT>(.*?<TYPE>(?:S-1|S-1/A).*?)</DOCUMENT>"
    # s1_matches = re.findall(s1_pattern, text, flags=re.DOTALL | re.MULTILINE)

    # if not s1_matches:
    #     logger.warning("No S-1 document found")
    #     return ""

    # # Take the first S-1 document if multiple exist
    # text = s1_matches[0]

    # # # Remove all content between <TABLE> and </TABLE> tags
    # # text = re.sub(r"<TABLE>.*?</TABLE>", "", text, flags=re.DOTALL | re.MULTILINE)

    # # # Remove document type markers and page markers
    # # text = re.sub(
    # #     r"<DOCUMENT>|<TYPE>.*?</TYPE>|<SEQUENCE>.*?</SEQUENCE>|"
    # #     r"<DESCRIPTION>.*?</DESCRIPTION>|<TEXT>|<PAGE>",
    # #     " ",
    # #     text,
    # #     flags=re.DOTALL | re.MULTILINE,
    # # )

    # # Remove all HTML tags (content between < and >)
    # text = re.sub(r"<[^>]*>", "", text, flags=re.DOTALL | re.MULTILINE)

    # Standard cleaning steps
    text = re.sub(r"\S*@\S*\s?", "", text)  # Remove emails
    text = re.sub(r"'", "", text)  # Remove apostrophes
    text = re.sub(r"&nbsp;", " ", text)  # Remove &nbsp;

    text = re.sub(r"[^a-zA-Z]", " ", text)  # Remove non-alphabet characters

    text = text.lower()  # Convert to lowercase
    text = re.sub(r"\s+", " ", text)  # Remove extra spaces

    return text


def load_stopwords_from_file(filepath: str) -> List[str]:
    """
    Load stopwords from a text file.

    Args:
        filepath: Path to the stopwords file

    Returns:
        List of stopwords

    Raises:
        FileNotFoundError: If the stopwords file doesn't exist
    """
    try:
        with open(filepath, "r", encoding="utf-8") as file:
            # Read words, strip whitespace, and convert to lowercase
            return [word.strip().lower() for word in file.readlines() if word.strip()]
    except FileNotFoundError:
        logger.warning("Stopwords file not found: %s", filepath)
        return []


def sent_to_words(sentences: List[str]) -> Iterator[List[str]]:
    """
    Tokenize sentences into words using gensim's simple_preprocess.

    Args:
        sentences (List[str]): List of sentences to tokenize

    Yields:
        Iterator[List[str]]: Iterator over tokenized sentences
    """
    for sentence in sentences:
        yield (
            simple_preprocess(str(sentence), deacc=True)
        )  # deacc=True removes punctuations


def remove_stopwords(
    texts: List[List[str]],
    stop_words: List[str],
) -> List[List[str]]:
    """
    Remove stopwords from tokenized texts.

    Args:
        texts (List[List[str]]): List of tokenized texts (each text is a list of words)
        stop_words (List[str]): List of stopwords to remove

    Returns:
        List[List[str]]: List of texts with stopwords removed (each text is a list of words)
    """
    return [[word for word in doc if word not in stop_words] for doc in texts]


def lemmatization(
    tokens: List[List[str]],
    nlp,
    allowed_postags: List[str],
) -> List[List[str]]:
    """
    Perform lemmatization on tokenized texts using spaCy efficiently.

    This function creates spaCy Doc objects directly from tokens without
    retokenization, which is 3-5x faster than the previous approach.

    Args:
        tokens (List[List[str]]): List of tokenized texts (each text is a list of words)
        nlp: spaCy language model for lemmatization
        allowed_postags (List[str]): List of allowed part-of-speech tags for lemmatization

    Returns:
        List[List[str]]: List of lemmatized texts (each text is a list of words)

    Note:
    - The `allowed_postags` parameter specifies which part-of-speech tags
        to keep during lemmatization. By default, it includes nouns, adjectives,
        verbs, and adverbs.
    - This approach maintains token boundaries and avoids costly retokenization
    """
    from spacy.tokens import Doc

    tokens_out = []
    for token_list in tokens:
        if not token_list:  # Handle empty token lists
            tokens_out.append([])
            continue

        try:
            # Create Doc directly from tokens - no retokenization needed
            doc = Doc(nlp.vocab, words=token_list)

            # Run only necessary pipeline components for lemmatization
            for pipe_name in ["tagger", "attribute_ruler", "lemmatizer"]:
                if pipe_name in nlp.pipe_names:
                    pipe = nlp.get_pipe(pipe_name)
                    doc = pipe(doc)

            # Extract lemmas with POS filtering
            lemmas = [token.lemma_ for token in doc if token.pos_ in allowed_postags]
            tokens_out.append(lemmas)

        except Exception as e:
            logger.warning("Lemmatization failed for document: %s", str(e))
            # Fall back to original tokens if lemmatization fails
            tokens_out.append(token_list)

    return tokens_out


def remove_words_less_than_length_three_characters(
    text: List[List[str]],
) -> List[List[str]]:
    """
    Remove words which have less than 3 characters

    Args:
        text (List[List[str]]): List of tokenized texts (each text is a list
        of words)

    Returns:
        List[List[str]]: List of texts with words of length less than 3 removed
    """
    return [[word for word in document if len(word) > 3] for document in text]


def timer_decorator(func):
    """
    Decorator to measure the execution time of a function.
    This decorator logs the time taken for the function to execute.

    Args:
        func (callable): The function to be decorated

    Returns:
        callable: The wrapped function that logs its execution time
    """

    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        logger.info(
            "%s took %.2f seconds to execute",
            func.__name__,
            execution_time,
        )
        return result

    return wrapper


def init_worker(
    spacy_model: str, spacy_disabled: List[str], allowed_postags: List[str]
):
    """Initialize spaCy model once per worker process"""
    global worker_nlp, worker_allowed_postags
    import spacy

    worker_nlp = spacy.load(spacy_model, disable=spacy_disabled)
    worker_allowed_postags = allowed_postags


def process_document_chunk(
    pdc_args: Tuple[str, set],
) -> List[str]:
    """
    Process a single document through all preprocessing steps.

    Args:
        pdc_args: Tuple containing (document, stop_words)

    Returns:
        List[str]: Processed tokens from the document

    Note:
    1. Basic preprocessing
    2. Tokenization
    3. Remove stopwords
    4. Remove words with less than 3 characters
    5. Lemmatization (using worker's spaCy model)
    """
    global worker_nlp, worker_allowed_postags
    document, stop_words = pdc_args

    # Apply preprocessing steps sequentially
    text = basic_preprocessing(document)
    tokens = list(sent_to_words([text]))[0]
    tokens = remove_stopwords([tokens], list(stop_words))[0]

    # Remove tokens that contain numbers
    tokens = [token for token in tokens if not any(char.isdigit() for char in token)]

    tokens = remove_words_less_than_length_three_characters([tokens])[0]

    # Lemmatization using worker's spaCy model
    if worker_nlp is not None and worker_allowed_postags is not None:
        try:
            from spacy.tokens import Doc

            doc = Doc(worker_nlp.vocab, words=tokens)
            for pipe_name in ["tagger", "attribute_ruler", "lemmatizer"]:
                if pipe_name in worker_nlp.pipe_names:
                    pipe = worker_nlp.get_pipe(pipe_name)
                    doc = pipe(doc)
            tokens = [
                token.lemma_ for token in doc if token.pos_ in worker_allowed_postags
            ]

        except Exception as e:
            logger.warning("Worker lemmatization failed: %s", str(e))
            # Fall back to original tokens

    return tokens


def make_bigrams(
    texts: List[List[str]],
    mode: models.Phrases,
) -> List[List[str]]:
    """
    Convert text tokens into bigrams using a pre-trained bigram model.

    Args:
        texts (List[List[str]]): List of tokenized documents
        mode (models.Phrases): Pre-trained bigram model

    Returns:
        List[List[str]]: List of documents with bigrams applied
    """
    return [mode[doc] for doc in texts]


def make_trigrams(
    texts: List[List[str]],
    trigram_mod: models.Phrases,
    bigram_mod: models.Phrases,
) -> List[List[str]]:
    """
    Convert text tokens into trigrams using pre-trained bigram and trigram models.

    Args:
        texts (List[List[str]]): List of tokenized documents
        trigram_mod (models.Phrases): Pre-trained trigram model
        bigram_mod (models.Phrases): Pre-trained bigram model

    Returns:
        List[List[str]]: List of documents with trigrams applied
    """
    return [trigram_mod[bigram_mod[doc]] for doc in texts]


def pre_processing_helper(
    texts: List[List[str]],
    mode: str,
    config: Dict,
) -> Tuple[
    corpora.Dictionary,  # dictionary
    List[List[Tuple[int, int]]],  # bow_corpus
]:
    # Create Dictionary - mapping of unique ids to words in the documents
    dic = corpora.Dictionary(texts)
    logger.info("Number of unique tokens in %s mode: %d", mode, len(dic))

    # # Filter out tokens that appear in less than 3 documents or more than 80%
    # # of documents
    # dic.filter_extremes(
    #     no_below=config.get("filter_extremes", {}).get("no_below", 3),
    #     no_above=config.get("filter_extremes", {}).get("no_above", 0.8),
    #     keep_n=config.get("filter_extremes", {}).get("keep_n", 100000),
    # )
    # logger.info(
    #     "Number of unique tokens after filtering in %s mode: %d",
    #     mode,
    #     len(dic),
    # )

    bc = [dic.doc2bow(text) for text in texts]

    return (
        dic,
        bc,
    )


@timer_decorator
def pre_processing_gensim(
    documents_generator: Iterator[List[str]],
    num_cores: int,
    config: Dict,
    mode: str,
    checkpoint_dir: str = None,
    resume: bool = False,
) -> Tuple[
    Dict[int, str],
    List[List[str]],
    List[List[Tuple[int, int]]],
    List[List[Tuple[int, float]]],
]:
    """
    Preprocess documents based on model type and configuration.

    Args:
        documents_generator (Iterator[List[str]]): Iterator of document batches
        num_cores (int): Number of CPU cores to use
        config (Dict): Configuration dictionary from YAML
        mode (str): Training or testing mode
        checkpoint_dir (str, optional): Directory to save checkpoints for resumability
        resume (bool): Whether to resume from existing checkpoint

    Returns:
        Tuple containing:
        - id2word (Dict[int, str]): Mapping of unique ids to words in the documents
        - texts (List[List[str]]): Preprocessed document texts
        - bow_corpus (List[List[Tuple[int, int]]]): Bag of words corpus
        - tfidf_corpus (List[List[Tuple[int, float]]]): TF-IDF corpus
    """

    num_cores = min(num_cores, mp.cpu_count()) if num_cores else mp.cpu_count()

    # Get base stopwords from NLTK
    stop_words = set(stopwords.words("english"))

    # Define paths to additional stopwords files
    stopwords_files_path = config.get("stop_words_extra", "./stopwords")
    stopwords_files = [
        os.path.join(stopwords_files_path, file)
        for file in os.listdir(stopwords_files_path)
    ]

    # Add domain-specific stopwords from files
    for stopwords_file in stopwords_files:
        try:
            additional_stopwords = load_stopwords_from_file(stopwords_file)
            stop_words.update(additional_stopwords)
            # logger.info(
            #     "Added %d stopwords from %s", len(additional_stopwords), stopwords_file
            # )
        except Exception as e:
            logger.warning(
                "Could not load stopwords from %s: %s", stopwords_file, str(e)
            )

    # Checkpoint handling
    processed_batches = []
    checkpoint_path = None
    if checkpoint_dir:
        checkpoint_path = Path(checkpoint_dir) / f"{mode}_checkpoint.json"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        if resume and checkpoint_path.exists():
            try:
                with open(checkpoint_path, "r") as f:
                    checkpoint_data = json.load(f)
                    processed_batches = checkpoint_data.get("processed_batches", [])
                    logger.info(
                        "Resuming %s from batch %d",
                        mode,
                        len(processed_batches),
                    )
            except Exception as e:
                logger.warning(
                    "Failed to load checkpoint: %s. Starting fresh.",
                    e,
                )
                processed_batches = []

    all_texts = []
    batch_num = 0

    # Check if the generator is empty and wrap with tqdm for progress
    empty_generator = True
    documents_list = list(documents_generator)  # Convert to list for tqdm

    if not documents_list:
        empty_generator = True
    else:
        empty_generator = False

    for batch_documents in tqdm(
        documents_list,
        desc=f"Processing {mode} batches",
    ):
        batch_num += 1

        # Skip if resuming and batch already processed
        if resume and batch_num <= len(processed_batches):
            logger.debug("Skipping batch %d (already processed)", batch_num)
            continue

        if not batch_documents:
            logger.warning("Empty batch received, skipping")
            continue

        # Prepare arguments for parallel processing
        process_args = [(doc, stop_words) for doc in batch_documents]

        # Use ProcessPoolExecutor for CPU-bound preprocessing with spaCy worker initialization
        with ProcessPoolExecutor(
            max_workers=num_cores,
            initializer=init_worker,
            initargs=(
                config.get("spacy_model", "en_core_web_sm"),
                config.get("spacy_disabled", ["parser", "ner"]),
                config.get("allowed_postags", ["NOUN", "ADJ", "VERB", "ADV"]),
            ),
        ):
            processed_batch = process_map(
                process_document_chunk,
                process_args,
                max_workers=num_cores,
                desc="Processing documents",
                chunksize=1,
            )

        # Lemmatization already done in worker processes
        all_texts.extend(processed_batch)  # Type: List[List[str]]

        # Note: Stopword and number removal already done in process_document_chunk
        # No need for additional cleaning here

        # Save checkpoint after each batch
        if checkpoint_path:
            processed_batches.append(batch_num)
            checkpoint_data = {
                "processed_batches": processed_batches,
                "timestamp": datetime.now().isoformat(),
                "mode": mode,
                "total_documents": len(all_texts),
            }
            try:
                with open(checkpoint_path, "w") as f:
                    json.dump(checkpoint_data, f, indent=2)
                logger.debug("Checkpoint saved after batch %d", batch_num)
            except Exception as e:
                logger.warning("Failed to save checkpoint: %s", e)

        del processed_batch

    if empty_generator:
        logger.error("No documents were provided for processing")
        return (
            corpora.Dictionary(),
            [],
            [],
            [],
        )

    if not all_texts:
        logger.error("No valid documents were processed")
        return (
            corpora.Dictionary(),
            [],
            [],
            [],
        )

    # Build bigram and trigram models on ALL texts (not per batch)
    bigram = models.Phrases(
        all_texts,
        min_count=5,
        threshold=100,
    )  # higher threshold fewer phrases.
    trigram = models.Phrases(
        bigram[all_texts],
        threshold=100,
    )

    # Faster way to get a sentence clubbed as a trigram/bigram
    bigram_mod = models.phrases.Phraser(bigram)
    trigram_mod = models.phrases.Phraser(trigram)

    # Apply bigrams and trigrams to all texts
    all_texts = make_bigrams(all_texts, bigram_mod)
    all_texts = make_trigrams(all_texts, trigram_mod, bigram_mod)

    # Clean up n-gram models to save memory
    del bigram, trigram, bigram_mod, trigram_mod

    (
        dictionary_gensim,
        bow_corpus_gensim,
    ) = pre_processing_helper(all_texts, mode, config)

    # Validate preprocessing output
    if not validate_preprocessing_output(
        all_texts,
        dictionary_gensim,
        bow_corpus_gensim,
        mode,
    ):
        logger.warning(
            "Preprocessing validation failed for %s mode, but continuing...",
            mode,
        )

    # Clean up checkpoint file on successful completion
    if checkpoint_path and checkpoint_path.exists():
        try:
            checkpoint_path.unlink()
            logger.debug("Checkpoint file removed after successful completion")
        except Exception as e:
            logger.warning("Failed to remove checkpoint file: %s", e)

    return (
        dictionary_gensim,
        bow_corpus_gensim,
        all_texts,
    )


def check_empty_docs(texts: List[List[str]], mode: str) -> bool:
    """
    Check for empty documents in the text data.

    Args:
        texts: List of preprocessed document texts
        mode: Processing mode (train/test)

    Returns:
        bool: True if empty document ratio is less than 50%, False otherwise
    """
    # Check for empty documents
    empty_docs = np.sum(np.array([len(doc) == 0 for doc in texts]))
    empty_ratio = empty_docs / len(texts) if texts else 1.0

    if empty_ratio > 0.5:
        logger.warning(
            "[%s] High empty document ratio: %d/%d (%.1f%%)",
            mode,
            empty_docs,
            len(texts),
            empty_ratio * 100,
        )
        return False
    return True


def check_document_lengths(texts: List[List[str]], mode: str) -> bool:
    """
    Check document lengths in the text data.

    Args:
        texts: List of preprocessed document texts
        mode: Processing mode (train/test)

    Returns:
        bool: True if document lengths are valid, False otherwise
    """
    # Check document stats
    if texts:
        doc_lengths = np.array([len(doc) for doc in texts])
        max_length = np.max(doc_lengths)
        min_length = np.min(doc_lengths)
        avg_length = np.mean(doc_lengths)
        median_length = np.median(doc_lengths)

        logger.info(
            "[%s] Document length stats - Min: %d, Max: %d, Avg: %.1f, Median: %.1f",
            mode,
            min_length,
            max_length,
            avg_length,
            median_length,
        )

        if max_length < 10:
            logger.error(
                "[%s] All documents too short (max=%d)",
                mode,
                max_length,
            )
            return False

    return True


def check_corpus_text_alignment(
    texts: List[List[str]], corpus: List[List[Tuple[int, int]]], mode: str
) -> bool:
    """
    Check corpus-text alignment.

    Args:
        texts: List of preprocessed document texts
        corpus: Bag of words corpus
        mode: Processing mode (train/test)

    Returns:
        bool: True if corpus-text alignment is valid, False otherwise
    """
    # Check corpus-text alignment
    if len(corpus) != len(texts):
        logger.error(
            "[%s] Mismatch: %d corpus docs vs %d text docs",
            mode,
            len(corpus),
            len(texts),
        )
        return False
    return True


def validate_preprocessing_output(
    texts: List[List[str]],
    corpus: List[List[Tuple[int, int]]],
    mode: str,
) -> bool:
    """
    Validate preprocessing output with minimal sanity checks.

    Args:
        texts: List of preprocessed document texts
        corpus: Bag of words corpus
        mode: Processing mode (train/test)

    Returns:
        bool: True if validation passes, False otherwise
    """

    if not check_empty_docs(texts, mode):
        return False

    if not check_document_lengths(texts, mode):
        return False

    if not check_corpus_text_alignment(texts, corpus, mode):
        return False

    return True


def test_corpus_filtering(
    dic: corpora.Dictionary,
    test_texts: List[List[str]],
) -> List[List[Tuple[int, int]]]:
    """
    Filter the test corpus based on the train dictionary and tfidf model.

    Args:
        dic (corpora.Dictionary): Gensim dictionary mapping word IDs to words for the train corpus
        test_texts (List[List[str]]): List of tokenized documents for the test corpus

    Returns:
        List[List[Tuple[int, int]]]: Filtered bow corpus
        Example:
        Dictionary: {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4}
        Test texts: [['a', 'b', 'c'], ['b', 'c'], ['a', 'b', 'b', 'e']]
        Filtered bow corpus: [
            [(0, 1), (1, 1), (2, 1)],
            [(1, 1), (2, 1)],
            [(0, 1), (1, 2), (4, 1)],
        ]
    """
    bow_corpus = [dic.doc2bow(text) for text in test_texts]

    return bow_corpus


def load_stop_words(config: Dict) -> set:
    """
    Load stop words from a configuration dictionary.
    """
    # Get base stopwords from NLTK
    stop_words = set(stopwords.words("english"))

    # Define paths to additional stopwords files
    stopwords_files_path = config.get("stop_words_extra", "./stopwords")
    stopwords_files = [
        os.path.join(stopwords_files_path, file)
        for file in os.listdir(stopwords_files_path)
    ]

    for stopwords_file in stopwords_files:
        try:
            additional_stopwords = load_stopwords_from_file(stopwords_file)
            stop_words.update(additional_stopwords)
        except Exception as e:
            logger.warning(
                "Could not load stopwords from %s: %s", stopwords_file, str(e)
            )
    return stop_words


def preprocess_mp(
    batch_documents: List[str],
    stop_words: set,
    num_cores: int,
    config: Dict,
) -> List[List[str]]:
    # Prepare arguments for parallel processing
    process_args = [(doc, stop_words) for doc in batch_documents]

    # Use ProcessPoolExecutor for CPU-bound preprocessing with spaCy worker initialization
    with ProcessPoolExecutor(
        max_workers=num_cores,
        initializer=init_worker,
        initargs=(
            config.get("spacy_model", "en_core_web_sm"),
            config.get("spacy_disabled", ["parser", "ner"]),
            config.get("allowed_postags", ["NOUN", "ADJ", "VERB", "ADV"]),
        ),
    ):
        processed_batch = process_map(
            process_document_chunk,
            process_args,
            max_workers=num_cores,
            chunksize=1,
        )

    return processed_batch


def build_bigrams_trigrams(
    all_texts: List[List[str]],
) -> Tuple[models.phrases.Phraser, models.phrases.Phraser]:
    # Build bigram and trigram models on ALL texts
    bigram = models.Phrases(
        all_texts,
        min_count=5,
        threshold=100,
    )
    trigram = models.Phrases(
        bigram[all_texts],
        threshold=100,
    )

    # Faster way to get a sentence clubbed as a trigram/bigram
    bigram_mod = models.phrases.Phraser(bigram)
    trigram_mod = models.phrases.Phraser(trigram)

    del bigram, trigram

    return bigram_mod, trigram_mod


def create_shared_dictionary(
    documents_generators: List[Iterator[List[str]]],
    num_cores: int,
    config: Dict,
) -> corpora.Dictionary:
    """
    Create a shared dictionary from multiple document generators (e.g., IPO and Analyst reports).
    This ensures both report types use the same vocabulary.

    Args:
        documents_generators: List of document generators (e.g., [ipo_generator, analyst_generator])
        num_cores: Number of CPU cores to use
        config: Configuration dictionary from YAML

    Returns:
        corpora.Dictionary: Shared dictionary for all document types
    """
    num_cores = min(num_cores, mp.cpu_count()) if num_cores else mp.cpu_count()

    stop_words = load_stop_words(config)

    all_texts = []

    # Process all document generators.
    # First - ipo documents
    # Second - analyst documents
    for gen_idx, documents_generator in enumerate(documents_generators):
        logger.info(
            f"Processing document generator {'ipo' if gen_idx == 0 else 'analyst'}/{len(documents_generators)}"
        )

        documents_list = list(documents_generator)

        for batch_documents in tqdm(
            documents_list,
            desc=f"Processing generator {'ipo' if gen_idx == 0 else 'analyst'} batches for shared dictionary",
        ):
            if not batch_documents:
                logger.warning("Empty batch received, skipping")
                continue

            processed_batch = preprocess_mp(
                batch_documents=batch_documents,
                stop_words=stop_words,
                num_cores=num_cores,
                config=config,
            )
            all_texts.extend(processed_batch)
            del processed_batch

    if not all_texts:
        logger.error("No valid documents were processed for shared dictionary")
        return corpora.Dictionary()

    logger.info(f"Total documents processed for shared dictionary: {len(all_texts)}")

    # Build bigram and trigram models on ALL texts
    bigram_mod, trigram_mod = build_bigrams_trigrams(all_texts)

    # Apply bigrams and trigrams to all texts
    all_texts = make_bigrams(all_texts, bigram_mod)
    all_texts = make_trigrams(all_texts, trigram_mod, bigram_mod)

    # Clean up n-gram models to save memory
    del bigram_mod, trigram_mod

    # Create Dictionary - mapping of unique ids to words in the documents
    shared_dic = corpora.Dictionary(all_texts)
    logger.info("Number of unique tokens in shared dictionary: %d", len(shared_dic))

    # # Filter out tokens that appear in less than N documents or more than X% of documents
    # shared_dic.filter_extremes(
    #     no_below=config.get("filter_extremes", {}).get("no_below", 3),
    #     no_above=config.get("filter_extremes", {}).get("no_above", 0.8),
    #     keep_n=config.get("filter_extremes", {}).get("keep_n", 100000),
    # )
    # logger.info(
    #     "Number of unique tokens after filtering in shared dictionary: %d",
    #     len(shared_dic),
    # )

    return shared_dic


def pre_processing_with_dictionary(
    documents_generator: Iterator[List[str]],
    shared_dictionary: corpora.Dictionary,
    num_cores: int,
    config: Dict,
    mode: str,
) -> Tuple[
    List[List[Tuple[int, int]]],
    List[List[str]],
]:
    """
    Preprocess documents using a pre-built shared dictionary.

    Args:
        documents_generator: Iterator of document batches
        shared_dictionary: Pre-built shared dictionary
        num_cores: Number of CPU cores to use
        config: Configuration dictionary from YAML
        mode: Training or testing mode

    Returns:
        Tuple containing:
        - bow_corpus: Bag of words corpus
        - texts: Preprocessed document texts
    """
    num_cores = min(num_cores, mp.cpu_count()) if num_cores else mp.cpu_count()

    # Get base stopwords from NLTK
    stop_words = load_stop_words(config)

    all_texts = []
    documents_list = list(documents_generator)

    for batch_documents in tqdm(
        documents_list,
        desc=f"Processing {mode} batches with shared dictionary",
    ):
        if not batch_documents:
            logger.warning("Empty batch received, skipping")
            continue

        processed_batch = preprocess_mp(
            batch_documents=batch_documents,
            stop_words=stop_words,
            num_cores=num_cores,
            config=config,
        )
        all_texts.extend(processed_batch)
        del processed_batch

    if not all_texts:
        logger.error("No valid documents were processed")
        return [], [], []

    # Build bigram and trigram models on ALL texts
    bigram_mod, trigram_mod = build_bigrams_trigrams(all_texts)

    # Apply bigrams and trigrams to all texts
    all_texts = make_bigrams(all_texts, bigram_mod)
    all_texts = make_trigrams(all_texts, trigram_mod, bigram_mod)

    # Clean up n-gram models to save memory
    del bigram_mod, trigram_mod

    # Validate preprocessing output
    if not validate_preprocessing_output(all_texts, shared_dictionary, mode):
        logger.error(
            "Preprocessing validation failed for %s mode",
            mode,
        )

    # Use the shared dictionary to create BOW corpus
    bow_corpus = [shared_dictionary.doc2bow(text) for text in all_texts]

    logger.info(
        f"Created {mode} corpus with {len(bow_corpus)} documents using shared dictionary"
    )

    return bow_corpus, all_texts


def collect_processed_texts(
    document_sources: List[Tuple[Optional[str], Iterator[List[str]]]],
    num_cores: int,
    config: Dict,
    mode: str,
    checkpoint_dir: Optional[Union[str, Path]] = None,
    resume: bool = False,
) -> Tuple[List[List[str]], Optional[Path]]:
    """Collect and preprocess documents from one or more generators."""

    logger = logging.getLogger(__name__)

    # Normalize sources to list of (label, generator)
    if isinstance(document_sources, (list, tuple)):
        normalized_sources: List[Tuple[Optional[str], Iterator[List[str]]]] = []
        for item in document_sources:
            if isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], str):
                label, generator = item
            else:
                label, generator = None, cast(Iterator[List[str]], item)
            normalized_sources.append((label, generator))
    else:
        normalized_sources = [(None, cast(Iterator[List[str]], document_sources))]

    num_cores = min(num_cores, mp.cpu_count()) if num_cores else mp.cpu_count()
    stop_words = load_stop_words(config)

    checkpoint_path: Optional[Path] = None
    processed_batches: List[int] = []

    if checkpoint_dir:
        checkpoint_path = Path(checkpoint_dir) / f"{mode}_checkpoint.json"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        if resume and checkpoint_path.exists():
            try:
                with open(checkpoint_path, "r", encoding="utf-8") as f:
                    checkpoint_data = json.load(f)
                processed_batches = checkpoint_data.get("processed_batches", [])
                logger.info(
                    "Resuming %s from batch %d",
                    mode,
                    len(processed_batches),
                )
            except Exception as exc:
                logger.warning(
                    "Failed to load checkpoint for %s: %s. Starting fresh.",
                    mode,
                    exc,
                )
                processed_batches = []

    all_texts: List[List[str]] = []
    batch_num = 0
    any_batches = False

    for source_idx, (label, generator) in enumerate(normalized_sources, start=1):
        documents_list = list(generator)
        if not documents_list:
            continue

        any_batches = True
        source_label = label or (
            f"{mode} source {source_idx}" if mode else f"source {source_idx}"
        )

        for batch_documents in tqdm(
            documents_list,
            desc=f"Processing {source_label} batches",
        ):
            batch_num += 1

            if resume and batch_num <= len(processed_batches):
                logger.debug(
                    "Skipping batch %d for %s (already processed)",
                    batch_num,
                    source_label,
                )
                continue

            if not batch_documents:
                logger.warning("Empty batch received for %s, skipping", source_label)
                continue

            processed_batch = preprocess_mp(
                batch_documents=batch_documents,
                stop_words=stop_words,
                num_cores=num_cores,
                config=config,
            )
            all_texts.extend(processed_batch)
            del processed_batch

            if checkpoint_path:
                processed_batches.append(batch_num)
                checkpoint_data = {
                    "processed_batches": processed_batches,
                    "timestamp": datetime.now().isoformat(),
                    "mode": mode,
                    "total_documents": len(all_texts),
                }
                try:
                    with open(checkpoint_path, "w", encoding="utf-8") as f:
                        json.dump(checkpoint_data, f, indent=2)
                except Exception as exc:
                    logger.warning("Failed to save checkpoint for %s: %s", mode, exc)

    if not any_batches:
        logger.error("No documents were provided for processing in %s mode", mode)

    if not all_texts and checkpoint_path and checkpoint_path.exists():
        logger.warning(
            "No valid documents processed for %s; retaining checkpoint at %s",
            mode,
            checkpoint_path,
        )

    return all_texts, checkpoint_path


def prepare_dictionary_and_corpus(
    texts: List[List[str]],
    config: Dict,
    mode: str,
    dictionary: Optional[corpora.Dictionary] = None,
    validate: bool = True,
) -> Tuple[corpora.Dictionary, List[List[Tuple[int, int]]], List[List[str]]]:
    """Apply n-gram models and build corpus (optionally creating the dictionary)."""

    logger = logging.getLogger(__name__)

    if not texts:
        if dictionary is None:
            dictionary = corpora.Dictionary()
        return dictionary, [], []

    bigram_mod, trigram_mod = build_bigrams_trigrams(texts)
    texts = make_bigrams(texts, bigram_mod)
    texts = make_trigrams(texts, trigram_mod, bigram_mod)
    del bigram_mod, trigram_mod

    if dictionary is None:
        dictionary, bow_corpus = pre_processing_helper(texts, mode, config)
    else:
        bow_corpus = [dictionary.doc2bow(text) for text in texts]

    if validate and not validate_preprocessing_output(texts, bow_corpus, mode):
        logger.warning("Preprocessing validation failed for %s mode", mode)

    return dictionary, bow_corpus, texts


def preprocess_documents_build_dictionary(
    documents_generator: Iterator[List[str]],
    num_cores: int,
    config: Dict,
    mode: str,
    checkpoint_dir: Optional[Union[str, Path]] = None,
    resume: bool = False,
    validate: bool = True,
) -> Tuple[corpora.Dictionary, List[List[Tuple[int, int]]], List[List[str]]]:
    """Preprocess documents and create a fresh dictionary and corpus."""

    all_texts, checkpoint_path = collect_processed_texts(
        document_sources=documents_generator,
        num_cores=num_cores,
        config=config,
        mode=mode,
        checkpoint_dir=checkpoint_dir,
        resume=resume,
    )

    if not all_texts:
        return corpora.Dictionary(), [], []

    dictionary, bow_corpus, texts = prepare_dictionary_and_corpus(
        texts=all_texts,
        config=config,
        mode=mode,
        dictionary=None,
        validate=validate,
    )

    if checkpoint_path and checkpoint_path.exists():
        try:
            checkpoint_path.unlink()
            logger.debug(
                "Checkpoint file %s removed after successful %s preprocessing",
                checkpoint_path,
                mode,
            )
        except Exception as exc:
            logger.warning(
                "Failed to remove checkpoint file %s: %s",
                checkpoint_path,
                exc,
            )

    return dictionary, bow_corpus, texts


def preprocess_documents_with_dictionary(
    documents_generator: Iterator[List[str]],
    shared_dictionary: corpora.Dictionary,
    num_cores: int,
    config: Dict,
    mode: str,
    validate: bool = True,
) -> Tuple[List[List[Tuple[int, int]]], List[List[str]]]:
    """Preprocess documents using a provided dictionary."""

    all_texts, _ = collect_processed_texts(
        document_sources=documents_generator,
        num_cores=num_cores,
        config=config,
        mode=mode,
    )

    if not all_texts:
        return [], []

    _, bow_corpus, texts = prepare_dictionary_and_corpus(
        texts=all_texts,
        config=config,
        mode=mode,
        dictionary=shared_dictionary,
        validate=validate,
    )

    return bow_corpus, texts


def build_dictionary_from_generators(
    documents_generators: Sequence[Iterator[List[str]]],
    num_cores: int,
    config: Dict,
    source_labels: Optional[Sequence[str]] = None,
    validate: bool = False,
) -> corpora.Dictionary:
    """Create a dictionary from multiple generators using the modular pipeline."""

    if source_labels and len(source_labels) != len(documents_generators):
        raise ValueError("source_labels must match the number of document generators")

    normalized_sources: List[Tuple[Optional[str], Iterator[List[str]]]] = []
    if source_labels:
        for label, generator in zip(source_labels, documents_generators):
            normalized_sources.append(
                (label, generator)
            )  # [('ipo', 'ipo_generator'), ('analyst', 'analyst_generator')]
    else:
        for generator in documents_generators:
            normalized_sources.append(
                (None, generator)
            )  # [(None, 'ipo_generator'), (None, 'analyst_generator')]

    all_texts, _ = collect_processed_texts(
        document_sources=normalized_sources,
        num_cores=num_cores,
        config=config,
        mode="shared",
    )

    dictionary, _, _ = prepare_dictionary_and_corpus(
        texts=all_texts,
        config=config,
        mode="shared",
        dictionary=None,
        validate=validate,
    )

    return dictionary


if __name__ == "__main__":
    batch_size = 100
    num_docs = 0
    num_cores = 8

    # Load CSV and extract file paths for both report types
    df = pd.read_csv("data/final_analyst_reports_for_latest_s1_filings_extracted.csv")

    # Extract file paths for both report types
    ipo_file_paths = df["s1_path"].dropna().tolist()
    analyst_file_paths = df["analyst_report_path"].dropna().tolist()
    config = utils.load_config("config.yaml")

    # Load all documents for shared dictionary creation
    ipo_all_docs_generator, _, _ = utils.load_files_in_batches_from_paths(
        file_paths=ipo_file_paths,
        batch_size=batch_size,
        num_docs=num_docs,
        test_perc=0.1,  # Load all as training for dictionary
    )

    analyst_all_docs_generator, _, _ = utils.load_files_in_batches_from_paths(
        file_paths=analyst_file_paths,
        batch_size=batch_size,
        num_docs=num_docs,
        test_perc=0.1,  # Load all as training for dictionary
    )

    # Create shared dictionary from both document types
    shared_dictionary = build_dictionary_from_generators(
        documents_generators=[
            ipo_all_docs_generator,
            analyst_all_docs_generator,
        ],
        num_cores=num_cores,
        config=config["preprocessing"],
        source_labels=["ipo", "analyst"],
    )
