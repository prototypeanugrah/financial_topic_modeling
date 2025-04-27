from datetime import datetime
from concurrent.futures import ProcessPoolExecutor
from typing import Tuple, Dict, List, Any, Iterator
import argparse
import logging
import multiprocessing as mp
import requests
import time
import yaml

from gensim.utils import simple_preprocess
from gensim import corpora
from gensim import models
from nltk.corpus import stopwords
from tqdm.contrib.concurrent import process_map
import numpy as np
import regex as re
import spacy

import utils


logger = logging.getLogger(__name__)


def load_data(
    data_path: str,
    num_docs: int = 5,
) -> List[str]:
    """
    Load and prepare text documents from the first num_docs URLs listed in the file.

    Args:
        data_path: Path to file containing URLs
        num_docs: Number of documents to load (default: 5)

    Returns:
        List of document texts
    """
    try:
        logger.info("Loading first %d URLs from %s", num_docs, data_path)
        documents = []

        with open(data_path, "r", encoding="utf-8") as f:
            # Get first num_docs non-empty lines
            urls = [line.strip() for line in f if line.strip()][:num_docs]

        if not urls:
            raise ValueError(f"No URLs found in {data_path}")

        logger.info("Found %d URLs to process", len(urls))

        # SEC specific headers with email identification
        headers = {
            "User-Agent": "Sample Company Name AdminContact@company.com",
            "Host": "www.sec.gov",
        }

        # Fetch each document
        for i, url in enumerate(urls, 1):
            try:
                logger.info("Fetching document %d from %s", i, url)
                response = requests.get(url, headers=headers)
                response.raise_for_status()  # Raise exception for bad status codes
                documents.append(response.text)
                logger.info("Successfully loaded document %d", i)
            except Exception as e:
                logger.error("Failed to fetch document from %s: %s", url, str(e))
                # Continue with other documents even if one fails
                continue

        if not documents:
            raise ValueError("Failed to load any documents")

        logger.info("Successfully loaded %d documents", len(documents))
        return documents

    except Exception as e:
        logger.error("Error loading data: %s", str(e))
        raise


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# Preprocess the text data
def basic_preprocessing(text: str) -> str:
    """
    Basic preprocessing of text data to remove unwanted characters and metadata.
    This includes:
    - Removing everything before <SEC-DOCUMENT> tag
    - Removing everything after </HTML> tag
    - Removing all HTML tags (content between < and >)
    - Removing SEC header metadata block
    - Removing document type markers and page markers
    - Removing extra spaces, emails, apostrophes, and non-alphabet characters
    - Converting text to lowercase

    Args:
        text (str): Input text to preprocess

    Returns:
        str: Preprocessed text
    """

    # Remove everything before <SEC-DOCUMENT> tag
    text = re.sub(
        r"^.*?<SEC-DOCUMENT>", "<SEC-DOCUMENT>", text, flags=re.DOTALL | re.MULTILINE
    )

    # Remove SEC header starting from <SEC-HEADER> metadata block until </SEC-HEADER>
    text = re.sub(
        r"<SEC-HEADER>.*?</SEC-HEADER>", "", text, flags=re.DOTALL | re.MULTILINE
    )

    # First find and keep only the S-1 and S-1/A document section
    s1_pattern = r"<DOCUMENT>(.*?<TYPE>(?:S-1|S-1/A).*?)</DOCUMENT>"
    s1_matches = re.findall(s1_pattern, text, flags=re.DOTALL | re.MULTILINE)

    if not s1_matches:
        logger.warning("No S-1 document found")
        return ""

    # Take the first S-1 document if multiple exist
    text = s1_matches[0]
    # logger.info("Length of S-1 document: %d characters", len(text))

    # # Remove all content between <TABLE> and </TABLE> tags
    # text = re.sub(r"<TABLE>.*?</TABLE>", "", text, flags=re.DOTALL | re.MULTILINE)

    # # Remove document type markers and page markers
    # text = re.sub(
    #     r"<DOCUMENT>|<TYPE>.*?</TYPE>|<SEQUENCE>.*?</SEQUENCE>|"
    #     r"<DESCRIPTION>.*?</DESCRIPTION>|<TEXT>|<PAGE>",
    #     " ",
    #     text,
    #     flags=re.DOTALL | re.MULTILINE,
    # )

    # Remove all HTML tags (content between < and >)
    text = re.sub(r"<[^>]*>", "", text, flags=re.DOTALL | re.MULTILINE)

    # Standard cleaning steps
    text = re.sub(r"\S*@\S*\s?", "", text)  # Remove emails
    text = re.sub(r"'", "", text)  # Remove apostrophes
    text = re.sub(r"&nbsp;", " ", text)  # Remove &nbsp;
    # with open("first_document_cleaned.txt", "w", encoding="utf-8") as f:
    #     f.write(text)
    text = re.sub(r"[^a-zA-Z]", " ", text)  # Remove non-alphabet characters
    text = text.lower()  # Convert to lowercase
    text = re.sub(r"\s+", " ", text)  # Remove extra spaces

    # logger.info(
    #     "Length of cleaned document after preprocessing: %d characters", len(text)
    # )
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
        UnicodeDecodeError: If there's an encoding issue reading the file
    """
    try:
        with open(filepath, "r", encoding="utf-8") as file:
            # Read words, strip whitespace, and convert to lowercase
            return [word.strip().lower() for word in file.readlines() if word.strip()]
    except FileNotFoundError:
        logger.warning("Stopwords file not found: %s", filepath)
        return []
    except UnicodeDecodeError:
        logger.error("Encoding error reading stopwords file: %s", filepath)
        return []
    except Exception as e:
        logger.error("Unexpected error reading stopwords file %s: %s", filepath, str(e))
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
    texts: List[str],
    stop_words: List[str],
) -> List[List[str]]:
    """
    Remove stopwords from tokenized texts.

    Args:
        texts (List[str]): List of tokenized texts (each text is a list of
        words)
        stop_words (List[str]): List of stopwords to remove

    Returns:
        List[List[str]]: List of texts with stopwords removed (each text is a
        list of words)
    """
    return [
        [word for word in simple_preprocess(str(doc)) if word not in stop_words]
        for doc in texts
    ]


def lemmatization(
    tokens: List[str],
    nlp,
    allowed_postags: List[str],
) -> List[List[str]]:
    """
    Perform lemmatization on tokenized texts using spaCy.

    Args:
        tokens (List[str]): List of tokenized texts (each text is a list of
        words)
        nlp: spaCy language model for lemmatization
        allowed_postags (List[str]): List of allowed part-of-speech tags for
        lemmatization

    Returns:
        List[List[str]]: List of lemmatized texts (each text is a list of
        words)

    Note:
    - The `allowed_postags` parameter specifies which part-of-speech tags
        to keep during lemmatization. By default, it includes nouns, adjectives,
        verbs, and adverbs.
    """
    tokens_out = []
    for token_chunk in tokens:
        # Join the tokens back into text
        full_text = " ".join(token_chunk)

        # Split into chunks if text is too long
        chunks = split_text_into_chunks(full_text)

        # Process each chunk separately
        processed_chunks = []
        for chunk in chunks:
            doc = nlp(chunk)
            processed_chunk = [
                token.lemma_ for token in doc if token.pos_ in allowed_postags
            ]
            processed_chunks.extend(processed_chunk)

        tokens_out.append(processed_chunks)

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
        logger.info("%s took %.2f seconds to execute", func.__name__, execution_time)
        return result

    return wrapper


def process_document_chunk(
    pdc_args: Tuple[str, set, spacy.language.Language],
) -> List[str]:
    """
    Process a single document through all preprocessing steps.

    Args:
        pdc_args: Tuple containing (document, stop_words, nlp)

    Returns:
        List[str]: Processed tokens from the document

    Note:
    1. Basic preprocessing
    2. Tokenization
    3. Remove stopwords
    4. Remove words with less than 3 characters
    5. Lemmatization
    """
    document, stop_words, nlp = pdc_args

    # Apply preprocessing steps sequentially
    text = basic_preprocessing(document)
    tokens = list(sent_to_words([text]))[0]
    tokens = remove_stopwords([tokens], stop_words)[0]
    # Remove tokens that contain numbers
    tokens = [token for token in tokens if not any(char.isdigit() for char in token)]
    tokens = remove_words_less_than_length_three_characters([tokens])[0]

    return tokens


def split_text_into_chunks(text: str, chunk_size: int = 900000) -> List[str]:
    """
    Split a long text into smaller chunks of specified size.
    This is useful for processing large documents that exceed the maximum
    length allowed by spaCy for lemmatization.

    Args:
        text (str): The input text to be split
        chunk_size (int, optional): The maximum size of each chunk
            (default: 900000 characters). Must be positive.

    Returns:
        List[str]: A list of text chunks, each with a maximum length of
        chunk_size

    Raises:
        ValueError: If chunk_size is not positive
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    if not text:
        return []

    return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]


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


def analyze_corpus(
    id2word: Dict[int, str],
    bow_corpus: List[List[Tuple[int, int]]],
    tfidf_corpus: List[List[Tuple[int, float]]],
    num_words: int = 10,
) -> None:
    """
    Comprehensive analysis of both bow and tfidf corpus.

    Args:
        id2word (Dict[int, str]): Mapping of word IDs to words
        bow_corpus (List[List[Tuple[int, int]]]): Bag of words corpus
        tfidf_corpus (List[List[Tuple[int, float]]]): TF-IDF corpus
        num_words (int): Number of top words to display for comparison

    Returns:
        None
    """
    # Get vocabulary
    vocabulary = list(id2word.values())
    logger.info("Total vocabulary size: %d", len(vocabulary))

    # Print some vocabulary statistics
    logger.info("Vocabulary Sample (first 20 words):")
    logger.info(vocabulary[:20])

    # Analyze document lengths
    doc_lengths = [len(doc) for doc in bow_corpus]
    logger.info("Document length statistics:")
    logger.info(
        "  - Average words per document: %.2f", sum(doc_lengths) / len(doc_lengths)
    )
    logger.info("  - Max words in a document: %d", max(doc_lengths))
    logger.info("  - Min words in a document: %d", min(doc_lengths))

    # Compare BOW vs TF-IDF weights for first document
    if len(bow_corpus) > 0:
        logger.info("Comparing BOW vs TF-IDF weights for first document:")
        bow_doc = bow_corpus[0]
        tfidf_doc = tfidf_corpus[0]

        # Get top 10 words by frequency (BOW) and by TF-IDF weight
        bow_sorted = sorted(bow_doc, key=lambda x: x[1], reverse=True)[:num_words]
        tfidf_sorted = sorted(tfidf_doc, key=lambda x: x[1], reverse=True)[:num_words]

        logger.info("Top %d words by frequency (BOW):", num_words)
        for word_id, freq in bow_sorted:
            logger.info("  - %s: %d", id2word[word_id], freq)

        logger.info("Top %d words by TF-IDF weight:", num_words)
        for word_id, weight in tfidf_sorted:
            logger.info("  - %s: %.3f", id2word[word_id], weight)


def check_vocabulary(
    id2word: Dict[int, str],
    texts: List[List[str]],
) -> None:
    """
    Check the vocabulary and processed texts for debugging purposes.

    Args:
        id2word (Dict[int, str]): Mapping of word IDs to words
        texts (List[List[str]]): List of processed texts (tokenized documents)

    Returns:
        None
    """
    logger.info("\nDebug: Vocabulary Check")
    logger.info("Vocabulary size: %d", len(id2word))
    logger.info(
        "Sample of vocabulary (first 10 words): %s", list(id2word.values())[:10]
    )

    logger.info("\nDebug: Processed Texts Check")
    logger.info("Number of documents: %d", len(texts))
    if texts:
        logger.info("First document length: %d", len(texts[0]))
        logger.info("Sample of first document (first 10 words): %s", texts[0][:10])


def pre_processing_helper(
    texts: List[List[str]],
) -> Tuple[
    corpora.Dictionary,  # dictionary
    List[List[Tuple[int, int]]],  # bow_corpus
    List[List[Tuple[int, float]]],  # tfidf_corpus
]:
    # Create Dictionary - mapping of unique ids to words in the documents
    dic = corpora.Dictionary(texts)

    bc = [dic.doc2bow(text) for text in texts]

    # Create Tf-IDF model
    tfidf = models.TfidfModel(
        corpus=bc,
        id2word=dic,
    )

    # Apply TF-IDF transformation to the corpus
    tfc = tfidf[bc]
    del tfidf

    logger.info("Number of unique tokens: %d", len(dic))

    return (
        dic,
        bc,
        tfc,
    )


@timer_decorator
def pre_processing_gensim(
    documents_generator: Iterator[List[str]],
    num_cores: int,
    config: Dict,
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
    stopwords_files = [
        "stopwords/additional_stopwords.txt",
        "stopwords/financial_stopwords.txt",
        "stopwords/generic_stopwords.txt",
    ]

    # Add domain-specific stopwords from files
    for stopwords_file in stopwords_files:
        try:
            additional_stopwords = load_stopwords_from_file(stopwords_file)
            stop_words.update(additional_stopwords)
            logger.info(
                "Added %d stopwords from %s", len(additional_stopwords), stopwords_file
            )
        except Exception as e:
            logger.warning(
                "Could not load stopwords from %s: %s", stopwords_file, str(e)
            )

    logger.info("Total number of stopwords: %d", len(stop_words))

    # Load the spacy (en_core_web_sm): small English pipeline
    # trained on written web text (blogs, news, comments), that includes
    # vocabulary, syntax and entities.
    try:
        nlp = spacy.load(
            config.get("spacy_model", "en_core_web_sm"),
            disable=config.get("spacy_disabled", ["parser", "ner"]),
        )
    except Exception as e:
        logger.error("Failed to load spacy model: %s", str(e))
        raise

    logger.info("Pre-processing the documents")

    all_bow_corpus = []
    all_tfidf_corpus = []
    all_texts = []

    # Check if the generator is empty
    empty_generator = True
    for batch_documents in documents_generator:
        empty_generator = False
        if not batch_documents:
            logger.warning("Empty batch received, skipping")
            continue

        # Prepare arguments for parallel processing
        process_args = [(doc, stop_words, nlp) for doc in batch_documents]

        # Use ProcessPoolExecutor for CPU-bound preprocessing
        parallel_start = time.time()
        with ProcessPoolExecutor(max_workers=num_cores) as executor:
            # processed_data = list(executor.map(process_document_chunk, process_args))
            processed_batch = process_map(
                process_document_chunk,
                process_args,
                max_workers=num_cores,
                desc="Processing documents",
                chunksize=1,
            )
        parallel_end = time.time()
        logger.info(
            "Completed parallel preprocessing of %d documents",
            len(processed_batch),
        )

        parallel_time = parallel_end - parallel_start
        logger.info(
            "Parallel processing completed in %.2f seconds",
            parallel_time,
        )
        logger.info(
            "Average time per document: %.2f seconds",
            parallel_time / len(batch_documents),
        )

        # 6. Build the bigram and trigram models
        bigram = models.Phrases(
            processed_batch,
            min_count=5,
            threshold=200,
        )  # higher threshold fewer phrases.
        trigram = models.Phrases(
            bigram[processed_batch],
            threshold=200,
        )

        # Faster way to get a sentence clubbed as a trigram/bigram
        bigram_mod = models.phrases.Phraser(bigram)
        trigram_mod = models.phrases.Phraser(trigram)

        # Form Bigrams
        batch_bigrams = make_bigrams(processed_batch, bigram_mod)
        batch_bigrams_trigrams = make_trigrams(
            batch_bigrams,
            trigram_mod,
            bigram_mod,
        )

        # Lemmatize the tokens
        data_lemmatized = lemmatization(
            batch_bigrams_trigrams,
            nlp,
            allowed_postags=config.get(
                "allowed_postags", ["NOUN", "ADJ", "VERB", "ADV"]
            ),
        )

        all_texts.extend(data_lemmatized)

        del bigram
        del bigram_mod
        del batch_bigrams
        del batch_bigrams_trigrams
        del data_lemmatized
        del processed_batch
        del trigram
        del trigram_mod

    if empty_generator:
        logger.error("No documents were provided for processing")
        # Return empty structures to avoid errors downstream
        return (
            corpora.Dictionary(),
            [],
            [],
            [],
        )

    if not all_texts:
        logger.error("No valid documents were processed")
        # Return empty structures to avoid errors downstream
        return (
            corpora.Dictionary(),
            [],
            [],
            [],
        )

    dictionary_gensim, bow_corpus_gensim, tfidf_corpus_gensim = pre_processing_helper(
        all_texts,
    )

    if config.get("debug", "enabled") is True:
        check_vocabulary(dictionary_gensim, all_texts)
        analyze_corpus(dictionary_gensim, bow_corpus_gensim, tfidf_corpus_gensim)

    return (
        dictionary_gensim,
        bow_corpus_gensim,
        tfidf_corpus_gensim,
        all_texts,
    )


def filter_corpus_by_tfidf(
    corpus: List[List[Tuple[int, float]]],
    dictionary: corpora.Dictionary,
    low_value_threshold: float = 0.025,
) -> Tuple[List[List[Tuple[int, float]]], corpora.Dictionary]:
    """
    Filter corpus based on TF-IDF scores and remove missing words.

    Args:
        corpus: List of documents in bow format
        dictionary: Gensim dictionary mapping word IDs to words
        low_value_threshold: Minimum TF-IDF value to keep a term (default: 0.025)

    Returns:
        Tuple containing:
        - Filtered corpus
        - Updated dictionary with only retained terms
    """
    # Check if dictionary is empty
    if not dictionary or len(dictionary) == 0:
        logger.warning(
            "Empty dictionary provided. Returning empty corpus and dictionary."
        )
        return [], corpora.Dictionary()

    # Create TF-IDF model
    tfidf_model = models.TfidfModel(corpus=corpus, id2word=dictionary)

    # Track which terms are retained
    retained_term_ids = set()
    filtered_corpus = []

    # Process each document
    for bow in corpus:
        # Get TF-IDF scores for this document
        tfidf_scores = dict(tfidf_model[bow])

        # Filter terms based on conditions:
        # 1. Term must have TF-IDF score above threshold
        # 2. Term must exist in TF-IDF model
        new_bow = [
            (term_id, freq)
            for term_id, freq in bow
            if term_id in tfidf_scores and tfidf_scores[term_id] >= low_value_threshold
        ]

        # Add retained terms to tracking set
        retained_term_ids.update(term_id for term_id, _ in new_bow)
        filtered_corpus.append(new_bow)

    # Create new dictionary with only retained terms
    new_dictionary = corpora.Dictionary()
    for term_id in retained_term_ids:
        if term_id in dictionary.id2token:
            token = dictionary.id2token[term_id]
            new_dictionary.doc2bow([token], allow_update=True)

    # Print statistics
    initial_terms = len(dictionary)
    final_terms = len(new_dictionary)
    removed_terms = initial_terms - final_terms

    # Only calculate percentage if initial_terms is not zero
    percentage_removed = (
        (removed_terms / initial_terms) * 100 if initial_terms > 0 else 0
    )

    logger.info("\nFiltering Statistics:")
    logger.info(f"Initial vocabulary size: {initial_terms}")
    logger.info(f"Terms removed: {removed_terms} ({percentage_removed:.1f}%)")
    logger.info(f"Final vocabulary size: {final_terms}")

    return filtered_corpus, new_dictionary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LDA Topic Modeling Pipeline")
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
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
        help="Number of documents to process in each batch. If not provided, the batch size will be 100.",
    )
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    documents_generator = utils.load_files_in_batches(
        batch_size=args.batch_size,
        num_docs=args.num_docs,
    )
    (
        dictionary,
        bow_corpus,
        tfidf_corpus,
        all_texts,
    ) = pre_processing_gensim(
        documents_generator,
        num_cores=args.num_cores,
        config=config["preprocessing"],
    )
