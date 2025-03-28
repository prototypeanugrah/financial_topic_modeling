from datetime import datetime
from concurrent.futures import ProcessPoolExecutor
from typing import Tuple, Dict, List, Any, Iterator
import argparse
import logging
import multiprocessing as mp
import requests
import time
import yaml

from nltk.corpus import stopwords
from gensim.utils import simple_preprocess
from gensim import corpora
from gensim import models
from tqdm.contrib.concurrent import process_map
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
import numpy as np
import regex as re
import spacy


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
        logger.info(f"Loading first {num_docs} URLs from {data_path}")
        documents = []

        with open(data_path, "r", encoding="utf-8") as f:
            # Get first num_docs non-empty lines
            urls = [line.strip() for line in f if line.strip()][:num_docs]

        if not urls:
            raise ValueError(f"No URLs found in {data_path}")

        logger.info(f"Found {len(urls)} URLs to process")

        # SEC specific headers with email identification
        headers = {
            "User-Agent": "Sample Company Name AdminContact@company.com",
            "Host": "www.sec.gov",
        }

        # Fetch each document
        for i, url in enumerate(urls, 1):
            try:
                logger.info(f"Fetching document {i} from {url}")
                response = requests.get(url, headers=headers)
                response.raise_for_status()  # Raise exception for bad status codes
                documents.append(response.text)
                logger.info(f"Successfully loaded document {i}")
            except Exception as e:
                logger.error(f"Failed to fetch document from {url}: {str(e)}")
                # Continue with other documents even if one fails
                continue

        if not documents:
            raise ValueError("Failed to load any documents")

        logger.info(f"Successfully loaded {len(documents)} documents")
        return documents

    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
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
    - Removing SEC header metadata block
    - Removing document type markers and page markers
    - Removing extra spaces, emails, apostrophes, and non-alphabet characters
    - Converting text to lowercase

    Args:
        text (str): Input text to preprocess

    Returns:
        str: Preprocessed text
    """

    # Remove SEC header metadata block
    text = re.sub(
        r"Proc-Type:.*?</SEC-HEADER>", " ", text, flags=re.DOTALL | re.MULTILINE
    )

    # Remove document type markers and page markers
    text = re.sub(
        r"<DOCUMENT>|<TYPE>.*?</TYPE>|<SEQUENCE>.*?</SEQUENCE>|"
        r"<DESCRIPTION>.*?</DESCRIPTION>|<TEXT>|<PAGE>",
        " ",
        text,
        flags=re.DOTALL | re.MULTILINE,
    )

    # Standard cleaning steps
    text = re.sub("\s+", " ", text)  # Remove extra spaces
    text = re.sub("\S*@\S*\s?", "", text)  # Remove emails
    text = re.sub("'", "", text)  # Remove apostrophes
    text = re.sub("[^a-zA-Z]", " ", text)  # Remove non-alphabet characters
    text = text.lower()  # Convert to lowercase
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
        logger.warning(f"Stopwords file not found: {filepath}")
        return []
    except UnicodeDecodeError:
        logger.error(f"Encoding error reading stopwords file: {filepath}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error reading stopwords file {filepath}: {str(e)}")
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
    texts: List[str],
    nlp,
    allowed_postags: List[str] = ["NOUN", "ADJ", "VERB", "ADV"],
) -> List[List[str]]:
    """
    Perform lemmatization on tokenized texts using spaCy.

    Args:
        texts (List[str]): List of tokenized texts (each text is a list of
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
    texts_out = []
    for sent in texts:
        # Join the tokens back into text
        full_text = " ".join(sent)

        # Split into chunks if text is too long
        chunks = split_text_into_chunks(full_text)

        # Process each chunk separately
        processed_chunks = []
        for chunk in chunks:
            doc = nlp(chunk)
            processed_chunk = [
                token.lemma_ if token.lemma_ not in ["-PRON-"] else ""
                for token in doc
                if token.pos_ in allowed_postags
            ]
            processed_chunks.extend(processed_chunk)

        texts_out.append(processed_chunks)

    return texts_out


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
        logger.info(f"{func.__name__} took {execution_time:.2f} seconds to execute")
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

    # Get the current process ID
    # process_id = mp.current_process().name
    # start_time = time.time()

    # Apply preprocessing steps sequentially
    processed = basic_preprocessing(document)
    tokens = list(sent_to_words([processed]))[0]
    tokens = remove_stopwords([tokens], stop_words)[0]
    tokens = remove_words_less_than_length_three_characters([tokens])[0]
    tokens = lemmatization([tokens], nlp)[0]

    # end_time = time.time()
    # processing_time = end_time - start_time
    # logger.info(
    #     f"Process {process_id} finished processing in {processing_time:.2f} seconds"
    # )

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
    logger.info(f"Total vocabulary size: {len(vocabulary)}")

    # Print some vocabulary statistics
    logger.info("Vocabulary Sample (first 20 words):")
    logger.info(vocabulary[:20])

    # Analyze document lengths
    doc_lengths = [len(doc) for doc in bow_corpus]
    logger.info("Document length statistics:")
    logger.info(
        f"  - Average words per document: {sum(doc_lengths)/len(doc_lengths):.2f}"
    )
    logger.info(f"  - Max words in a document: {max(doc_lengths)}")
    logger.info(f"  - Min words in a document: {min(doc_lengths)}")

    # Compare BOW vs TF-IDF weights for first document
    if len(bow_corpus) > 0:
        logger.info("Comparing BOW vs TF-IDF weights for first document:")
        bow_doc = bow_corpus[0]
        tfidf_doc = tfidf_corpus[0]

        # Get top 10 words by frequency (BOW) and by TF-IDF weight
        bow_sorted = sorted(bow_doc, key=lambda x: x[1], reverse=True)[:num_words]
        tfidf_sorted = sorted(tfidf_doc, key=lambda x: x[1], reverse=True)[:num_words]

        logger.info(f"Top {num_words} words by frequency (BOW):")
        for word_id, freq in bow_sorted:
            logger.info(f"  - {id2word[word_id]}: {freq}")

        logger.info(f"Top {num_words} words by TF-IDF weight:")
        for word_id, weight in tfidf_sorted:
            logger.info(f"  - {id2word[word_id]}: {weight:.3f}")


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
    print("\nDebug: Vocabulary Check")
    print(f"Vocabulary size: {len(id2word)}")
    print("Sample of vocabulary (first 10 words):", list(id2word.values())[:10])

    print("\nDebug: Processed Texts Check")
    print(f"Number of documents: {len(texts)}")
    if texts:
        print(f"First document length: {len(texts[0])}")
        print("Sample of first document (first 10 words):", texts[0][:10])


@timer_decorator
def pre_processing_gensim(
    documents: List[str],
    config: Dict = None,
) -> Tuple[
    Dict[int, str],
    List[List[str]],
    List[List[Tuple[int, int]]],
    List[List[Tuple[int, float]]],
]:
    """
    Preprocess documents based on model type and configuration.

    Args:
        documents (List[str]): List of documents to process
        config (Dict): Configuration dictionary from YAML

    Returns:
        Tuple containing:
        - id2word (Dict[int, str]): Mapping of unique ids to words in the documents
        - texts (List[List[str]]): Preprocessed document texts
        - bow_corpus (List[List[Tuple[int, int]]]): Bag of words corpus
        - tfidf_corpus (List[List[Tuple[int, float]]]): TF-IDF corpus
    """

    start_time = datetime.now()
    num_cores = min(8, mp.cpu_count())
    logger.info(f"Starting preprocessing at {start_time}")
    logger.info(f"Number of CPU cores available: {num_cores}")
    logger.info(f"Number of documents to process: {len(documents)}")

    # Get base stopwords from NLTK
    stop_words = set(stopwords.words("english"))

    # Define paths to additional stopwords files
    stopwords_files = [
        "stopwords/financial_stopwords.txt",
        "stopwords/generic_stopwords.txt",
    ]

    # Add domain-specific stopwords from files
    for stopwords_file in stopwords_files:
        additional_stopwords = load_stopwords_from_file(stopwords_file)
        stop_words.update(additional_stopwords)
        logger.info(
            f"Added {len(additional_stopwords)} stopwords from {stopwords_file}"
        )

    logger.info(f"Total number of stopwords: {len(stop_words)}")

    # Load the spacy (en_core_web_sm): small English pipeline
    # trained on written web text (blogs, news, comments), that includes
    # vocabulary, syntax and entities.
    nlp = spacy.load(
        config.get("spacy_model", "en_core_web_sm"),
        disable=config.get("spacy_disabled", ["parser", "ner"]),
    )
    # nlp.max_length = 2000000  # 1158674

    if not documents or len(documents) == 0:
        raise ValueError("Empty document list provided")

    logger.info("Pre-processing the documents")

    # Prepare arguments for parallel processing
    process_args = [(doc, stop_words, nlp) for doc in documents]

    # Use ProcessPoolExecutor for CPU-bound preprocessing
    parallel_start = time.time()
    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        # processed_data = list(executor.map(process_document_chunk, process_args))
        processed_data = process_map(
            process_document_chunk,
            process_args,
            max_workers=num_cores,
            desc="Processing documents",
            chunksize=1,
        )
    parallel_end = time.time()
    logger.info(f"Completed parallel preprocessing of {len(processed_data)} documents")

    parallel_time = parallel_end - parallel_start
    logger.info(f"Parallel processing completed in {parallel_time:.2f} seconds")
    logger.info(
        f"Average time per document: {parallel_time/len(documents):.2f} seconds"
    )

    # 6. Build the bigram and trigram models
    bigram = models.Phrases(
        processed_data,
        min_count=5,
        threshold=200,
    )  # higher threshold fewer phrases.
    trigram = models.Phrases(
        bigram[processed_data],
        threshold=200,
    )

    # Faster way to get a sentence clubbed as a trigram/bigram
    bigram_mod = models.phrases.Phraser(bigram)
    trigram_mod = models.phrases.Phraser(trigram)

    # Form Bigrams
    data_bigrams = make_bigrams(processed_data, bigram_mod)
    data_bigrams_trigrams = make_trigrams(
        data_bigrams,
        trigram_mod,
        bigram_mod,
    )
    logger.info(f"Number of documents after tokenization: {len(data_bigrams_trigrams)}")

    # Create Dictionary - mapping of unique ids to words in the documents
    id2word = corpora.Dictionary(data_bigrams_trigrams)

    # Create Corpus - bag of words
    texts = data_bigrams_trigrams

    bow_corpus = [id2word.doc2bow(text) for text in texts]

    # Create Tf-IDF model
    tfidf = models.TfidfModel(bow_corpus)

    # Apply TF-IDF transformation to the corpus
    tfidf_corpus = tfidf[bow_corpus]

    if config.get("debug", "enabled") is True:
        check_vocabulary(id2word, texts)
        analyze_corpus(id2word, bow_corpus, tfidf_corpus)

    return (
        id2word,
        texts,
        bow_corpus,
        tfidf_corpus,
    )


@timer_decorator
def pre_processing_sklearn(
    documents: List[str],
    config: Dict = None,
) -> Tuple[List[str], TfidfVectorizer, np.ndarray]:
    """
    Preprocess documents for sklearn LDA model.

    Args:
        documents (List[str]): List of documents to process
        config (Dict): Configuration dictionary from YAML

    Returns:
        vectorizer (CountVectorizer): Fitted CountVectorizer object
        processed_texts (List[str]): List of preprocessed document texts
        tfidf_vectorizer (TfidfVectorizer): Fitted TfidfVectorizer object
        tfidf_matrix (np.ndarray): TF-IDF matrix of the documents
    """
    start_time = datetime.now()
    num_cores = min(8, mp.cpu_count())
    logger.info(f"Starting preprocessing at {start_time}")
    logger.info(f"Number of CPU cores available: {num_cores}")
    logger.info(f"Number of documents to process: {len(documents)}")

    # Get base stopwords from NLTK
    stop_words = set(stopwords.words("english"))

    # Add domain-specific stopwords
    stopwords_files = [
        "stopwords/financial_stopwords.txt",
        "stopwords/generic_stopwords.txt",
    ]
    for stopwords_file in stopwords_files:
        additional_stopwords = load_stopwords_from_file(stopwords_file)
        stop_words.update(additional_stopwords)
        logger.info(
            f"Added {len(additional_stopwords)} stopwords from {stopwords_file}"
        )

    logger.info(f"Total number of stopwords: {len(stop_words)}")

    # Load spacy for lemmatization
    nlp = spacy.load(
        config.get("spacy_model", "en_core_web_sm"),
        disable=config.get("spacy_disabled", ["parser", "ner"]),
    )

    if not documents or len(documents) == 0:
        raise ValueError("Empty document list provided")

    # Parallel document processing
    process_args = [(doc, stop_words, nlp) for doc in documents]
    parallel_start = time.time()
    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        processed_data = process_map(
            process_document_chunk,
            process_args,
            max_workers=num_cores,
            desc="Processing documents",
            chunksize=1,
        )
    parallel_end = time.time()

    # Join tokens back into text for sklearn vectorizers
    processed_texts = [" ".join(doc) for doc in processed_data]

    # Create and fit CountVectorizer
    # ngram_range=(1,3) captures unigrams, bigrams, and trigrams
    count_vectorizer = CountVectorizer(
        min_df=0.01,  # 1% of the documents should contain the word
        max_df=0.95,  # 95% of the documents should contain the word
        # ngram_range=(1, 3),  # Include unigrams, bigrams, and trigrams
        stop_words=list(stop_words),  # Pass our combined stopwords
        # token_pattern=r"(?u)\b\w+\b",  # Match any word character
        # max_features=config.get("max_features", None),  # Optional vocab size limit
    )

    # Create document-term matrix and fit the vectorizer
    bow_matrix = count_vectorizer.fit_transform(processed_texts)

    # Create TF-IDF transformer and transform bow_matrix
    tfidf_vectorizer = TfidfVectorizer(
        vocabulary=count_vectorizer.vocabulary_,  # Now vocabulary_ exists after fitting
        use_idf=True,  # Use IDF weights for scaling
        smooth_idf=True,  # Smooth the IDF weights to avoid zero values
        sublinear_tf=True,  # Apply sublinear scaling to term frequencies
    )
    tfidf_matrix = tfidf_vectorizer.fit_transform(processed_texts)

    if config.get("debug", "enabled") is True:
        # Debug information about the matrices
        logger.info(f"Vocabulary size: {len(count_vectorizer.vocabulary_)}")
        logger.info(f"Document-term matrix shape: {bow_matrix.shape}")
        logger.info(f"TF-IDF matrix shape: {tfidf_matrix.shape}")

    del count_vectorizer  # Free up memory

    return (
        processed_texts,
        tfidf_vectorizer,
        tfidf_matrix,
    )


if __name__ == "__main__":
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
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    documents = load_data(args.data)
    id2word, texts, bow_corpus, tfidf_corpus = pre_processing_gensim(
        documents,
        config=config["preprocessing"],
    )
