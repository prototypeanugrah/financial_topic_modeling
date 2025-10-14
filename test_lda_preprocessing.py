import json
import logging
import os
import re
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd
import spacy
from gensim import corpora, models
from gensim.utils import simple_preprocess
from nltk.corpus import stopwords
from tqdm import tqdm

import utils
from utils import PairRecord

logger = logging.getLogger(__name__)

# Global worker variables for efficient spaCy initialization
_NLP = None
_STOP = None
_ALLOWED_POS = None


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
    # Standard cleaning steps
    text = re.sub(r"\S*@\S*\s?", "", text)  # Remove emails
    text = re.sub(r"'", "", text)  # Remove apostrophes
    text = re.sub(r"&nbsp;", " ", text)  # Remove &nbsp;

    text = re.sub(r"[^a-zA-Z]", " ", text)  # Remove non-alphabet characters

    text = text.lower()  # Convert to lowercase
    text = re.sub(r"\s+", " ", text)  # Remove extra spaces

    # remove any words that contain digits/numbers
    text = re.sub(r"\b\d+\b", "", text)

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


def load_stop_words(config: Dict) -> set:
    """
    Load stop words from a folder of text files.

    Args:
        config: Configuration dictionary from YAML

    Returns:
        Set of stopwords
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


def init_worker(spacy_model, disabled, allowed_pos, stop_words: List[str]):
    """
    Initialize spaCy model once per worker process

    Args:
        spacy_model: Name of the spaCy model to load
        disabled: List of spaCy components to disable
        allowed_pos: List of POS tags to allow
        stop_words: List of stopwords
    """
    global _NLP, _STOP, _ALLOWED_POS
    try:
        _NLP = spacy.load(spacy_model, disable=disabled)
        _ALLOWED_POS = set(allowed_pos)
        _STOP = stop_words
    except Exception as e:
        logger.error(f"Failed to initialize worker: {e}")
        raise


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


def process_document(
    document: str,
) -> List[str]:
    """
    Process a single document through all preprocessing steps.

    Args:
        document: Single document to process

    Returns:
        List[str]: Processed tokens from the document

    Note:
    1. Basic preprocessing
    2. Tokenization
    3. Remove stopwords
    4. Remove words with less than 3 characters
    5. Lemmatization (using worker's spaCy model)
    """
    global _NLP, _ALLOWED_POS, _STOP

    # Apply preprocessing steps sequentially
    text = basic_preprocessing(document)
    tokens = list(sent_to_words([text]))[0]
    tokens = remove_stopwords([tokens], list(_STOP))[0]
    tokens = remove_words_less_than_length_three_characters([tokens])[0]

    # Lemmatization using worker's spaCy model
    if _NLP is not None and _ALLOWED_POS is not None:
        try:
            from spacy.tokens import Doc

            doc = Doc(_NLP.vocab, words=tokens)
            for pipe_name in ["tagger", "attribute_ruler", "lemmatizer"]:
                if pipe_name in _NLP.pipe_names:
                    pipe = _NLP.get_pipe(pipe_name)
                    doc = pipe(doc)
            tokens = [token.lemma_ for token in doc if token.pos_ in _ALLOWED_POS]

        except Exception as e:
            logger.warning("Lemmatization failed: %s", str(e))
            # Fall back to original tokens

    return tokens


def check_document_lengths(texts: List[List[str]], report_type: str) -> bool:
    """
    Check document lengths in the text data.

    Args:
        texts: List of preprocessed document texts
        report_type: Processing mode (ipo/analyst)

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
            report_type,
            min_length,
            max_length,
            avg_length,
            median_length,
        )


def run_batch(
    docs: List[str],
    config: Dict,
    stop_words: List[str],
    num_workers: int,
) -> List[List[str]]:
    """
    Run a batch of documents through the preprocessing pipeline using a
    worker pool.

    Args:
        docs: List of documents to process
        config: Configuration dictionary from YAML
        stop_words: List of stopwords
        num_workers: Number of workers to use

    Returns:
        List[List[str]]: List of processed documents
    """
    try:
        with ProcessPoolExecutor(
            max_workers=num_workers,
            initializer=init_worker,
            initargs=(
                config["preprocessing"]["spacy_model"],
                config["preprocessing"]["spacy_disabled"],
                config["preprocessing"]["allowed_postags"],
                stop_words,
            ),
        ) as ex:
            return list(
                ex.map(process_document, docs),
            )
    except Exception as e:
        logger.error(f"Error in run_batch: {e}")
        logger.error(f"Number of workers: {num_workers}, Number of docs: {len(docs)}")
        raise


def process_all_batches(
    doc_generator: Iterator[List[Tuple[str, str]]],
    config: Dict,
    stop_words: List[str],
    num_workers: int,
    dataset_label: str,
    mode: str,
) -> Tuple[
    List[List[str]],
    List[Tuple[str, str]],
    List[str],
]:
    """
    Run preprocessing across every batch yielded by a generator while keeping the
    original ordering of documents intact.

    Args:
        doc_generator: Generator of batches of documents
        config: Configuration dictionary from YAML
        stop_words: List of stopwords
        num_workers: Number of workers to use
        dataset_label: Label for the dataset
        mode: Processing mode (train/test)
    Returns:
        Tuple containing:
        - processed_docs: List of processed documents
        - order_dict: Dictionary of document order
        - zero_token_docs: List of documents that resulted in zero tokens
        - doc_paths: Ordered list of source file paths aligned with processed_docs
    """
    processed_docs: List[List[str]] = []
    order_dict: Dict[str, str] = {}
    zero_token_docs: List[Tuple[str, str]] = []
    doc_paths: List[str] = []

    for batch_index, batch_docs in enumerate(
        tqdm(
            doc_generator,
            desc=f"Processing {dataset_label} batches ({mode})",
            unit="batch",
        ),
        start=1,
    ):
        if not batch_docs:
            raise ValueError(
                f"Encountered empty {dataset_label} batch at index {batch_index}. "
                "Please inspect the upstream file list or adjust preprocessing to handle blanks."
            )

        batch_texts = [doc for doc, _ in batch_docs]
        batch_paths = [path for _, path in batch_docs]

        processed_batch = run_batch(batch_texts, config, stop_words, num_workers)
        processed_docs.extend(processed_batch)  # Type: List[List[str]]
        for tokens, doc_path in zip(processed_batch, batch_paths):
            doc_index = len(order_dict) + 1
            doc_label = f"{dataset_label}_{doc_index}"
            order_dict[f"doc{doc_index}"] = doc_label
            doc_paths.append(doc_path)  # Type: List[str]

            if not tokens:
                zero_token_docs.append(
                    (doc_label, doc_path)
                )  # Type: List[Tuple[str, str]]
                logger.warning(
                    "Preprocessed %s document %s yielded 0 tokens after processing",
                    doc_label,
                    doc_path,
                )

    return processed_docs, zero_token_docs, doc_paths


def check_empty_docs(
    ipo_zero_token_docs: List[Tuple[str, str]],
    analyst_zero_token_docs: List[Tuple[str, str]],
    ipo_docs: List[List[str]],
    analyst_docs: List[List[str]],
    ipo_paths: List[str],
    analyst_paths: List[str],
    output_dir: Path,
    mode: str,
    *,
    return_metadata: bool = False,
) -> Tuple[List[List[str]], List[List[str]]]:
    if ipo_zero_token_docs:
        logger.warning(
            "Detected %d IPO documents with zero tokens after preprocessing",
            len(ipo_zero_token_docs),
        )
    if analyst_zero_token_docs:
        logger.warning(
            "Detected %d analyst documents with zero tokens after preprocessing",
            len(analyst_zero_token_docs),
        )

    ipo_order_dict: Dict[str, str] = {
        f"doc{idx + 1}": f"ipo_{idx + 1}" for idx in range(len(ipo_docs))
    }
    analyst_order_dict: Dict[str, str] = {
        f"doc{idx + 1}": f"analyst_{idx + 1}" for idx in range(len(analyst_docs))
    }

    ipo_zero_indices = utils.extract_zero_indices(ipo_zero_token_docs)
    analyst_zero_indices = utils.extract_zero_indices(analyst_zero_token_docs)
    indices_to_remove = ipo_zero_indices.union(analyst_zero_indices)

    removed_pairs_info: List[Dict[str, Any]] = []

    filtered_ipo_paths: List[str] = []
    filtered_analyst_paths: List[str] = []
    kept_pair_ids: List[int] = []

    if indices_to_remove:
        logger.warning(
            "Removing %d IPO/analyst document pairs due to zero-token output",
            len(indices_to_remove),
        )
        filtered_ipo_docs: List[List[str]] = []
        filtered_analyst_docs: List[List[str]] = []
        filtered_ipo_order_dict: Dict[str, str] = {}
        filtered_analyst_order_dict: Dict[str, str] = {}

        for pair_index, (
            ipo_doc,
            analyst_doc,
            ipo_path,
            analyst_path,
        ) in enumerate(
            zip(ipo_docs, analyst_docs, ipo_paths, analyst_paths),
            start=1,
        ):
            if pair_index in indices_to_remove:
                reasons = []
                if pair_index in ipo_zero_indices:
                    reasons.append("ipo_zero_tokens")
                if pair_index in analyst_zero_indices:
                    reasons.append("analyst_zero_tokens")

                removed_pairs_info.append(
                    {
                        "pair_index": pair_index,
                        "reasons": reasons,
                        "ipo": {"label": f"ipo_{pair_index}", "path": ipo_path},
                        "analyst": {
                            "label": f"analyst_{pair_index}",
                            "path": analyst_path,
                        },
                    }
                )
                continue

            new_index = len(filtered_ipo_docs) + 1
            filtered_ipo_docs.append(ipo_doc)
            filtered_analyst_docs.append(analyst_doc)
            filtered_ipo_paths.append(ipo_path)
            filtered_analyst_paths.append(analyst_path)
            kept_pair_ids.append(pair_index)
            # Preserve the original pair index so labels still map back to the
            # source documents after zero-token filtering.
            filtered_ipo_order_dict[f"doc{new_index}"] = f"ipo_{pair_index}"
            filtered_analyst_order_dict[f"doc{new_index}"] = f"analyst_{pair_index}"

        ipo_docs = filtered_ipo_docs
        analyst_docs = filtered_analyst_docs
        ipo_order_dict = filtered_ipo_order_dict
        analyst_order_dict = filtered_analyst_order_dict
    else:
        removed_pairs_info = []
        filtered_ipo_paths = list(ipo_paths)
        filtered_analyst_paths = list(analyst_paths)
        kept_pair_ids = list(range(1, len(ipo_docs) + 1))

    if len(ipo_docs) != len(analyst_docs):
        raise ValueError(
            "Mismatched document counts after preprocessing: "
            f"{len(ipo_docs)} IPO documents vs {len(analyst_docs)} analyst documents.",
        )

    combined_order_dict: Dict[str, str] = {}
    for idx, label in enumerate(ipo_order_dict.values(), start=1):
        combined_order_dict[f"doc{idx}"] = label

    offset = len(ipo_order_dict)
    for idx, label in enumerate(analyst_order_dict.values(), start=1):
        combined_order_dict[f"doc{offset + idx}"] = label

    zero_token_report = {
        "ipo": [
            {"doc_label": label, "path": path} for label, path in ipo_zero_token_docs
        ],
        "analyst": [
            {"doc_label": label, "path": path}
            for label, path in analyst_zero_token_docs
        ],
        "removed_pairs": removed_pairs_info,
    }
    with open(
        str(output_dir / f"preprocessed/{mode}_zero_token_documents.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(zero_token_report, f, indent=2)

    # Save combined order dictionary
    with open(
        str(output_dir / f"preprocessed/{mode}_combined_order_dict.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(combined_order_dict, f, indent=2)

    if return_metadata:
        metadata = {
            "kept_pair_ids": kept_pair_ids,
            "filtered_ipo_paths": filtered_ipo_paths,
            "filtered_analyst_paths": filtered_analyst_paths,
            "removed_pairs": removed_pairs_info,
        }
        return ipo_docs, analyst_docs, metadata

    return ipo_docs, analyst_docs


@dataclass(frozen=True)
class PreprocessedPair:
    pair_id: int
    ipo_tokens: List[str]
    analyst_tokens: List[str]
    ipo_path: str
    analyst_path: str


@dataclass
class PreprocessedPairsResult:
    pairs: List[PreprocessedPair]
    removed_pairs: List[Dict[str, Any]]
    order_metadata_path: Path
    zero_token_report_path: Path


def train_ngram_models(
    texts: List[List[str]],
    *,
    min_count: int = 5,
    threshold: int = 100,
) -> Tuple[models.phrases.Phraser, models.phrases.Phraser]:
    """
    Fit bigram and trigram models on the provided tokenized texts.
    """
    if not texts:
        raise ValueError("Cannot train n-gram models on an empty text collection.")

    bigram = models.Phrases(
        texts,
        min_count=min_count,
        threshold=threshold,
    )
    bigram_mod = models.phrases.Phraser(bigram)

    trigram = models.Phrases(
        bigram[texts],
        threshold=threshold,
    )
    trigram_mod = models.phrases.Phraser(trigram)

    return bigram_mod, trigram_mod


def apply_ngram_models(
    texts: List[List[str]],
    bigram_mod: models.phrases.Phraser,
    trigram_mod: models.phrases.Phraser,
) -> List[List[str]]:
    """
    Apply fitted bigram and trigram models to a collection of texts.
    """
    if not texts:
        return []
    texts_with_bigrams = make_bigrams(texts, bigram_mod)
    texts_with_trigrams = make_trigrams(texts_with_bigrams, trigram_mod, bigram_mod)
    return texts_with_trigrams


def preprocess_pairs_for_cv(
    pair_records: List[PairRecord],
    config: Dict,
    *,
    batch_size: int,
    num_cores: int,
    output_dir: Path,
    mode: str = "cv",
) -> PreprocessedPairsResult:
    """
    Preprocess IPO/analyst document pairs once for cross-validation.

    Returns paired token lists (no n-grams applied) alongside metadata so that
    CV folds can operate on in-memory representations without re-running spaCy.
    """
    if not pair_records:
        raise ValueError("No pair records provided for preprocessing.")

    preprocessed_dir = output_dir / "preprocessed"
    preprocessed_dir.mkdir(parents=True, exist_ok=True)

    ipo_paths = [record.ipo_path for record in pair_records]
    analyst_paths = [record.analyst_path for record in pair_records]

    ipo_generator = utils.BatchStream(ipo_paths, batch_size)
    analyst_generator = utils.BatchStream(analyst_paths, batch_size)

    stop_words = list(load_stop_words(config))

    ipo_docs, ipo_zero_token_docs, ipo_paths_processed = process_all_batches(
        doc_generator=ipo_generator,
        config=config,
        stop_words=stop_words,
        num_workers=num_cores,
        dataset_label="ipo",
        mode=mode,
    )
    check_document_lengths(texts=ipo_docs, report_type="ipo")

    analyst_docs, analyst_zero_token_docs, analyst_paths_processed = (
        process_all_batches(
            doc_generator=analyst_generator,
            config=config,
            stop_words=stop_words,
            num_workers=num_cores,
            dataset_label="analyst",
            mode=mode,
        )
    )
    check_document_lengths(texts=analyst_docs, report_type="analyst")

    (
        ipo_docs,
        analyst_docs,
        metadata,
    ) = check_empty_docs(
        ipo_zero_token_docs=ipo_zero_token_docs,
        analyst_zero_token_docs=analyst_zero_token_docs,
        ipo_docs=ipo_docs,
        analyst_docs=analyst_docs,
        ipo_paths=ipo_paths_processed,
        analyst_paths=analyst_paths_processed,
        output_dir=output_dir,
        mode=mode,
        return_metadata=True,
    )

    kept_pair_ids = metadata["kept_pair_ids"]
    filtered_ipo_paths = metadata["filtered_ipo_paths"]
    filtered_analyst_paths = metadata["filtered_analyst_paths"]
    removed_pairs = metadata["removed_pairs"]

    if len(ipo_docs) != len(analyst_docs):
        raise ValueError(
            "IPO and analyst document counts diverged after zero-token filtering."
        )

    if len(kept_pair_ids) != len(ipo_docs):
        raise ValueError("Mismatch between kept pair metadata and processed documents.")

    pairs: List[PreprocessedPair] = []
    for (
        pair_id,
        ipo_tokens,
        analyst_tokens,
        ipo_path,
        analyst_path,
    ) in zip(
        kept_pair_ids,
        ipo_docs,
        analyst_docs,
        filtered_ipo_paths,
        filtered_analyst_paths,
    ):
        pairs.append(
            PreprocessedPair(
                pair_id=pair_id,
                ipo_tokens=ipo_tokens,
                analyst_tokens=analyst_tokens,
                ipo_path=ipo_path,
                analyst_path=analyst_path,
            )
        )

    logger.info(
        "Preprocessed %d IPO/analyst pairs for cross-validation (removed %d pairs)",
        len(pairs),
        len(removed_pairs),
    )

    order_metadata_path = preprocessed_dir / f"{mode}_combined_order_dict.json"
    zero_token_report_path = preprocessed_dir / f"{mode}_zero_token_documents.json"

    return PreprocessedPairsResult(
        pairs=pairs,
        removed_pairs=removed_pairs,
        order_metadata_path=order_metadata_path,
        zero_token_report_path=zero_token_report_path,
    )


def apply_bigrams_and_trigrams(
    texts: List[List[str]],
    output_dir: Path,
    mode: str,
    *,
    train_mode: bool,
    phraser_source_mode: Optional[str] = None,
) -> List[List[str]]:
    """
    Apply bigrams and trigrams to the texts.
    """

    if train_mode:
        # Build bigram and trigram models on ALL texts (not per batch)
        bigram = models.Phrases(
            texts,
            min_count=5,  # only include phrases that appear at least 5 times in the corpus
            threshold=100,  # higher threshold fewer phrases.
        )
        trigram = models.Phrases(
            bigram[texts],
            threshold=100,  # higher threshold fewer phrases.
        )

        # Faster way to get a sentence clubbed as a trigram/bigram
        bigram_mod = models.phrases.Phraser(bigram)
        trigram_mod = models.phrases.Phraser(trigram)

        # Persist phrasers for reuse on test
        bigram_mod.save(str(output_dir / f"preprocessed/{mode}_bigram.phr"))
        trigram_mod.save(str(output_dir / f"preprocessed/{mode}_trigram.phr"))

        # Apply bigrams and trigrams to all texts
        texts = make_bigrams(texts, bigram_mod)
        texts = make_trigrams(texts, trigram_mod, bigram_mod)  # Type: List[List[str]]

    else:
        source_mode = phraser_source_mode or mode
        bigram_path = output_dir / f"preprocessed/{source_mode}_bigram.phr"
        trigram_path = output_dir / f"preprocessed/{source_mode}_trigram.phr"
        if not bigram_path.exists() or not trigram_path.exists():
            raise FileNotFoundError(
                f"N-gram phrasers for '{source_mode}' not found. "
                "Ensure the corresponding training split has been generated first."
            )
        bigram_mod = models.phrases.Phraser.load(str(bigram_path))
        trigram_mod = models.phrases.Phraser.load(str(trigram_path))
        texts = make_bigrams(texts, bigram_mod)
        texts = make_trigrams(texts, trigram_mod, bigram_mod)  # Type: List[List[str]]

    return texts


def create_texts_with_preprocessing(
    ipo_all_docs_generator: Iterator[List[Tuple[str, str]]],
    analyst_all_docs_generator: Iterator[List[Tuple[str, str]]],
    config: Dict,
    num_cores: int,
    output_dir: Path,
    mode: str,
) -> List[List[str]]:
    """
    Shared LDA preprocessing pipeline.

    Args:
        ipo_all_docs_generator: Generator of IPO documents batches
        analyst_all_docs_generator: Generator of analyst documents batches
        config: Configuration dictionary from YAML
        num_cores: Number of cores to use
        output_dir: Path object pointing to the output directory
        mode: Processing mode (train/test)

    Returns:
        List[List[str]]: List of processed documents
    """
    stop_words = list(load_stop_words(config))

    ipo_docs, ipo_zero_token_docs, ipo_paths = process_all_batches(
        doc_generator=ipo_all_docs_generator,
        config=config,
        stop_words=stop_words,
        dataset_label="ipo",
        num_workers=num_cores,
        mode=mode,
    )
    check_document_lengths(texts=ipo_docs, report_type="ipo")

    analyst_docs, analyst_zero_token_docs, analyst_paths = process_all_batches(
        doc_generator=analyst_all_docs_generator,
        config=config,
        stop_words=stop_words,
        dataset_label="analyst",
        num_workers=num_cores,
        mode=mode,
    )
    check_document_lengths(texts=analyst_docs, report_type="analyst")

    ipo_docs, analyst_docs = check_empty_docs(
        ipo_zero_token_docs=ipo_zero_token_docs,
        analyst_zero_token_docs=analyst_zero_token_docs,
        ipo_docs=ipo_docs,
        analyst_docs=analyst_docs,
        ipo_paths=ipo_paths,
        analyst_paths=analyst_paths,
        output_dir=output_dir,
        mode=mode,
    )

    return ipo_docs + analyst_docs


def create_dictionary(
    texts: List[List[str]],
    output_dir: Path,
    mode: str,
    filter_params: Optional[Dict[str, Any]] = None,
) -> Tuple[corpora.Dictionary, List[List[Tuple[int, int]]]]:
    """
    Create a dictionary from the texts.
    """
    dictionary = corpora.Dictionary(texts)  # Type: corpora.Dictionary
    initial_len = len(dictionary)
    logger.info(
        "[%s] Dictionary size before filtering: %d tokens",
        mode,
        initial_len,
    )

    if filter_params:
        no_below = filter_params.get("no_below")
        no_above = filter_params.get("no_above")
        keep_n = filter_params.get("keep_n")

        dictionary.filter_extremes(
            no_below=no_below,
            no_above=no_above,
            keep_n=keep_n,
        )

        if len(dictionary) == 0:
            logger.warning(
                "[%s] Dictionary emptied by filter_extremes (params=%s). "
                "Rebuilding dictionary without filtering for this split.",
                mode,
                filter_params,
            )
            dictionary = corpora.Dictionary(texts)

    logger.info(
        "[%s] Dictionary size after filtering: %d tokens",
        mode,
        len(dictionary),
    )

    # Create BOW corpus
    bow_corpus = [
        dictionary.doc2bow(text) for text in texts
    ]  # Type: List[List[Tuple[int, int]]]

    return dictionary, bow_corpus


def corpus_filtering_from_dictionary(
    dic: corpora.Dictionary,
    texts: List[List[str]],
) -> List[List[Tuple[int, int]]]:
    """
    Filter the test corpus based on the train dictionary and tfidf model.

    Args:
        dic (corpora.Dictionary): Gensim shared/train dictionary mapping word IDs to words for the train corpus
        texts (List[List[str]]): List of tokenized documents for the test corpus

    Returns:
        List[List[Tuple[int, int]]]: Filtered bow corpus
        Example:
        Shared/train dictionary: {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4}
        Test texts: [['a', 'b', 'c'], ['b', 'c'], ['a', 'b', 'b', 'e']]
        Filtered bow corpus: [
            [(0, 1), (1, 1), (2, 1)],
            [(1, 1), (2, 1)],
            [(0, 1), (1, 2), (4, 1)],
        ]
    """
    bow_corpus = [dic.doc2bow(text) for text in texts]

    return bow_corpus


def prep_data_train(
    ipo_generator: Iterator[List[Tuple[str, str]]],
    analyst_generator: Iterator[List[Tuple[str, str]]],
    config: Dict,
    num_cores: int,
    output_dir: Path,
    mode: str,
):
    texts = create_texts_with_preprocessing(
        ipo_all_docs_generator=ipo_generator,
        analyst_all_docs_generator=analyst_generator,
        config=config,
        num_cores=num_cores,
        output_dir=output_dir,
        mode=mode,
    )

    texts = apply_bigrams_and_trigrams(
        texts=texts,
        output_dir=output_dir,
        mode=mode,
        train_mode=True,
    )

    dictionary, bow_corpus = create_dictionary(
        texts=texts,
        output_dir=output_dir,
        mode=mode,
        filter_params=config.get("preprocessing", {}).get("filter_extremes"),
    )

    # Save BOW corpus using MmCorpus.serialize
    corpora.MmCorpus.serialize(
        str(output_dir / f"preprocessed/{mode}_bow_corpus.mm"),
        bow_corpus,
    )

    # Save dictionary
    dictionary.save(str(output_dir / f"preprocessed/{mode}_dictionary.id2word"))

    # Save all texts
    with open(
        str(output_dir / f"preprocessed/{mode}_texts.txt"),
        "w",
        encoding="utf-8",
    ) as f:
        for text in texts:
            f.write(" ".join(text) + "\n")

    return texts, dictionary, bow_corpus


def prep_data_non_train(
    ipo_generator: Iterator[List[Tuple[str, str]]],
    analyst_generator: Iterator[List[Tuple[str, str]]],
    train_dictionary: corpora.Dictionary,
    config: Dict,
    num_cores: int,
    output_dir: Path,
    mode: str,
    phraser_source_mode: Optional[str] = None,
):
    texts = create_texts_with_preprocessing(
        ipo_all_docs_generator=ipo_generator,
        analyst_all_docs_generator=analyst_generator,
        config=config,
        num_cores=num_cores,
        output_dir=output_dir,
        mode=mode,
    )

    texts = apply_bigrams_and_trigrams(
        texts=texts,
        output_dir=output_dir,
        mode=mode,
        train_mode=False,
        phraser_source_mode=phraser_source_mode or "train",
    )

    corpus = corpus_filtering_from_dictionary(
        dic=train_dictionary,
        texts=texts,
    )

    # Save BOW corpus using MmCorpus.serialize
    corpora.MmCorpus.serialize(
        str(output_dir / f"preprocessed/{mode}_bow_corpus.mm"),
        corpus,
    )

    # Save all texts
    with open(
        str(output_dir / f"preprocessed/{mode}_texts.txt"),
        "w",
        encoding="utf-8",
    ) as f:
        for text in texts:
            f.write(" ".join(text) + "\n")

    return texts, corpus


if __name__ == "__main__":
    config = utils.load_config("config.yaml")
    batch_size = config["preprocessing"]["batch_size"]
    num_docs = config["preprocessing"]["num_docs"]
    num_cores = config["preprocessing"]["num_cores_preprocessing"]
    test_perc = config["preprocessing"]["test_perc"]

    # Create preprocessed subdirectory
    output_dir = utils.setup_output_directory(config)
    preprocessed_dir = output_dir / "preprocessed"
    preprocessed_dir.mkdir(exist_ok=True)

    # Load CSV and extract file paths for both report types
    df = pd.read_csv("data/final_analyst_reports_for_latest_s1_filings_extracted.csv")

    # Extract file paths for both report types
    ipo_file_paths = df["s1_path"].dropna().tolist()
    analyst_file_paths = df["analyst_report_path"].dropna().tolist()

    print("===============================================")
    print("Loading Data Started")
    print("===============================================")

    # Load all documents for shared dictionary creation
    train_ipo_generator, test_ipo_generator = utils.load_splits_from_paths(
        file_paths=ipo_file_paths,
        batch_size=batch_size,
        num_docs=num_docs,
        test_perc=test_perc,  # Load all as training for dictionary
        report_type="ipo",
    )

    train_analyst_generator, test_analyst_generator = utils.load_splits_from_paths(
        file_paths=analyst_file_paths,
        batch_size=batch_size,
        num_docs=num_docs,
        test_perc=test_perc,  # Load all as training for dictionary
        report_type="analyst",
    )

    train_texts, train_dictionary, train_bow_corpus = prep_data_train(
        ipo_generator=train_ipo_generator,
        analyst_generator=train_analyst_generator,
        config=config,
        num_cores=num_cores,
        output_dir=output_dir,
        mode="train",
    )

    test_texts, test_corpus = prep_data_non_train(
        ipo_generator=test_ipo_generator,
        analyst_generator=test_analyst_generator,
        train_dictionary=train_dictionary,
        config=config,
        num_cores=num_cores,
        output_dir=output_dir,
        mode="test",
        phraser_source_mode="train",
    )

    # Once all artifacts have been created, remove these files from the output directory
    # delete all files ending with .phr
    for file in preprocessed_dir.glob("*.phr"):
        file.unlink()
