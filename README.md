# Financial Topic Modeling

A Python-based implementation of Latent Dirichlet Allocation (LDA) topic modeling for financial documents, supporting both Gensim and scikit-learn implementations.

## Overview

This project provides a robust framework for topic modeling on financial documents, with the following key features:

- Support for both Gensim and scikit-learn LDA implementations
- Parallel processing for efficient document preprocessing
- Comprehensive text preprocessing pipeline including:
  - Stopword removal
  - Lemmatization
  - Bigram and trigram detection
  - TF-IDF transformation
- Topic optimization capabilities
- Performance metrics calculation
- Visualization tools for topics and word frequencies

## Requirements

- Python 3.x
- Virtual environment (recommended)
- uv (recommended for faster package installation)

## Installation

1. Clone the repository:
```bash
git clone git@github.com:prototypeanugrah/financial_topic_modeling.git
cd financial_topic_modeling
```

2. Install uv (if not already installed):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

3. Create and activate a virtual environment:
```bash
uv venv --python 3.11
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

4. Install dependencies using uv:
```bash
uv pip install -r requirements.txt
```

5. Download required NLTK data:
```python
import nltk
nltk.download('stopwords')
```

6. Install spaCy model:
```bash
python -m spacy download en_core_web_sm
```

## Project Structure

```
financial_topic_modeling/
├── main.py                 # Main script for running the pipeline
├── data_preprocessing.py   # Document preprocessing utilities
├── lda_model_gensim.py     # Gensim LDA implementation
├── lda_model_sklearn.py    # scikit-learn LDA implementation
├── visualizing_wordcloud.py # Topic visualization utilities
├── config.yaml            # Configuration file
├── stopwords/             # Custom stopword lists
│   ├── financial_stopwords.txt
│   └── generic_stopwords.txt
└── requirements.txt       # Project dependencies
```

## Usage

The main script can be run with the following command:

```bash
uv run main.py \
  --config config.yaml \
  -n <num_docs> \
  [-k <num_topics>] \
  [-nc <num_cores>] \
  [-b <batch_size>] \
  [-t <test_perc>]
```

Parameters:
- `--config`: Path to configuration file (default: `config.yaml`)
- `-n/--num_docs`: Number of documents to process (0 for all documents, required).
- `-k/--num_topics`: Number of topics. If not provided, it's determined automatically via optimization.
- `-nc/--num_cores`: Number of CPU cores for parallel processing (default: 16).
- `-b/--batch_size`: Number of documents to process in each batch (default: 100).
- `-t/--test_perc`: Percentage of documents for the test set (default: 0.1, e.g., 0.1 for 10%).

**Note:** Input documents are expected to be located in a directory named `raw_data_files` in the project root.

Example:
```bash
uv run main.py --config config.yaml -n 1000 -k 20 -nc 16 -b 200 -t 0.1
```

## Features

### Document Preprocessing
- Parallel processing for efficient document handling
- Custom stopword removal for financial domain
- Lemmatization using spaCy
- Bigram and trigram detection
- TF-IDF transformation

### Topic Modeling
- Support for both Gensim and scikit-learn implementations
- Topic optimization using perplexity scores
- Performance metrics calculation
- Document-topic assignment analysis

### Visualization
- Word cloud generation for topics
- Word frequency analysis
- Perplexity score plotting

## Output

The pipeline generates several output files:
- `topics.txt`: Top words for each topic
- `document_topics.txt`: Topic distribution for each document
- `metrics.yaml`: Model performance metrics
- `wordcloud.png`: Visual representation of topics
- `word_frequency_plot.png`: Analysis of word frequencies
- `perplexity_plot_test.png`: Topic optimization results (perplexity vs. number of topics for the test set)
- `topic_perplexity_scores.txt`: Perplexity scores for each number of topics evaluated during optimization.

## Configuration

The `config.yaml` file allows customization of:
- Preprocessing parameters
- Model parameters
- Output settings
- Logging configuration

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

