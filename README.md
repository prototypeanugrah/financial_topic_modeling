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
uv run main.py
  --config config.yaml
  --data <path_to_documents>
  -n <num_docs>
  -m <model_type>
```

Parameters:
- `--config`: Path to configuration file (default: config.yaml)
- `--data`: Path to input text documents (required)
- `-n/--num_docs`: Number of documents to process (0 for all documents)
- `-m/--model_type`: LDA implementation to use ('gensim' or 'sklearn')

Example:
```bash
uv run main.py --config config.yaml --data documents.txt -n 10 -m sklean
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
- `perplexity_plot.png`: Topic optimization results

## Configuration

The `config.yaml` file allows customization of:
- Preprocessing parameters
- Model parameters
- Output settings
- Logging configuration

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

