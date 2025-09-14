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

3. Install dependencies using uv (this will automatically create a virtual environment):
```bash
uv sync
```

4. Download required NLTK data:
```bash
uv run python -c "import nltk; nltk.download('stopwords')"
```

5. Install spaCy model:
```bash
uv run python -m spacy download en_core_web_sm
```

## Project Structure

```
financial_topic_modeling/
├── main.py                         # Main script for running the pipeline
├── data_preprocessing/             # Preprocessing module
│   ├── lda_preprocessing.py        # Core document preprocessing utilities
│   ├── enhanced_ipo_parser.py      # Advanced IPO document parsing
│   └── batch_ipo_processor.py      # Batch processing for IPO documents
├── lda_model_gensim.py            # Gensim LDA implementation with optimization
├── utils.py                       # Utility functions for file handling and metrics
├── visualizing_wordcloud.py       # Topic visualization utilities
├── config.yaml                    # Configuration file
├── pyproject.toml                 # Modern Python project configuration
├── uv.lock                        # Lock file for reproducible dependencies
├── stopwords/                     # Custom stopword lists
│   ├── financial_stopwords.txt
│   └── generic_stopwords.txt
├── data/
│   └── raw_reports/
└── outputs/
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

**Note:** Input documents are expected to be located in `data/raw_reports/` directory.

Example:
```bash
uv run main.py --config config.yaml -n 1000 -k 20 -nc 16 -b 200 -t 0.1
```

## Features

### Document Preprocessing
- **Modular preprocessing pipeline** organized in `data_preprocessing/` module
- Parallel processing for efficient document handling
- Custom stopword removal for financial domain
- Advanced lemmatization using spaCy with configurable POS tags
- Bigram and trigram detection
- TF-IDF transformation

### Topic Modeling
- **Enhanced Gensim LDA implementation** with comprehensive optimization
- Automated topic number optimization using multiple metrics (perplexity, coherence)
- Document-topic assignment analysis

### Visualization
- Word cloud generation for topics
- Word frequency analysis with statistical plots
- Perplexity score plotting across topic ranges

## Output

The pipeline generates several output files in the `outputs/` directory:

### Core Results
- `topics.txt`: Top words for each topic
- `document_topics.txt`: Topic distribution for each document
- `metrics.yaml`: Model performance metrics for train/test sets
- `lda_model.pkl`: Trained LDA model (if save_model is enabled)

### Optimization Results (when optimize_topics is enabled)
- `optimization_results.yaml`: Comprehensive optimization metrics including perplexity and coherence scores
- `topic_perplexity_scores.txt`: Legacy perplexity scores format
- `metrics_comparison.png`: Multi-metric comparison plot across topic ranges
- `perplexity_plot_test.png`: Perplexity vs. number of topics visualization
- `intermediate_models/`: Saved models during optimization (optional, configurable)

### Visualizations
- `wordcloud.png`: Word cloud visualization of topics
- `word_frequency_analysis.png`: Statistical analysis of word frequencies
- `word_frequency_distribution.png`: Distribution analysis of word frequencies

## Configuration

The `config.yaml` file provides comprehensive customization options:

### Preprocessing Settings
- TF-IDF filtering thresholds and parameters
- Allowed POS tags for text processing
- Custom stopword directories
- spaCy model configuration and disabled components

### LDA Model Settings
- **Topic optimization**: Enable/disable automatic topic number selection
- **Memory management**: Optional intermediate model saving during optimization
- Topic range configuration (start, limit, step)
- Gensim-specific parameters (random_state, iterations, passes, alpha, eta)

### Visualization & Output
- Wordcloud customization (dimensions, colors)
- Output directory structure
- Model/metrics/visualization saving preferences
- Logging level and format configuration

### Key New Features
- `save_intermediate_models`: Control memory usage during optimization
- Enhanced TF-IDF filtering with configurable thresholds
- Flexible topic range specification for optimization

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

