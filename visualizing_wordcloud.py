# 1. Wordcloud of Top N words in each topic
import math
import random
import numpy as np
import os

from matplotlib import pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import matplotlib.colors as mcolors
from nltk.corpus import stopwords


def visualize_wordcloud(lda_model, output_path, config):
    """
    Generate and save wordcloud visualizations for each topic.

    Args:
        lda_model: Trained LDA model (either Gensim or scikit-learn)
        output_path: Path where to save the wordcloud images
        config: Dictionary containing wordcloud configuration parameters
    """
    # Define a list of colors for different topics
    colors = [
        "#FF9999",
        "#66B2FF",
        "#99FF99",
        "#FFCC99",
        "#FF99CC",
        "#99FFCC",
        "#FFB366",
        "#FF99FF",
        "#99CCFF",
        "#FF8080",
    ]

    # Get topic words and their weights based on model type
    topics_words = {}

    # Check if it's a Gensim model
    if hasattr(lda_model, "show_topics"):
        for idx, topic in lda_model.show_topics(formatted=False):
            topic_words = {}
            for word, weight in topic:
                topic_words[word] = weight
            topics_words[idx] = topic_words
    # Check if it's a scikit-learn model
    elif hasattr(lda_model, "components_"):
        # Get feature names from the vectorizer
        feature_names = config.get("feature_names", [])
        if len(feature_names) == 0:
            raise ValueError(
                "For scikit-learn models, feature_names must be provided in config"
            )

        # Get top words for each topic
        for topic_idx, topic in enumerate(lda_model.components_):
            topic_words = {}
            top_indices = topic.argsort()[: -config.get("n_top_words", 10) - 1 : -1]
            for idx in top_indices:
                word = feature_names[idx]
                weight = topic[idx]
                topic_words[word] = weight
            topics_words[topic_idx] = topic_words
    else:
        raise ValueError("Unsupported LDA model type")

    # Calculate grid size for subplots (max 4x4 grid per figure)
    n_topics = len(topics_words)
    topics_per_figure = 16  # Maximum number of subplots per figure
    n_figures = math.ceil(n_topics / topics_per_figure)
    grid_size = 4  # Fixed grid size of 4x4

    # Create and configure the WordCloud object
    cloud = WordCloud(
        width=config.get("width", 400),
        height=config.get("height", 400),
        background_color=config.get("background_color", "white"),
        max_words=config.get("max_words", 50),
        colormap=config.get("colormap", "viridis"),
        prefer_horizontal=config.get("prefer_horizontal", 0.7),
        relative_scaling=config.get("relative_scaling", 0.5),
        min_font_size=config.get("min_font_size", 10),
        max_font_size=config.get("max_font_size", 100),
        random_state=config.get("random_state", 42),
    )

    # Generate wordclouds for each topic, split across multiple figures if needed
    for fig_idx in range(n_figures):
        # Create figure
        fig = plt.figure(figsize=(20, 20))

        # Calculate topics for this figure
        start_idx = fig_idx * topics_per_figure
        end_idx = min((fig_idx + 1) * topics_per_figure, n_topics)

        # Generate wordcloud for each topic in this figure
        for topic_idx in range(start_idx, end_idx):
            if topic_idx in topics_words:
                # Calculate position in grid
                pos = topic_idx - start_idx
                row = pos // grid_size
                col = pos % grid_size

                # Create subplot
                ax = fig.add_subplot(grid_size, grid_size, pos + 1)

                # Generate wordcloud for this topic
                cloud.generate_from_frequencies(topics_words[topic_idx])

                # Display the wordcloud
                ax.imshow(cloud, interpolation="bilinear")
                ax.set_title(f"Topic {topic_idx + 1}", pad=20)
                ax.axis("off")

        # Adjust layout and save
        plt.tight_layout()
        if n_figures > 1:
            # If multiple figures, add figure number to filename
            base_path = str(output_path)
            name, ext = os.path.splitext(base_path)
            fig_path = f"{name}_fig{fig_idx + 1}{ext}"
        else:
            fig_path = str(output_path)

        plt.savefig(fig_path, bbox_inches="tight", dpi=300)
        plt.close()


# if __name__ == "__main__":
#     visualize_wordcloud()
