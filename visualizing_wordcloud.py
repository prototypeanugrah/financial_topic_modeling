# 1. Wordcloud of Top N words in each topic
import math
import random
import numpy as np

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

    # Calculate grid size for subplots
    n_topics = len(topics_words)
    grid_size = int(np.ceil(np.sqrt(n_topics)))

    # Create figure with subplots
    fig = plt.figure(figsize=(5 * grid_size, 5 * grid_size))

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

    # Generate wordcloud for each topic
    for topic_idx, topic_words in topics_words.items():
        # Create subplot
        ax = fig.add_subplot(grid_size, grid_size, topic_idx + 1)

        # Generate wordcloud for this topic
        cloud.generate_from_frequencies(topic_words)

        # Display the wordcloud
        ax.imshow(cloud, interpolation="bilinear")
        ax.set_title(f"Topic {topic_idx + 1}", pad=20)
        ax.axis("off")

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(str(output_path), bbox_inches="tight", dpi=300)
    plt.close()


# def visualize_wordcloud(
#     model,
#     output_path,
#     wordcloud_config,
# ):
#     cols = [
#         color for _, color in mcolors.TABLEAU_COLORS.items()
#     ]  # more colors: 'mcolors.XKCD_COLORS'

#     width = wordcloud_config["width"]
#     height = wordcloud_config["height"]
#     background_color = wordcloud_config["background_color"]

#     stop_words = stopwords.words("english")

#     topics = model.show_topics(formatted=False)
#     num_topics = len(topics)

#     grid_size = math.ceil(math.sqrt(num_topics))
#     rows = grid_size
#     cols = grid_size

#     cloud = WordCloud(
#         stopwords=stop_words,
#         background_color=background_color,
#         width=width,
#         height=height,
#         max_words=10,
#         colormap="tab10",
#         color_func=lambda *args, **kwargs: cols[i % len(cols)],
#         prefer_horizontal=1.0,
#     )

#     # fig, axes = plt.subplots(2, 5, figsize=(7, 7), sharex=True, sharey=True)

#     # for i, ax in enumerate(axes.flatten()):
#     #     fig.add_subplot(ax)
#     #     topic_words = dict(topics[i][1])
#     #     cloud.generate_from_frequencies(topic_words, max_font_size=200)
#     #     plt.gca().imshow(cloud)
#     #     plt.gca().set_title("Topic " + str(i + 1), fontdict=dict(size=13))
#     #     plt.gca().axis("off")

#     # plt.subplots_adjust(wspace=0, hspace=0)
#     # plt.axis("off")
#     # plt.margins(x=0, y=0)
#     # plt.tight_layout()
#     # plt.savefig(output_path)
#     # # plt.show()

#     fig = plt.figure(figsize=(20, 20))

#     for i in range(num_topics):
#         if i < len(topics):  # Check if we have this topic
#             topic_words = dict(topics[i][1])

#             # Create subplot
#             ax = fig.add_subplot(rows, cols, i + 1)
#             cloud.generate_from_frequencies(topic_words, max_font_size=200)
#             plt.gca().imshow(cloud)
#             plt.gca().set_title(f"Topic {i + 1}", fontdict=dict(size=16))
#             plt.gca().axis("off")

#     plt.subplots_adjust(wspace=0, hspace=0)
#     plt.margins(x=0, y=0)
#     plt.tight_layout()

#     # Save the figure
#     plt.savefig(output_path, bbox_inches="tight", dpi=300)
#     plt.close()


# if __name__ == "__main__":
#     visualize_wordcloud()
