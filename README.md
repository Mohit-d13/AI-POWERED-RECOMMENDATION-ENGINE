# AI-POWERED-RECOMMENDATION-ENGINE

*COMPANY*: CODETECH IT SOLUTIONS

*NAME*: MOHIT DAMLE

*INTERN ID*: CT12WSJC

*DOMAIN*: BACKEND WEB DEVELOPMENT

*DURATION*: 12 WEEKS

*MENTOR*: NEELA SANTOSH

## 🎬 Description

A powerful movie recommendation engine that leverages natural language processing and machine learning to provide personalized movie suggestions based on semantic understanding of user preferences, movie content, and sentiment analysis.

This project implements a semantic movie recommender system that leverages KaggleHub movie datasets, Hugging Face Large Language Models (LLMs), embedding models, and a Gradio dashboard for user interaction. It employs techniques like data exploration, text classification, vector search, and sentiment analysis to provide personalized movie recommendations.

## 📋 Table of Contents

- [Description](#description)
- [Project Overview](#project-overview)
- [Features](#features)
- [How its Works](#how-it-works)
- [Technologies Used](#technologies-used)
- [Setup and Installation](#setup-and-installation)
- [Gradio Dashboard](#gradio-dashboard)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## 👀 Project Overview

The goal of this project is to build a movie recommender system that goes beyond traditional collaborative filtering by understanding the semantic meaning of movie descriptions and user preferences. It utilizes:

-   **KaggleHub Datasets:** Movie metadata, including plots, genres, and user reviews.
-   **Hugging Face LLMs:** For tasks like text classification and sentiment analysis.
-   **Hugging Face Embedding Models:** To create vector representations of movie descriptions and user queries.
-   **Vector Search:** To find movies with similar semantic meanings.
-   **Gradio:** To create an interactive web interface.

## 🚀 Features

- **Semantic Understanding**: Utilizes Hugging Face language models to comprehend movie plots, genres, and themes
- **Content-Based Filtering**: Recommends movies based on content similarity using vector embeddings
- **Sentiment Analysis**: Analyzes user reviews to factor emotional responses into recommendations
- **Interactive Dashboard**: Built with Gradio for easy exploration and testing
- **Text Classification**: Categorizes movies based on multiple attributes
- **Vector Search**: Fast retrieval of similar movies using vector embeddings

## 🔍 How It Works

1. **Data Processing**:
   - Movies are cleaned and processed from the Kaggle dataset
   - Text fields (plot, synopsis, reviews) are normalized

2. **Feature Extraction**:
   - Hugging Face embedding models convert text to vector representations
   - Sentiment analysis classifies review sentiment

3. **Recommendation Engine**:
   - Vector similarity search finds semantically similar movies
   - Results are filtered based on user preferences
   - Recommendations are scored using multiple factors

4. **User Interface**:
   - Gradio provides an interactive dashboard
   - Users can search by movie title or describe preferences
   - System displays recommendations with explanation

## 🛠️ Technologies Used

- **Data Source**: Kaggle Movies Dataset via KaggleHub
- **NLP Models**: Hugging Face Transformers
- **Embeddings**: Hugging Face Sentence Transformers
- **UI**: Gradio Dashboard
- **Analysis**: Python, Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn

## 📦 Setup and Installation

1.  **Clone the repository:**

    ```bash
    git clone <repository_url>
    cd movie_recommender_system
    ```

2.  **Create a virtual environment (recommended):**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Linux/macOS
    venv\Scripts\activate  # On Windows
    ```

3.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

4.  **Download necessary datasets and place them in the `data/` directory.**

5.  **Download the huggingface models and api, if necessary, or configure the notebook to download them on demand.**

## 🚀 Getting Started

1. **Download and prepare the dataset**:
   ```python
   import kagglehub
   
   # Download latest version
    path = kagglehub.dataset_download("path/to/movie/dataset")
    print("Path to dataset files:", path)

   ```

2.  **Run the notebooks:**

    -   Execute the notebooks in the `notebooks/` directory in the specified order:
        -   `data_exploration.ipynb`: For dataset analysis and preprocessing. Find missing data, create new tagged description with every movie unique id and overview. Make new cleaned movie csv file for text classification.
        -   `vector_search.ipynb`: Convert the whole document into meaningful chunks of data then convert them into document embeddings with HuggingfaceEmbedding model and store that data in chroma vector database.
        -   `text_classification.ipynb`: For classifying movie genres or other relevant categories with zero-shot classification.
        -   `sentiment_analysis.ipynb`: To analyze movie emotional tone classify them and make it new filter for search engine.

3.  **Launch the Gradio dashboard:**

    ```bash
    python gradio_dashboard.py
    ```

    This will start the Gradio web interface, which you can access in your browser.


## 🖥️ Gradio Dashboard

The `gradio_dashboard.py` script creates an interactive web interface using Gradio. Users can:

-   Enter a movie description or query.
-   Receive personalized movie recommendations.
-   View movie details and sentiment analysis results.

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1.  Fork the repository.
2.  Create a new branch for your feature or bug fix.
3.  Make your changes and commit them.
4.  Push your changes to your fork.
5.  Submit a pull request.

## 📝 License

This project is licensed under the [MIT License](LICENSE).
 

## 🙏 Acknowledgements

- [Kaggle](https://www.kaggle.com/) for providing the movie datasets
- [Hugging Face](https://huggingface.co/) for transformers and embedding models
- [Gradio](https://gradio.app/) for the interactive UI framework