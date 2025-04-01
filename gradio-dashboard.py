import pandas as pd
import numpy as np
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma

import gradio as gr

load_dotenv()

movies = pd.read_csv("data/movies_with_emotion.csv")
movies["Large_Poster"] = movies["Poster_Link"] + "&fife=w800"
movies["Large_Poster"] = np.where(
    movies["Large_Poster"].isna(),
    "image_not_found.png",
    movies["Large_Poster"],
)

# Load the tagged descriptions from a text file using TextLoader.
# Instantiate a CharacterTextSplitter to split the documents into smaller chunks.
# Apply the text splitter to each document to create smaller text chunks.
# Convert these text chunks into document embeddings using the HuggingFaceEmbeddings model.
# Store the resulting document embeddings into a Chroma vector database for efficient retrieval.

raw_documents = TextLoader('tagged_description.txt').load()
text_splitter = CharacterTextSplitter(chunk_size=0, chunk_overlap=0, separator="\n")
documents = text_splitter.split_documents(raw_documents)
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
db_movies = Chroma.from_documents(documents, embedding=embedding_model)


# Get recommendations from vector database db_movies by user's query and limit it to top 24.
# Get Unique Id from splitting it from front of the tagged description.
# Return only those recommendations which matches the unique id of our movies database

def retrieve_semantic_recommendations(
    query: str,
    genre: str = None,
    tone: str = None,
    initial_top_k: int = 32,
    final_top_k: int = 8,
) -> pd.DataFrame:
    recs = db_movies.similarity_search(query, k=initial_top_k)
    movies_list = [int(rec.page_content.strip('"').split()[0]) for rec in recs]
    movies_recs = movies[movies["Unique_id"].isin(movies_list)].head(final_top_k)
    
    # Applying filter based on genre
    if genre != "All":
        movies_recs = movies_recs[movies_recs["Simple_genre"] == genre][:final_top_k]
    else:
        movies_recs = movies_recs.head(final_top_k)
        
    # Sorting recommendation based on emotion tone
    if tone == "Happy":
        movies_recs = movies_recs[movies_recs["Emotion_tone"] == "joy"].sort_values(by="Emotion_tone")
    elif tone == "Surprising":
        movies_recs = movies_recs[movies_recs["Emotion_tone"] == "surprise"].sort_values(by="Emotion_tone")
    elif tone == "Anger":
        movies_recs = movies_recs[movies_recs["Emotion_tone"] == "anger"].sort_values(by="Emotion_tone")
    elif tone == "Suspenseful":
        movies_recs = movies_recs[movies_recs["Emotion_tone"] == "fear"].sort_values(by="Emotion_tone")
    elif tone == "Sad":
        movies_recs = movies_recs[movies_recs["Emotion_tone"] == "sadness"].sort_values(by="Emotion_tone")

    return movies_recs


def recommend_movies(
    query: str,
    genre: str,
    tone: str
):
    recommendations = retrieve_semantic_recommendations(query, genre, tone)
    result = []
    
    for _, row in recommendations.iterrows():
        # Truncate Overview if it's over 30 words
        overview = row["Overview"]
        truncated_overview_split = overview.split()
        truncated_overview = " ".join(truncated_overview_split[:30]) + "..."
        
        caption = f"Movie: {row['Series_Title']}\n Starring: {row['Star1']}, {row['Star2']}, {row['Star3']} and {row['Star4']}\n Directed_by: {row['Director']}\n Description: {truncated_overview}"
        result.append((row["Large_Poster"], caption))
        
    return result
    

genres = ["All"] + sorted(movies["Simple_genre"].unique())
tones = ["All"] + ["Happy", "Surprising", "Angry", "Suspenseful", "Sad"]

with gr.Blocks(theme = gr.themes.Glass()) as dashboard:
    gr.Markdown("# Sementic Movie Recommender")
    
    with gr.Row():
        user_query = gr.Textbox(label = "Please enter a description of a movie:",
                                placeholder = "e.g., A suspense triller movie")
        genre_dropdown = gr.Dropdown(choices = genres, label = "Select a genre:", value="All")
        tone_dropdown = gr.Dropdown(choices = tones, label = "Select a emotional tone:", value="All")
        submit_button = gr.Button("Find Recommendations")
        
    gr.Markdown('## Recommendations')
    output = gr.Gallery(label = "Recommended movies", columns = 8)
    
    submit_button.click(fn = recommend_movies,
                        inputs = [user_query, genre_dropdown, tone_dropdown],
                        outputs = output)
    
if __name__ == "__main__":
    dashboard.launch()