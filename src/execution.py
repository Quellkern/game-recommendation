import json
import pickle
import re
import pandas as pd
from scipy.sparse import load_npz
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem import WordNetLemmatizer
import nltk

nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()

# Load and process the JSON data with UTF-8 encoding
with open('./games.json', encoding='utf-8') as f:
    data = json.load(f)

# Load the saved TF-IDF vectorizer and TF-IDF matrix from training
with open('tfidf_vectorizer.pkl', 'rb') as f:
    tfidf_vectorizer = pickle.load(f)
tfidf_matrix = load_npz('games_tfidf_matrix.npz')

# Convert the nested JSON structure to a flat DataFrame
games_list = []
for game_id, game_info in data.items():
    game_info['game_id'] = int(game_id)  # Add game ID to each game entry
    games_list.append(game_info)

# Create the DataFrame
df = pd.DataFrame(games_list)

# Check for 'genres' and 'categories' columns and handle missing ones
if 'genres' not in df.columns:
    print("The 'genres' column is missing.")
else:
    df['genres'] = df['genres'].apply(lambda x: ' '.join(x) if isinstance(x, list) else '')

if 'categories' not in df.columns:
    print("The 'categories' column is missing.")
else:
    df['categories'] = df['categories'].apply(lambda x: ' '.join(x) if isinstance(x, list) else '')

# Text cleaning and lemmatization function
def clean_text(text):
    # Remove special characters and convert to lowercase
    text = re.sub(r'\W+', ' ', text).lower()
    # Lemmatize each word
    text = ' '.join(lemmatizer.lemmatize(word) for word in text.split())
    return text

# Function to recommend games based on genre, price, user score, and description similarity
def recommend_games(df, genres=None, max_price=None, min_user_score=0, description=None):
    genre_filter = pd.Series([True] * len(df))
    price_filter = pd.Series([True] * len(df))
    user_score_filter = pd.Series([True] * len(df))
    description_score = pd.Series([0] * len(df), dtype=float)  # Default similarity score for all games

    # Calculate genre match score (require all specified genres to appear in game genres)
    if genres:
        genre_match_count = df['genres'].apply(lambda x: all(genre.lower() in x.lower() for genre in genres))
        genre_filter = genre_match_count  # Only select games that match all genres
        genre_score = genre_match_count.astype(float)  # Set to 1 if all genres match, else 0
    else:
        genre_score = pd.Series([1] * len(df))  # Default to 1 if no genre filter is applied

    # Calculate price score (1 if within max_price, otherwise 0)
    if max_price is not None:
        price_filter = df['price'] <= max_price
        price_score = price_filter.astype(int)  # 1 if within budget, 0 otherwise
    else:
        price_score = pd.Series([1] * len(df))  # Default to 1 if no max price

    # Calculate user score compliance (1 if above min_user_score, otherwise 0)
    if min_user_score > 0:
        user_score_filter = df['user_score'] >= min_user_score
        user_score_score = user_score_filter.astype(int)  # 1 if meets user score threshold, 0 otherwise
    else:
        user_score_score = pd.Series([1] * len(df))  # Default to 1 if no user score threshold

    # Calculate description similarity if description is provided
    if description:
        cleaned_description = clean_text(description)
        description_tfidf = tfidf_vectorizer.transform([cleaned_description])
        cosine_similarities = cosine_similarity(description_tfidf, tfidf_matrix).flatten()
        description_score = pd.Series(cosine_similarities)

    # Apply filters
    filtered_df = df[genre_filter & price_filter & user_score_filter]
    filtered_df = filtered_df[['name', 'price', 'genres', 'categories', 'user_score']].copy()

    # Calculate single final score by combining all criteria with adjusted weighting
    filtered_df['final_score'] = (
        0.3 * genre_score[genre_filter & price_filter & user_score_filter].values +  # Moderate weight for genre
        0.1 * price_score[genre_filter & price_filter & user_score_filter].values +  # Lower weight for price
        0.1 * user_score_score[genre_filter & price_filter & user_score_filter].values +  # Lower weight for user score
        0.5 * description_score[genre_filter & price_filter & user_score_filter].values  # High weight for description similarity
    ) * 100

    return filtered_df.sort_values(by='final_score', ascending=False)

# Example usage
recommendations = recommend_games(df, genres=['Racing', 'Sports'], max_price=15, description="realistic racing game with customizable cars and competitive tracks")

# Print the top 10 recommendations
print(recommendations.head(10))
