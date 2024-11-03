import json
import pickle
import re
import pandas as pd
from scipy.sparse import load_npz
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem import WordNetLemmatizer
import nltk
import matplotlib.pyplot as plt
import mpld3  # Import mpld3 for interactive plots

nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()

# Load and process the JSON data with UTF-8 encoding
with open('./games.json', encoding='utf-8') as f:
    data = json.load(f)

# Extract game_id and estimated_owners into a DataFrame
# games_list = []
# for game_id, game_info in data.items():
#     game_info['game_id'] = int(game_id)  # Add game ID to each game entry
#     games_list.append(game_info)

# # Create the DataFrame
# df = pd.DataFrame(games_list)

# # Save the extracted game_id and estimated_owners to a CSV file
# if 'estimated_owners' in df.columns:
#     owners_df = df[['game_id', 'estimated_owners']].copy()
#     owners_df.to_csv('./estimated_owners.csv', index=False)
#     print("Extracted game_id and estimated_owners saved to 'estimated_owners.csv'.")
# else:
#     print("The 'estimated_owners' column is missing.")

game_id = input("Please enter the game ID: ")

# Fetch the game data
if game_id in data:
    game_data = data[game_id]
else:
    print(f"Game ID {game_id} not found in the data.")
    exit()

# Data for the graph
labels = ['Positive Reviews', 'Negative Reviews']
values = [game_data["positive"], game_data["negative"]]

# Create the bar graph
plt.figure(figsize=(8, 5))  # Set figure size
plt.bar(labels, values, color=['green', 'red'])
plt.title(f"Reviews for {game_data['name']}", fontsize=16)
plt.xlabel('Review Type', fontsize=14)
plt.ylabel('Number of Reviews', fontsize=14)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Display the graph
plt.tight_layout()  # Adjust layout to make room for labels
plt.show()

# Function to calculate total reviews for each game
def calculate_total_reviews(data):
    total_reviews = {}
    for game_id, game_info in data.items():
        total_reviews[game_id] = game_info["positive"] + game_info["negative"]
    return total_reviews

# Get total reviews
total_reviews = calculate_total_reviews(data)

# Get top N games with the highest reviews
N = 5  # Change this number to display more or fewer top games
top_games = sorted(total_reviews.items(), key=lambda x: x[1], reverse=True)[:N]

# Prepare data for the bar graph
top_game_ids = [game[0] for game in top_games]
top_game_totals = [game[1] for game in top_games]
top_game_names = [data[game_id]['name'] for game_id in top_game_ids]

# Create a bar graph for the top N games
plt.figure(figsize=(10, 6))  # Set figure size
plt.barh(top_game_names, top_game_totals, color='skyblue')
plt.xlabel('Total Reviews', fontsize=14)
plt.title('Top Games with Highest Total Reviews', fontsize=16)
plt.grid(axis='x', linestyle='--', alpha=0.7)

# Display the graph
plt.tight_layout()  # Adjust layout to make room for labels
plt.show()

# Prompt the user to enter a game ID
game_id_input = input("\nPlease enter a game ID to see its reviews: ")

# Fetch the game data for the input game ID
if game_id_input in data:
    game_data = data[game_id_input]
else:
    print(f"Game ID {game_id_input} not found in the data.")
    exit()

# Data for the individual game graph
labels = ['Positive Reviews', 'Negative Reviews']
values = [game_data["positive"], game_data["negative"]]

# Create a bar graph for the specific game
plt.figure(figsize=(8, 5))  # Set figure size
plt.bar(labels, values, color=['green', 'red'])
plt.title(f"Reviews for {game_data['name']}", fontsize=16)
plt.xlabel('Review Type', fontsize=14)
plt.ylabel('Number of Reviews', fontsize=14)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Display the graph
plt.tight_layout()  # Adjust layout to make room for labels
plt.show()

# Load the saved TF-IDF vectorizer and TF-IDF matrix from training
with open('tfidf_vectorizer.pkl', 'rb') as f:
    tfidf_vectorizer = pickle.load(f)
tfidf_matrix = load_npz('games_tfidf_matrix.npz')

# Check for 'genres', 'categories', and handle missing ones
if 'genres' not in df.columns:
    print("The 'genres' column is missing.")
else:
    df['genres'] = df['genres'].apply(lambda x: ' '.join(x) if isinstance(x, list) else '')

if 'categories' not in df.columns:
    print("The 'categories' column is missing.")
else:
    df['categories'] = df['categories'].apply(lambda x: ' '.join(x) if isinstance(x, list) else '')

if 'estimated_owners' not in df.columns:
    print("The 'estimated_owners' column is missing.")
else:
    df['estimated_owners'] = pd.to_numeric(df['estimated_owners'], errors='coerce')  # Ensure numeric

# Calculate average estimated owners, ignoring NaN values
avg_owners = df.groupby('game_id')['estimated_owners'].mean().reset_index()
avg_owners = avg_owners.dropna()  # Remove NaN values

# Plotting average estimated owners for each game
plt.figure(figsize=(16, 8))
plt.bar(avg_owners['game_id'], avg_owners['estimated_owners'], color='blue')
plt.title("Average Estimated Owners by Game ID")
plt.xlabel("Game ID")
plt.ylabel("Average Estimated Owners")
plt.xticks(rotation=90)  # Rotate x-axis labels for better readability

# Set y-axis limit slightly above max value, handling Inf or NaN
max_y = avg_owners['estimated_owners'].max() if not avg_owners['estimated_owners'].isnull().all() else 0
plt.ylim(0, max_y + 10)  # Set y-axis limit slightly above max value

# Save the plot to a file (optional)
plt.savefig('average_estimated_owners_by_game_id.png', bbox_inches='tight')

# Show interactive plot
mpld3.show()  # This will open the interactive plot in a browser

# Text cleaning and lemmatization function
def clean_text(text):
    # Remove special characters and convert to lowercase
    text = re.sub(r'\W+', ' ', text).lower()
    # Lemmatize each word
    text = ' '.join(lemmatizer.lemmatize(word) for word in text.split())
    return text

# Function to recommend games based on genre, price, and description similarity
def recommend_games(df, genres=None, max_price=None, description=None):
    genre_filter = pd.Series([True] * len(df))
    price_filter = pd.Series([True] * len(df))
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

    # Calculate description similarity if description is provided
    if description:
        cleaned_description = clean_text(description)
        description_tfidf = tfidf_vectorizer.transform([cleaned_description])
        cosine_similarities = cosine_similarity(description_tfidf, tfidf_matrix).flatten()
        description_score = pd.Series(cosine_similarities)

    # Apply filters
    filtered_df = df[genre_filter & price_filter]
    filtered_df = filtered_df[['name', 'price', 'genres', 'categories']].copy()

    # Calculate single final score by combining all criteria with adjusted weighting
    filtered_df['final_score'] = (
        0.4 * genre_score[genre_filter & price_filter].values +     # Moderate weight for genre
        0.2 * price_score[genre_filter & price_filter].values +     # Moderate weight for price
        0.4 * description_score[genre_filter & price_filter].values # High weight for description similarity
    ) * 100

    return filtered_df.sort_values(by='final_score', ascending=False)

# Example usage for recommendations
genres = list(input("Enter Genres: ").split(" "))
max_price = int(input("Enter max price: "))
description = input("Enter description: ")
recommendations = recommend_games(df, genres, max_price, description)

# Print the top 10 recommendations
print(recommendations.head(10))