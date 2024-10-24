import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split
from surprise import accuracy

# Step 1: Load the JSON data (assume 'games.json' is the file path)
with open('/path_to_your_file/games.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Step 2: Normalize JSON data into a pandas DataFrame
games_df = pd.json_normalize(data)

# Step 3: Select relevant columns for the recommendation model
games_df = games_df[['name', 'genres', 'categories', 'developers', 'price', 'positive_ratings', 'negative_ratings']]

# Fill missing values
games_df.fillna('', inplace=True)

# Step 4: Incorporate user ratings (Normalize Ratings)
games_df['positive_ratings'] = games_df['positive_ratings'].fillna(0)
games_df['negative_ratings'] = games_df['negative_ratings'].fillna(0)
games_df['user_score'] = games_df['positive_ratings'] / (games_df['positive_ratings'] + games_df['negative_ratings'] + 1e-5)

# Combine genres, categories, developers, and user score into a single feature
games_df['combined_features'] = games_df['genres'] + ' ' + games_df['categories'] + ' ' + games_df['developers'] + ' ' + games_df['user_score'].astype(str)

# Step 5: Content-Based Filtering using TF-IDF Vectorizer
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(games_df['combined_features'])

# Compute the cosine similarity matrix
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Step 6: Collaborative Filtering using SVD
# Prepare example user rating data (adjust as needed)
rating_data = pd.DataFrame({
    'user_id': [1, 2, 1, 2, 3],  # Example user IDs
    'name': ['Portal 2', 'Galactic Bowling', 'Half-Life 2', 'Portal', 'Train Bandit'],
    'user_rating': [5, 4, 5, 3, 4]  # Example ratings
})

# Create a Reader object to specify the rating scale
reader = Reader(rating_scale=(1, 5))

# Load the data into the Surprise dataset format
data = Dataset.load_from_df(rating_data[['user_id', 'name', 'user_rating']], reader)

# Train-test split
trainset, testset = train_test_split(data, test_size=0.2)

# Train the SVD model
model = SVD()
model.fit(trainset)

# Test the model
predictions = model.test(testset)
accuracy.rmse(predictions)

# Function to predict the rating for a specific user and game
def predict_user_rating(user_id, game_name):
    return model.predict(user_id, game_name).est

# Step 7: Hybrid Recommendation System ensuring at least 5-8 games
def hybrid_recommendations(game_name, user_id, price_range=None, cosine_sim=cosine_sim, min_recommendations=5, top_n=10):
    # Content-based filtering to get similar games
    try:
        idx = games_df[games_df['name'].str.lower() == game_name.lower()].index[0]
    except IndexError:
        return "Game not found. Please check the spelling or try another game."
    
    # Get similarity scores
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n+1]  # Exclude the game itself, retrieve more than 5 games

    # Normalize similarity scores to percentage
    max_score = sim_scores[0][1]
    recommended_games = []
    for idx, score in sim_scores:
        confidence_score = (score / max_score) * 100
        game_details = games_df.iloc[idx][['name', 'genres', 'price', 'developers', 'user_score']].to_dict()
        game_details['confidence_score'] = confidence_score
        
        # Collaborative filtering to predict the user's rating for each game
        predicted_rating = predict_user_rating(user_id, game_details['name'])
        game_details['predicted_user_rating'] = predicted_rating
        
        # Combine content similarity score and predicted rating (weighted hybrid score)
        hybrid_score = 0.5 * confidence_score + 0.5 * predicted_rating  # Equal weights for content and collaborative scores
        game_details['hybrid_score'] = hybrid_score
        
        recommended_games.append(game_details)
    
    # Sort the games by hybrid score
    recommended_games = sorted(recommended_games, key=lambda x: x['hybrid_score'], reverse=True)

    # Apply price filtering if a range is provided
    if price_range:
        min_price, max_price = price_range
        recommended_games = [game for game in recommended_games if min_price <= game['price'] <= max_price]

    # Ensure at least min_recommendations (5-8) games are returned
    if len(recommended_games) < min_recommendations:
        recommended_games = recommended_games[:min_recommendations]  # If not enough, return top by hybrid score
    
    return recommended_games

# Example usage: Get hybrid recommendations for user 1 based on 'Portal 2' with a price filter
hybrid_recs = hybrid_recommendations('Portal 2', user_id=1, price_range=(0, 20), min_recommendations=5, top_n=15)

for game in hybrid_recs:
    print(f"Game: {game['name']}, Hybrid Score: {game['hybrid_score']:.2f}, Predicted User Rating: {game['predicted_user_rating']:.2f}, Confidence: {game['confidence_score']:.2f}%")
