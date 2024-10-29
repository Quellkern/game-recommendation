import json
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from scipy.sparse import save_npz
import re
from nltk.stem import WordNetLemmatizer
import nltk

nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()

# Paths
json_file_path = './games.json'
tfidf_matrix_path = 'games_tfidf_matrix.npz'
tfidf_vocab_path = 'tfidf_vectorizer.pkl'
model_path = 'game_recommender_model.pkl'

# Load JSON data into DataFrame
def load_json_to_df(file_path):
    games_list = []
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
        for app in data:
            game = data[app]
            games_list.append({
                'AppID': app,
                'Name': game.get('name', ''),
                'ReleaseDate': game.get('release_date', ''),
                'Developer': ', '.join(game.get('developers', ['Unknown'])),
                'Publisher': ', '.join(game.get('publishers', [])),
                'Price': game.get('price', 0.0),
                'Categories': ', '.join(game.get('categories', [])),
                'Genres': ', '.join(game.get('genres', [])),
                'ShortDescription': game.get('short_description', ''),
                'DetailedDescription': game.get('detailed_description', ''),
            })
    return pd.DataFrame(games_list)

# Text cleaning and lemmatization function
def clean_text(text):
    # Remove special characters and convert to lowercase
    text = re.sub(r'\W+', ' ', text).lower()
    # Lemmatize each word
    text = ' '.join(lemmatizer.lemmatize(word) for word in text.split())
    return text

# Load data and apply text cleaning to Combined features
df = load_json_to_df(json_file_path)
df['Combined'] = df['Categories'] + " " + df['Genres'] + " " + df['ShortDescription'] + " " + df['DetailedDescription']
df['Combined'] = df['Combined'].apply(clean_text)

# Define target and features
tfidf_vectorizer = TfidfVectorizer(max_features=25000, ngram_range=(1, 3), stop_words='english')  # Allow up to trigrams
X = tfidf_vectorizer.fit_transform(df['Combined'])
y = (df['Price'] < 10).astype(int)  # Binary classification (affordable vs. expensive)

# Save TF-IDF matrix and vectorizer
save_npz(tfidf_matrix_path, X)
with open(tfidf_vocab_path, 'wb') as f:
    pickle.dump(tfidf_vectorizer, f)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train logistic regression model
model = LogisticRegression(max_iter=100)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Save the trained model
with open(model_path, 'wb') as f:
    pickle.dump(model, f)

print(f"Model training completed with accuracy: {accuracy:.2f}")