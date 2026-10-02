import pandas as pd
import nltk
import pickle
import os
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Download necessary NLTK data
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)

# 1. Load Data
print("Loading data...")
df = pd.read_csv('data/intents.csv')

# 2. Preprocessing Function
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return " ".join(tokens)

df['processed_text'] = df['text'].apply(preprocess_text)

# 3. Vectorization (TF-IDF)
print("Vectorizing text...")
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['processed_text'])
y = df['intent']

# 4. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Train the Model (Linear SVM)
print("Training model...")
model = LinearSVC()
model.fit(X_train, y_train)

# 6. Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# 7. Save the Model and Vectorizer
os.makedirs('models', exist_ok=True)
with open('models/chatbot_model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('models/vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

print("Model saved in 'models/' folder.")
