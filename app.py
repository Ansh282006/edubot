from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
import pickle
import nltk
from nltk.stem import WordNetLemmatizer

app = Flask(__name__)
CORS(app) 

# Load the model and vectorizer
with open('models/chatbot_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('models/vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# Load the data to get responses
df = pd.read_csv('data/intents.csv')

lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return " ".join(tokens)

# Serve the website
@app.route('/')
def home():
    return render_template('index.html')

# The API endpoint that the website will call
@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message')
    
    # Preprocess and predict
    processed_input = preprocess_text(user_message)
    vectorized_input = vectorizer.transform([processed_input])
    predicted_intent = model.predict(vectorized_input)[0]
    
    # Find the response
    response = df[df['intent'] == predicted_intent]['response'].values[0]
    
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
