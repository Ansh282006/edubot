from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
import pickle
import nltk
from nltk.stem import WordNetLemmatizer
from entity_extractor import EntityExtractor

app = Flask(__name__)
CORS(app)

with open('models/chatbot_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('models/vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

df_intents = pd.read_csv('data/intents.csv')
lemmatizer = WordNetLemmatizer()
extractor = EntityExtractor('data/knowledge_base.csv')

def preprocess_text(text):
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return " ".join(tokens)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message')
    processed = preprocess_text(user_message)

    vec = vectorizer.transform([processed])
    intent = model.predict(vec)[0]

    entity = extractor.extract(user_message)

    if entity:
        if intent == 'ask_formula':
            formula = extractor.get_formula(entity)
            if formula:
                return jsonify({'response': "The formula for " + entity.replace('_',' ') + " is: " + formula})
        if intent == 'ask_definition':
            definition = extractor.get_definition(entity)
            if definition:
                return jsonify({'response': definition})
        if intent not in ['greeting', 'goodbye', 'thanks']:
            definition = extractor.get_definition(entity)
            formula = extractor.get_formula(entity)
            combined = definition or ""
            if formula:
                combined += "  Formula: " + formula
            if combined:
                return jsonify({'response': combined})

    responses = df_intents[df_intents['intent'] == intent]['response'].values
    if len(responses) == 0:
        return jsonify({'response': "Sorry, I don't know that yet. Try asking about a different topic."})
    return jsonify({'response': responses[0]})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
