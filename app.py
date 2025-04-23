from flask import Flask, request, jsonify
from flask_cors import CORS
import os

from spam_classifier import SpamClassifier

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize the spam classifier
classifier = SpamClassifier()

@app.route('/api/classify', methods=['POST'])
def classify_message():
    """API endpoint to classify a message as spam or ham"""
    data = request.json

    if not data or 'message' not in data:
        return jsonify({'error': 'No message provided'}), 400

    message = data['message']

    try:
        # Ensure the model is loaded
        if not classifier.load_model():
            # If model doesn't exist, train it
            print("Training model...")
            classifier.train('spam.csv')

        # Classify the message
        result = classifier.predict(message)

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/train', methods=['POST'])
def train_model():
    """API endpoint to train the model"""
    try:
        accuracy = classifier.train('spam.csv')
        return jsonify({
            'success': True,
            'accuracy': accuracy
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/nlp-techniques', methods=['GET'])
def get_nlp_techniques():
    """API endpoint to get information about NLP techniques used"""
    techniques = [
        {
            'id': 'preprocessing',
            'name': 'Text Preprocessing',
            'techniques': [
                'Text Normalization - Converting text to lowercase and removing extra whitespace',
                'Text Cleaning - Removing special characters, numbers, URLs, and email addresses',
                'Tokenization - Breaking text into individual words',
                'Stopword Removal - Filtering out common words with low semantic value',
                'Stemming - Reducing words to their root form',
                'N-gram Extraction - Capturing sequences of words as features'
            ]
        },
        {
            'id': 'feature_extraction',
            'name': 'Feature Extraction',
            'techniques': [
                'TF-IDF Vectorization - Converting text to numerical features based on term frequency and importance',
                'Term Frequency (TF) - How often a word appears in a document',
                'Inverse Document Frequency (IDF) - How rare a word is across all documents',
                'TF-IDF Weighting - Giving higher weight to important, distinctive words'
            ]
        },
        {
            'id': 'classification',
            'name': 'Text Classification',
            'techniques': [
                'Naive Bayes - Probabilistic classifier based on Bayes theorem',
                'Conditional Probability - Calculating P(word|class) for each word and class',
                'Laplace Smoothing - Handling zero probabilities for unseen words',
                'Log Probability - Using log probabilities to avoid numerical underflow',
                'Feature Importance Analysis - Identifying words most indicative of spam or ham'
            ]
        },
        {
            'id': 'evaluation',
            'name': 'Evaluation Metrics',
            'techniques': [
                'Accuracy - Proportion of correctly classified messages',
                'Precision - Proportion of predicted spam messages that are actually spam',
                'Recall - Proportion of actual spam messages correctly identified',
                'F1-Score - Harmonic mean of precision and recall',
                'Confusion Matrix - Detailed breakdown of prediction results'
            ]
        }
    ]

    return jsonify(techniques)

@app.route('/api/health', methods=['GET'])
def health_check():
    """API endpoint to check if the server is running"""
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    # Force retrain the model with our improved implementation
    print("Training model with improved NLP techniques...")
    classifier.train('spam.csv')

    app.run(debug=True, host='0.0.0.0', port=5000)
