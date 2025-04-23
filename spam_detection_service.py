"""
Spam Detection Service

This module provides a service layer for spam detection, orchestrating
the entire process from data loading to prediction using NLP techniques.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import joblib
import os

from nlp_engine import (
    preprocess_text, TfidfVectorizer, MultinomialNaiveBayes,
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)

class SpamClassifier:
    """
    A spam detection service that orchestrates the entire spam classification process
    using NLP techniques from the NLP engine.
    """
    
    def __init__(self):
        """Initialize the spam classifier service"""
        self.model = None
        self.vectorizer = None
        self.model_file = 'spam_classifier_model.pkl'
        self.vectorizer_file = 'tfidf_vectorizer.pkl'
    
    def load_data(self, file_path):
        """Load and prepare the dataset"""
        # Read the CSV file
        df = pd.read_csv(file_path, encoding='latin-1')
        
        # Keep only the necessary columns
        df = df[['v1', 'v2']]
        df.columns = ['label', 'text']
        
        # Convert labels to binary (0 for ham, 1 for spam)
        df['label'] = df['label'].map({'ham': 0, 'spam': 1})
        
        # Check class distribution
        class_distribution = df['label'].value_counts(normalize=True)
        print("Class Distribution:")
        print(f"Ham: {class_distribution[0]:.2%}")
        print(f"Spam: {class_distribution[1]:.2%}")
        
        return df
    
    def preprocess_data(self, df):
        """Preprocess the text data using NLP techniques"""
        # Apply preprocessing to each text
        df['processed_text'] = df['text'].apply(lambda x: preprocess_text(x, include_ngrams=True))
        
        return df
    
    def train(self, file_path):
        """Train the spam classification model using NLP techniques"""
        print("Training spam classifier with NLP techniques...")
        
        # Load and preprocess data
        df = self.load_data(file_path)
        df = self.preprocess_data(df)
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            df['processed_text'], df['label'], test_size=0.2, random_state=42
        )
        
        # Create and fit TF-IDF vectorizer
        print("Applying TF-IDF vectorization...")
        self.vectorizer = TfidfVectorizer(max_features=5000)
        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        X_test_tfidf = self.vectorizer.transform(X_test)
        
        # Train Naive Bayes classifier with TF-IDF features
        print("Training Naive Bayes classifier...")
        self.model = MultinomialNaiveBayes(alpha=0.1)  # Lower alpha for better performance
        
        # Convert TF-IDF matrix to list of strings for our custom NB implementation
        feature_names = self.vectorizer.get_feature_names()
        X_train_features = []
        
        # For each document, create a string with the most important features
        for i in range(X_train_tfidf.shape[0]):
            doc_features = X_train_tfidf[i]
            # Get indices of non-zero features
            feature_indices = np.where(doc_features > 0)[0]
            # Get the corresponding feature names
            doc_terms = [feature_names[idx] for idx in feature_indices]
            # Join into a single string with repetition based on TF-IDF weight
            X_train_features.append(' '.join(doc_terms))
        
        # Train the model with the extracted features
        self.model.fit(X_train_features, y_train)
        
        # Evaluate the model
        print("Evaluating model performance...")
        
        # Convert test data to TF-IDF features
        X_test_tfidf = self.vectorizer.transform(X_test)
        
        # Convert TF-IDF to feature strings for our custom NB implementation
        X_test_features = []
        for i in range(X_test_tfidf.shape[0]):
            doc_features = X_test_tfidf[i]
            feature_indices = np.where(doc_features > 0)[0]
            doc_terms = [feature_names[idx] for idx in feature_indices]
            X_test_features.append(' '.join(doc_terms))
        
        # Make predictions
        y_pred = self.model.predict(X_test_features)
        
        # Calculate evaluation metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, pos_label=1)
        recall = recall_score(y_test, y_pred, pos_label=1)
        f1 = f1_score(y_test, y_pred, pos_label=1)
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
        
        print(f"Model Performance:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1 Score: {f1:.4f}")
        print(f"  Confusion Matrix:")
        print(f"    True Ham  | True Spam")
        print(f"  Pred Ham  | {cm[0, 0]:<8} | {cm[1, 0]:<8}")
        print(f"  Pred Spam | {cm[0, 1]:<8} | {cm[1, 1]:<8}")
        
        # Get feature importance
        feature_importance = self.model.get_feature_importance(top_n=10)
        print("\nTop 10 features for each class:")
        for c in sorted(feature_importance.keys()):
            class_name = "Spam" if c == 1 else "Ham"
            print(f"\n  {class_name} class:")
            for word, score in feature_importance[c]:
                print(f"    {word}: {score:.4f}")
        
        # Save the model and vectorizer
        joblib.dump(self.model, self.model_file)
        joblib.dump(self.vectorizer, self.vectorizer_file)
        
        print(f"\nModel saved to {self.model_file}")
        print(f"Vectorizer saved to {self.vectorizer_file}")
        
        return accuracy
    
    def load_model(self):
        """Load a pre-trained model if it exists"""
        if os.path.exists(self.model_file) and os.path.exists(self.vectorizer_file):
            self.model = joblib.load(self.model_file)
            self.vectorizer = joblib.load(self.vectorizer_file)
            return True
        return False
    
    def predict(self, text):
        """Predict if a message is spam or ham"""
        if self.model is None or self.vectorizer is None:
            if not self.load_model():
                raise Exception("Model not trained. Please train the model first.")
        
        # Preprocess the input text
        processed_text = preprocess_text(text, include_ngrams=True)
        
        # Apply TF-IDF vectorization
        text_tfidf = self.vectorizer.transform([processed_text])
        
        # Convert TF-IDF to feature string for our custom NB implementation
        feature_names = self.vectorizer.get_feature_names()
        doc_features = text_tfidf[0]
        feature_indices = np.where(doc_features > 0)[0]
        doc_terms = [feature_names[idx] for idx in feature_indices]
        feature_string = ' '.join(doc_terms)
        
        # Make prediction
        prediction = self.model.predict([feature_string])[0]
        
        # Get probabilities
        proba = self.model._predict_proba_single(feature_string)
        spam_prob = proba.get(1, 0.0)
        ham_prob = proba.get(0, 0.0)
        
        # Get important features
        words = processed_text.split()
        important_features = []
        
        # Find which words contributed most to the classification
        if self.model.vocab:
            # Calculate word importance for the predicted class
            word_importance = {}
            predicted_class = 1 if prediction else 0
            
            for word in words:
                if word in self.model.vocab:
                    # Calculate P(word|class) / P(word|not_class)
                    p_word_given_class = (self.model.class_word_counts[predicted_class][word] + self.model.alpha) / (
                        self.model.class_total_words[predicted_class] + self.model.alpha * len(self.model.vocab))
                    
                    # P(word|not_class)
                    not_class = 0 if predicted_class == 1 else 1
                    p_word_given_not_class = (self.model.class_word_counts[not_class][word] + self.model.alpha) / (
                        self.model.class_total_words[not_class] + self.model.alpha * len(self.model.vocab))
                    
                    # Importance score
                    if p_word_given_not_class > 0:
                        word_importance[word] = p_word_given_class / p_word_given_not_class
                    else:
                        word_importance[word] = float('inf')
            
            # Get top 5 important words
            important_features = [word for word, _ in sorted(word_importance.items(), key=lambda x: x[1], reverse=True)[:5]]
        
        result = {
            'is_spam': bool(prediction),
            'spam_probability': float(spam_prob),
            'ham_probability': float(ham_prob),
            'important_features': important_features
        }
        
        return result
