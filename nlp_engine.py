"""
NLP Engine for Text Processing and Classification

This module implements core NLP algorithms and techniques for text preprocessing,
feature extraction, and classification for spam detection.
"""

import re
import string
import math
import numpy as np
from collections import Counter, defaultdict

# =====================================================================
# TEXT PREPROCESSING TECHNIQUES
# =====================================================================

def normalize_text(text):
    """
    NLP TECHNIQUE 1: TEXT NORMALIZATION
    Normalize text by converting to lowercase and removing extra whitespace
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove extra whitespace
    text = re.sub(r'\\s+', ' ', text).strip()
    
    return text

def clean_text(text):
    """
    NLP TECHNIQUE 2: TEXT CLEANING
    Clean text by removing special characters, numbers, and URLs
    """
    # Remove URLs
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    
    # Remove email addresses
    text = re.sub(r'[\\w\\.-]+@[\\w\\.-]+', '', text)
    
    # Remove phone numbers
    text = re.sub(r'\\b\\d{3}[-.]?\\d{3}[-.]?\\d{4}\\b', '', text)
    
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\\s]', '', text)
    
    return text

def tokenize_text(text):
    """
    NLP TECHNIQUE 3: TOKENIZATION
    Tokenize text into individual words
    """
    # Simple tokenization by splitting on whitespace
    return text.split()

def remove_stopwords(tokens, stop_words=None):
    """
    NLP TECHNIQUE 4: STOPWORD REMOVAL
    Remove common stopwords from token list
    """
    if stop_words is None:
        # Common English stopwords
        stop_words = {
            'a', 'an', 'the', 'and', 'or', 'but', 'if', 'because', 'as', 'what',
            'which', 'this', 'that', 'these', 'those', 'then', 'just', 'so', 'than',
            'such', 'both', 'through', 'about', 'for', 'is', 'of', 'while', 'during',
            'to', 'from', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further',
            'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any',
            'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor',
            'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 'can', 'will',
            'just', 'should', 'now', 'i', 'me', 'my', 'myself', 'we', 'our', 'ours',
            'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him',
            'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself',
            'they', 'them', 'their', 'theirs', 'themselves', 'am', 'is', 'are', 'was',
            'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',
            'did', 'doing', 'would', 'should', 'could', 'ought', 'i\'m', 'you\'re',
            'he\'s', 'she\'s', 'it\'s', 'we\'re', 'they\'re', 'i\'ve', 'you\'ve',
            'we\'ve', 'they\'ve', 'i\'d', 'you\'d', 'he\'d', 'she\'d', 'we\'d',
            'they\'d', 'i\'ll', 'you\'ll', 'he\'ll', 'she\'ll', 'we\'ll', 'they\'ll',
            'isn\'t', 'aren\'t', 'wasn\'t', 'weren\'t', 'hasn\'t', 'haven\'t', 'hadn\'t',
            'doesn\'t', 'don\'t', 'didn\'t', 'won\'t', 'wouldn\'t', 'shan\'t', 'shouldn\'t',
            'can\'t', 'cannot', 'couldn\'t', 'mustn\'t', 'let\'s', 'that\'s', 'who\'s',
            'what\'s', 'here\'s', 'there\'s', 'when\'s', 'where\'s', 'why\'s', 'how\'s'
        }
    
    return [word for word in tokens if word not in stop_words]

def stem_word(word):
    """
    NLP TECHNIQUE 5: STEMMING
    Simple implementation of Porter stemming algorithm (simplified version)
    """
    # Handle some common suffix patterns
    if len(word) > 3:
        if word.endswith('ing'):
            return word[:-3]
        elif word.endswith('ed'):
            return word[:-2]
        elif word.endswith('s') and not word.endswith('ss'):
            return word[:-1]
        elif word.endswith('ly'):
            return word[:-2]
        elif word.endswith('ment'):
            return word[:-4]
    return word

def stem_tokens(tokens):
    """Apply stemming to a list of tokens"""
    return [stem_word(word) for word in tokens]

def extract_ngrams(tokens, n=2):
    """
    NLP TECHNIQUE 6: N-GRAM EXTRACTION
    Extract n-grams (sequences of n words) from tokens
    """
    ngrams = []
    for i in range(len(tokens) - n + 1):
        ngrams.append(' '.join(tokens[i:i+n]))
    return ngrams

def extract_text_statistics(text):
    """
    NLP TECHNIQUE 7: TEXT STATISTICS
    Extract statistical features from text
    """
    stats = {
        'char_count': len(text),
        'word_count': len(text.split()),
        'unique_word_count': len(set(text.split())),
        'sentence_count': len(re.split(r'[.!?]+', text)),
        'avg_word_length': sum(len(word) for word in text.split()) / max(len(text.split()), 1),
        'punctuation_count': sum(1 for char in text if char in string.punctuation),
        'uppercase_count': sum(1 for char in text if char.isupper()),
        'digit_count': sum(1 for char in text if char.isdigit()),
        'special_char_count': sum(1 for char in text if not char.isalnum() and not char.isspace())
    }
    return stats

def preprocess_text(text, include_ngrams=False):
    """
    MAIN PREPROCESSING FUNCTION
    Complete preprocessing pipeline combining multiple NLP techniques
    """
    # Apply text normalization (just lowercase and whitespace normalization)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Apply gentle text cleaning (keep some special characters that might be indicative of spam)
    # Remove URLs but keep the word 'http' as it might be indicative of spam
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', 'URL', text)
    
    # Replace email addresses with 'EMAIL'
    text = re.sub(r'[\w\.-]+@[\w\.-]+', 'EMAIL', text)
    
    # Replace phone numbers with 'PHONE'
    text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', 'PHONE', text)
    
    # Replace currency symbols and amounts with 'MONEY'
    text = re.sub(r'[$€£¥]\s*\d+([.,]\d+)?', 'MONEY', text)
    text = re.sub(r'\d+([.,]\d+)?\s*[$€£¥]', 'MONEY', text)
    
    # Keep exclamation and question marks as they might be indicative of spam
    text = re.sub(r'!+', ' ! ', text)
    text = re.sub(r'\?+', ' ? ', text)
    
    # Replace numbers with 'NUM'
    text = re.sub(r'\b\d+\b', 'NUM', text)
    
    # Remove remaining special characters
    text = re.sub(r'[^a-zA-Z0-9\s!?]', '', text)
    
    # Tokenize the text (simple split by whitespace)
    tokens = text.split()
    
    # Remove very short tokens (length < 2)
    tokens = [token for token in tokens if len(token) > 1]
    
    # Remove stopwords (but keep some that might be relevant for spam detection)
    important_words = {'free', 'win', 'won', 'prize', 'call', 'text', 'urgent', 'click'}
    stop_words = set(remove_stopwords([])) - important_words
    tokens = [token for token in tokens if token not in stop_words or token in important_words]
    
    # Apply light stemming (only for longer words)
    tokens = [stem_word(token) if len(token) > 3 else token for token in tokens]
    
    # Extract n-grams if requested
    if include_ngrams:
        # Only create bigrams for tokens that are not special tokens (URL, EMAIL, etc.)
        regular_tokens = [t for t in tokens if t not in {'URL', 'EMAIL', 'PHONE', 'MONEY', 'NUM'}]
        if len(regular_tokens) > 1:
            bigrams = extract_ngrams(regular_tokens, 2)
            # Add bigrams to tokens (but limit to avoid too many features)
            tokens.extend(bigrams[:5])
    
    # Join tokens back into a string
    processed_text = ' '.join(tokens)
    
    return processed_text

# =====================================================================
# FEATURE EXTRACTION TECHNIQUES
# =====================================================================

class TfidfVectorizer:
    """
    NLP TECHNIQUE 8: TF-IDF VECTORIZATION
    Converts text documents to a matrix of TF-IDF features
    """
    
    def __init__(self, max_features=None):
        """Initialize the TF-IDF vectorizer"""
        self.max_features = max_features
        self.vocabulary_ = {}  # Maps terms to indices
        self.idf_ = None  # IDF values for each term
        self.document_count = 0
    
    def fit(self, documents):
        """Learn vocabulary and IDF from training documents"""
        self.document_count = len(documents)
        
        # Count document frequency for each term
        df = defaultdict(int)
        for doc in documents:
            # Get unique terms in the document
            terms = set(doc.split())
            for term in terms:
                df[term] += 1
        
        # Sort terms by document frequency (descending)
        sorted_terms = sorted(df.items(), key=lambda x: x[1], reverse=True)
        
        # Limit to max_features if specified
        if self.max_features is not None and len(sorted_terms) > self.max_features:
            sorted_terms = sorted_terms[:self.max_features]
        
        # Create vocabulary mapping
        self.vocabulary_ = {term: idx for idx, (term, _) in enumerate(sorted_terms)}
        
        # Calculate IDF for each term
        self.idf_ = np.zeros(len(self.vocabulary_))
        for term, idx in self.vocabulary_.items():
            self.idf_[idx] = math.log(self.document_count / df[term]) + 1.0
        
        return self
    
    def transform(self, documents):
        """Transform documents to TF-IDF matrix"""
        if not self.vocabulary_:
            raise ValueError("Vectorizer needs to be fitted before transform")
        
        n_samples = len(documents)
        n_features = len(self.vocabulary_)
        X = np.zeros((n_samples, n_features))
        
        for doc_idx, doc in enumerate(documents):
            # Count term frequencies
            term_counts = Counter(doc.split())
            doc_len = len(doc.split())
            
            # Calculate TF-IDF for each term in the document
            for term, count in term_counts.items():
                if term in self.vocabulary_:
                    term_idx = self.vocabulary_[term]
                    # TF = term count / total terms in document
                    tf = count / doc_len
                    # TF-IDF = TF * IDF
                    X[doc_idx, term_idx] = tf * self.idf_[term_idx]
        
        return X
    
    def fit_transform(self, documents):
        """Learn vocabulary and IDF, then transform documents to TF-IDF matrix"""
        self.fit(documents)
        return self.transform(documents)
    
    def get_feature_names(self):
        """Get feature names (terms in the vocabulary)"""
        return [term for term, _ in sorted(self.vocabulary_.items(), key=lambda x: x[1])]

# =====================================================================
# CLASSIFICATION TECHNIQUES
# =====================================================================

class MultinomialNaiveBayes:
    """
    NLP TECHNIQUE 9: NAIVE BAYES CLASSIFICATION
    A Multinomial Naive Bayes classifier implementation for text classification
    """
    
    def __init__(self, alpha=1.0):
        """Initialize the classifier with smoothing parameter alpha"""
        self.alpha = alpha  # Smoothing parameter
        self.class_priors = {}  # P(c) for each class
        self.class_word_counts = {}  # Word counts for each class
        self.vocab = set()  # Vocabulary (all unique words)
        self.class_total_words = {}  # Total words in each class
        self.classes = []  # List of classes
    
    def fit(self, X, y):
        """Train the Naive Bayes classifier"""
        # Get unique classes
        self.classes = list(set(y))
        n_samples = len(X)
        
        # Calculate class priors P(c)
        class_counts = Counter(y)
        for c in self.classes:
            self.class_priors[c] = class_counts[c] / n_samples
        
        # Initialize word counts for each class
        for c in self.classes:
            self.class_word_counts[c] = defaultdict(int)
            self.class_total_words[c] = 0
        
        # Count word occurrences for each class
        for doc, label in zip(X, y):
            # Split document into words
            words = doc.split()
            
            # Update vocabulary
            self.vocab.update(words)
            
            # Count words for this class
            for word in words:
                self.class_word_counts[label][word] += 1
                self.class_total_words[label] += 1
        
        return self
    
    def predict(self, X):
        """Predict class labels for documents in X"""
        return [self._predict_single(doc) for doc in X]
    
    def _predict_single(self, doc):
        """Predict class for a single document"""
        # Split document into words
        words = doc.split()
        
        # Calculate log probabilities for each class
        log_probs = {}
        for c in self.classes:
            # Start with log of class prior
            log_probs[c] = math.log(self.class_priors[c])
            
            # Add log of conditional probabilities for each word
            for word in words:
                # Check if the word is in our vocabulary
                if word in self.vocab:
                    # Get word probability with Laplace smoothing
                    # P(word|class) = (count(word, class) + alpha) / (total_words_in_class + alpha * vocab_size)
                    word_count = self.class_word_counts[c][word]
                    word_prob = (word_count + self.alpha) / (self.class_total_words[c] + self.alpha * len(self.vocab))
                    log_probs[c] += math.log(word_prob)
                else:
                    # For unknown words, use a small probability based on smoothing
                    word_prob = self.alpha / (self.class_total_words[c] + self.alpha * len(self.vocab))
                    log_probs[c] += math.log(word_prob)
            
            # Add a weight to spam class to improve detection (slight bias towards spam detection)
            if c == 1:  # Assuming 1 is the spam class
                log_probs[c] += 0.1  # Small bias towards spam detection
        
        # Return class with highest log probability
        return max(log_probs, key=log_probs.get)
    
    def predict_proba(self, X):
        """Predict class probabilities for documents in X"""
        return [self._predict_proba_single(doc) for doc in X]
    
    def _predict_proba_single(self, doc):
        """Predict class probabilities for a single document"""
        # Split document into words
        words = doc.split()
        
        # Calculate log probabilities for each class
        log_probs = {}
        for c in self.classes:
            # Start with log of class prior
            log_probs[c] = math.log(self.class_priors[c])
            
            # Add log of conditional probabilities for each word
            for word in words:
                # Check if the word is in our vocabulary
                if word in self.vocab:
                    # Get word probability with Laplace smoothing
                    word_count = self.class_word_counts[c][word]
                    word_prob = (word_count + self.alpha) / (self.class_total_words[c] + self.alpha * len(self.vocab))
                    log_probs[c] += math.log(word_prob)
                else:
                    # For unknown words, use a small probability based on smoothing
                    word_prob = self.alpha / (self.class_total_words[c] + self.alpha * len(self.vocab))
                    log_probs[c] += math.log(word_prob)
            
            # Add a weight to spam class to improve detection (slight bias towards spam detection)
            if c == 1:  # Assuming 1 is the spam class
                log_probs[c] += 0.1  # Small bias towards spam detection
        
        # Convert log probabilities to actual probabilities
        # First, find the maximum log probability to avoid numerical issues
        max_log_prob = max(log_probs.values())
        
        # Subtract the max and exponentiate
        probs = {c: math.exp(log_prob - max_log_prob) for c, log_prob in log_probs.items()}
        
        # Normalize to get probabilities that sum to 1
        total = sum(probs.values())
        return {c: prob / total for c, prob in probs.items()}
    
    def get_feature_importance(self, top_n=20):
        """Get the most important features (words) for each class"""
        feature_importance = {}
        
        for c in self.classes:
            # Calculate importance score for each word
            # Score = P(word|class) / P(word|not_class)
            word_scores = {}
            
            for word in self.vocab:
                # P(word|class)
                p_word_given_class = (self.class_word_counts[c][word] + self.alpha) / (
                    self.class_total_words[c] + self.alpha * len(self.vocab))
                
                # P(word|not_class) - combine all other classes
                not_class_word_count = sum(self.class_word_counts[other_c][word] 
                                          for other_c in self.classes if other_c != c)
                not_class_total_words = sum(self.class_total_words[other_c] 
                                           for other_c in self.classes if other_c != c)
                
                p_word_given_not_class = (not_class_word_count + self.alpha) / (
                    not_class_total_words + self.alpha * len(self.vocab))
                
                # Avoid division by zero
                if p_word_given_not_class > 0:
                    word_scores[word] = p_word_given_class / p_word_given_not_class
                else:
                    word_scores[word] = float('inf')
            
            # Get top N words by score
            top_words = sorted(word_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
            feature_importance[c] = top_words
        
        return feature_importance

# =====================================================================
# EVALUATION METRICS
# =====================================================================

def accuracy_score(y_true, y_pred):
    """
    NLP TECHNIQUE 10: EVALUATION METRICS
    Calculate accuracy score
    """
    if len(y_true) != len(y_pred):
        raise ValueError("Length of y_true and y_pred must be the same")
    
    correct = sum(1 for true, pred in zip(y_true, y_pred) if true == pred)
    return correct / len(y_true)

def precision_score(y_true, y_pred, pos_label=1):
    """Calculate precision score"""
    if len(y_true) != len(y_pred):
        raise ValueError("Length of y_true and y_pred must be the same")
    
    true_positives = sum(1 for true, pred in zip(y_true, y_pred) 
                         if true == pos_label and pred == pos_label)
    predicted_positives = sum(1 for pred in y_pred if pred == pos_label)
    
    if predicted_positives == 0:
        return 0.0
    
    return true_positives / predicted_positives

def recall_score(y_true, y_pred, pos_label=1):
    """Calculate recall score"""
    if len(y_true) != len(y_pred):
        raise ValueError("Length of y_true and y_pred must be the same")
    
    true_positives = sum(1 for true, pred in zip(y_true, y_pred) 
                         if true == pos_label and pred == pos_label)
    actual_positives = sum(1 for true in y_true if true == pos_label)
    
    if actual_positives == 0:
        return 0.0
    
    return true_positives / actual_positives

def f1_score(y_true, y_pred, pos_label=1):
    """Calculate F1 score"""
    precision = precision_score(y_true, y_pred, pos_label)
    recall = recall_score(y_true, y_pred, pos_label)
    
    if precision + recall == 0:
        return 0.0
    
    return 2 * (precision * recall) / (precision + recall)

def confusion_matrix(y_true, y_pred, labels=None):
    """Calculate confusion matrix"""
    if len(y_true) != len(y_pred):
        raise ValueError("Length of y_true and y_pred must be the same")
    
    if labels is None:
        labels = sorted(list(set(y_true) | set(y_pred)))
    
    n_labels = len(labels)
    label_to_index = {label: i for i, label in enumerate(labels)}
    
    # Initialize confusion matrix
    cm = np.zeros((n_labels, n_labels), dtype=int)
    
    # Fill confusion matrix
    for true, pred in zip(y_true, y_pred):
        if true in label_to_index and pred in label_to_index:
            cm[label_to_index[true], label_to_index[pred]] += 1
    
    return cm
