 # SMS Spam Detection Web Application

A modern web application for classifying SMS messages as spam or ham using advanced Natural Language Processing (NLP) techniques implemented from scratch.

## Project Overview

This project implements a spam detection system that uses machine learning and Natural Language Processing (NLP) techniques to classify SMS messages as either spam or legitimate (ham). The application includes both a backend Flask API for classification and a React frontend for user interaction.

The system achieves high accuracy (98.21%) in distinguishing between spam and legitimate messages by implementing various NLP techniques from scratch, including text preprocessing, feature extraction with TF-IDF, and classification using a custom Naive Bayes algorithm.


## Features

- **Advanced Text Preprocessing**: Custom implementation of tokenization, normalization, cleaning, stopword removal, and stemming
- **Intelligent Feature Extraction**: TF-IDF vectorization with custom implementation to identify important words
- **Custom Naive Bayes Classifier**: Built from scratch with Laplace smoothing and log probability calculations
- **Comprehensive Evaluation Metrics**: Accuracy, precision, recall, F1-score, and confusion matrix
- **Feature Importance Analysis**: Identification of words most indicative of spam or ham
- **Interactive Web Interface**: User-friendly interface for message classification
- **Real-time Classification**: Instant results with probability scores and important features
- **Example Messages**: Pre-loaded spam and ham examples for testing

## Technologies Used

### Backend
- **Python**: Core programming language
- **Flask**: Lightweight web framework for API endpoints
- **NumPy**: Numerical computing for vector operations
- **Pandas**: Data manipulation for dataset handling
- **Custom NLP Implementation**: All NLP techniques implemented from scratch
  - Text preprocessing (normalization, cleaning, tokenization)
  - TF-IDF vectorization
  - Naive Bayes classification
  - Evaluation metrics

### Frontend
- **React**: JavaScript library for building the user interface
- **Vite**: Next-generation frontend build tool
- **Bootstrap**: CSS framework for responsive design
- **Axios**: Promise-based HTTP client for API requests

## Project Structure

```
├── api_server.py            # Flask API server with endpoints
├── spam_detection_service.py # Spam detection service orchestrating the classification process
├── nlp_engine.py            # Core NLP algorithms and techniques implementation
├── requirements.txt        # Python dependencies
├── spam.csv                # Dataset
├── frontend/               # React frontend
│   ├── src/
│   │   ├── components/     # React components
│   │   │   ├── Header.jsx
│   │   │   ├── Footer.jsx
│   │   │   ├── SpamClassifier.jsx
│   │   │   └── ResultDisplay.jsx
│   │   ├── App.jsx         # Main application component
│   │   └── App.css         # Styles
```

The backend code is organized into just 3 files with clear, descriptive names for better maintainability:

1. **api_server.py**: Flask API server providing endpoints for the frontend to interact with
2. **spam_detection_service.py**: Service layer that orchestrates the entire spam detection process
3. **nlp_engine.py**: Core implementation of all NLP algorithms and techniques used in the project

## Project Flow

### Backend Flow

1. **Data Loading and Preparation**:
   - The SMS dataset is loaded from `spam.csv`
   - Messages are labeled as ham (0) or spam (1)
   - The dataset is split into training and testing sets

2. **Text Preprocessing**:
   - Messages undergo normalization (lowercase, whitespace removal)
   - Special patterns (URLs, emails, phone numbers) are identified and replaced with tokens
   - Text is tokenized into individual words
   - Stopwords are removed (except for spam-indicative words like "free", "win", etc.)
   - Words are stemmed to their root form
   - N-grams (word pairs) are extracted to capture phrases

3. **Feature Extraction**:
   - TF-IDF vectorization converts text to numerical features
   - Term Frequency (TF) measures how often a word appears in a message
   - Inverse Document Frequency (IDF) measures how rare a word is across all messages
   - TF-IDF combines these to give higher weight to important, distinctive words

4. **Model Training**:
   - A custom Multinomial Naive Bayes classifier is trained on the features
   - The model learns word probabilities for each class (spam/ham)
   - Laplace smoothing handles unseen words
   - Log probabilities prevent numerical underflow

5. **Model Evaluation**:
   - The model is evaluated on the test set
   - Performance metrics (accuracy, precision, recall, F1) are calculated
   - A confusion matrix shows the breakdown of predictions
   - Feature importance analysis identifies key spam/ham indicators

6. **API Endpoints**:
   - Flask provides RESTful API endpoints
   - `/api/classify` classifies new messages
   - `/api/train` retrains the model
   - `/api/nlp-techniques` provides information about the NLP techniques used

### Frontend Flow

1. **User Interface**:
   - React components render the user interface
   - Users can enter messages or select examples
   - The interface communicates with the backend API

2. **Classification Process**:
   - User enters a message and clicks "Classify"
   - The message is sent to the backend API
   - The backend preprocesses the message, extracts features, and makes a prediction
   - The result is returned to the frontend

3. **Result Display**:
   - Classification result (spam/ham) is displayed
   - Probability scores show confidence in the prediction
   - Important features that contributed to the classification are highlighted

## Setup and Installation

### Backend Setup

1. Install Python dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Run the Flask server:
   ```
   python app.py
   ```

### Frontend Setup

1. Navigate to the frontend directory:
   ```
   cd frontend
   ```

2. Install Node.js dependencies:
   ```
   npm install¬
   ```

3. Start the development server:
   ```
   npm run dev
   ```

4. Open your browser and navigate to the URL shown in the terminal (typically http://localhost:5173)

## How to Use

### Installation

1. **Clone the repository**:
   ```
   git clone <repository-url>
   cd sms-spam-detection
   ```

2. **Install backend dependencies**:
   ```
   pip install -r requirements.txt
   ```

3. **Install frontend dependencies**:
   ```
   cd frontend
   npm install
   cd ..
   ```

### Running the Application

1. **Start the Backend Server**:
   ```
   python api_server.py
   ```
   The first time you run this, it will train the model which may take a few minutes. The server will be available at http://localhost:5000.

2. **Start the Frontend Server**:
   ```
   cd frontend
   npm run dev
   ```
   The frontend will be available at http://localhost:5173 (or the URL shown in the terminal).

3. **Access the Web Interface**:
   Open your browser and navigate to http://localhost:5173.

### Troubleshooting

- **Backend Server Offline**: If you see "Backend server is offline" message, make sure the Flask server is running on port 5000.
- **Model Training Issues**: If the model fails to train, check that the spam.csv file is in the root directory.
- **Frontend Connection Issues**: Ensure the backend URL in the frontend code is set to http://localhost:5000.

### Using the Classifier

1. **Enter a Message**:
   - Type a message in the text area, or
   - Click on one of the example messages provided

2. **Classify the Message**:
   - Click the "Classify Message" button
   - Wait for the classification result (usually instant)

3. **Interpret the Results**:
   - **Classification**: The message will be classified as either SPAM or HAM (not spam)
   - **Probability Scores**: See the confidence level of the classification
   - **Important Features**: View the words/phrases that contributed most to the classification

4. **Try Different Messages**:
   - Test with obvious spam messages (e.g., "You've won a prize! Call now!")
   - Test with legitimate messages (e.g., "I'll meet you at 5pm")
   - Observe how the classifier handles different types of content

### Understanding the Results

- **High Spam Probability** (>90%): The message is very likely to be spam
- **Medium Spam Probability** (50-90%): The message has some spam-like characteristics
- **Low Spam Probability** (<50%): The message is likely legitimate
- **Important Features**: Words highlighted in the results show what influenced the classification

## NLP Techniques Implemented

This project implements several advanced NLP techniques from scratch to demonstrate core principles:

### 1. Text Preprocessing (`nlp_techniques.py`)

- **Text Normalization**: Converting text to lowercase and removing extra whitespace
- **Intelligent Text Cleaning**:
  - Preserving spam-indicative patterns by replacing them with tokens (URL, EMAIL, PHONE, MONEY)
  - Keeping exclamation and question marks which are often indicative of spam
  - Replacing numbers with a NUM token
- **Tokenization**: Breaking text into individual words
- **Smart Stopword Removal**:
  - Filtering common words with low semantic value
  - Preserving important words like "free", "win", "prize" that are indicative of spam
- **Selective Stemming**:
  - Reducing words to their root form
  - Only applying stemming to longer words to preserve meaning
- **N-gram Extraction**: Capturing sequences of words as features to identify phrases

### 2. Feature Extraction (`nlp_techniques.py`)

- **TF-IDF Vectorization**: Custom implementation of Term Frequency-Inverse Document Frequency
  - **Term Frequency (TF)**: How often a word appears in a message
  - **Inverse Document Frequency (IDF)**: How rare a word is across all messages
  - **TF-IDF Weighting**: Giving higher weight to important, distinctive words
- **Feature Importance Analysis**:
  - Identifying which words are most indicative of spam or ham
  - Calculating importance scores based on conditional probabilities

### 3. Classification (`nlp_techniques.py`)

- **Multinomial Naive Bayes**: Custom implementation of the probabilistic classifier
  - **Bayes' Theorem**: P(class|text) ∝ P(text|class) × P(class)
  - **Conditional Probability**: Calculating P(word|class) for each word and class
  - **Laplace Smoothing**: Handling zero probabilities for unseen words
  - **Log Probability**: Using log probabilities to avoid numerical underflow
  - **Spam Detection Bias**: Slight bias towards spam detection to improve recall
- **Prediction Explanation**:
  - Showing how each word contributes to the classification
  - Identifying the most important features for each prediction

### 4. Evaluation Metrics (`nlp_techniques.py`)

- **Accuracy**: Proportion of correctly classified messages (98.21%)
- **Precision**: Proportion of predicted spam messages that are actually spam (96.43%)
- **Recall**: Proportion of actual spam messages correctly identified (90.00%)
- **F1-Score**: Harmonic mean of precision and recall (93.10%)
- **Confusion Matrix**: Detailed breakdown of prediction results

### 5. Integration (`spam_classifier.py`)

- **End-to-End Pipeline**: Combines all NLP techniques into a cohesive workflow
- **Model Training**: Handles the complete training process
- **Feature Engineering**: Converts raw text to meaningful features
- **Performance Analysis**: Evaluates and reports model performance
- **Prediction Service**: Provides classification for new messages

## Implementation Details

### File Structure and Responsibilities

1. **`api_server.py`**
   - Provides RESTful API endpoints for the frontend
   - Handles HTTP requests and responses
   - Routes requests to the spam detection service
   - Endpoints: `/api/classify`, `/api/train`, `/api/nlp-techniques`, `/api/health`

2. **`spam_detection_service.py`**
   - Orchestrates the spam detection process
   - Manages data loading and preprocessing
   - Coordinates model training and evaluation
   - Handles prediction requests
   - Extracts important features for classification

3. **`nlp_engine.py`**
   - Implements core NLP algorithms from scratch
   - Contains text preprocessing functions
   - Implements TF-IDF vectorization
   - Provides Naive Bayes classification
   - Includes evaluation metrics

### NLP Pipeline

1. **Text Preprocessing**
   - Text normalization (lowercase, whitespace removal)
   - Intelligent cleaning (preserving spam indicators like URLs, emails)
   - Tokenization (splitting text into words)
   - Stopword removal (filtering common words)
   - Stemming (reducing words to their root form)
   - N-gram extraction (capturing word sequences)

2. **Feature Extraction**
   - TF-IDF vectorization
   - Term frequency calculation
   - Inverse document frequency calculation
   - Feature weighting

3. **Classification**
   - Naive Bayes algorithm implementation
   - Conditional probability calculation
   - Laplace smoothing for unseen words
   - Log probability to prevent underflow
   - Feature importance analysis

4. **Evaluation**
   - Accuracy measurement (98.21%)
   - Precision calculation (96.43%)
   - Recall determination (90.00%)
   - F1-score computation (93.10%)
   - Confusion matrix analysis

## Dataset

The application uses the SMS Spam Collection dataset, which contains labeled SMS messages classified as spam or ham. The dataset contains 5,574 messages, with 747 spam messages (13.41%) and 4,827 ham messages (86.59%).

## License

This project is created for educational purposes as part of an NLP course project.
