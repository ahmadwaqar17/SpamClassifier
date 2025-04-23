import React, { useState } from 'react';
import axios from 'axios';
import ResultDisplay from './ResultDisplay';

const SpamClassifier = () => {
  const [message, setMessage] = useState('');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (!message.trim()) {
      setError('Please enter a message to classify');
      return;
    }
    
    setLoading(true);
    setError('');
    
    try {
      const response = await axios.post('http://localhost:5000/api/classify', {
        message: message
      });
      
      setResult(response.data);
    } catch (err) {
      console.error('Error classifying message:', err);
      setError('Error classifying message. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const handleClear = () => {
    setMessage('');
    setResult(null);
    setError('');
  };

  const handleExampleClick = (example) => {
    setMessage(example);
  };

  // Example messages
  const spamExamples = [
    "WINNER!! As a valued network customer you have been selected to receive a £900 prize reward!",
    "Free entry in 2 a weekly comp to win FA Cup final tickets 21st May 2005.",
    "URGENT! You have won a 1 week FREE membership in our £100,000 prize Jackpot!"
  ];
  
  const hamExamples = [
    "I'll be there in 10 minutes. Wait for me.",
    "Can you pick up some groceries on your way home?",
    "The meeting has been rescheduled to tomorrow at 2pm."
  ];

  return (
    <div className="container">
      <div className="card shadow">
        <div className="card-body">
          <h2 className="card-title mb-4">Message Classifier</h2>
          
          <form onSubmit={handleSubmit}>
            <div className="mb-3">
              <label htmlFor="message" className="form-label">Enter a message to classify:</label>
              <textarea
                id="message"
                className="form-control"
                rows="5"
                value={message}
                onChange={(e) => setMessage(e.target.value)}
                placeholder="Type or paste a message here..."
              ></textarea>
            </div>
            
            {error && <div className="alert alert-danger">{error}</div>}
            
            <div className="d-flex gap-2 mb-4">
              <button type="submit" className="btn btn-primary" disabled={loading}>
                {loading ? 'Classifying...' : 'Classify Message'}
              </button>
              <button type="button" className="btn btn-secondary" onClick={handleClear}>
                Clear
              </button>
            </div>
          </form>
          
          {result && <ResultDisplay result={result} message={message} />}
          
          <div className="mt-4">
            <h5>Try with examples:</h5>
            <div className="row">
              <div className="col-md-6">
                <div className="card mb-3">
                  <div className="card-header bg-danger text-white">
                    Spam Examples
                  </div>
                  <div className="card-body">
                    <ul className="list-group">
                      {spamExamples.map((example, index) => (
                        <li key={`spam-${index}`} className="list-group-item">
                          <button 
                            className="btn btn-link text-danger p-0 text-decoration-none"
                            onClick={() => handleExampleClick(example)}
                          >
                            {example.length > 50 ? example.substring(0, 50) + '...' : example}
                          </button>
                        </li>
                      ))}
                    </ul>
                  </div>
                </div>
              </div>
              <div className="col-md-6">
                <div className="card mb-3">
                  <div className="card-header bg-success text-white">
                    Ham Examples
                  </div>
                  <div className="card-body">
                    <ul className="list-group">
                      {hamExamples.map((example, index) => (
                        <li key={`ham-${index}`} className="list-group-item">
                          <button 
                            className="btn btn-link text-success p-0 text-decoration-none"
                            onClick={() => handleExampleClick(example)}
                          >
                            {example}
                          </button>
                        </li>
                      ))}
                    </ul>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default SpamClassifier;
