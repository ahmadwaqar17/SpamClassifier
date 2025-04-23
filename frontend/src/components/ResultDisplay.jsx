import React from 'react';

const ResultDisplay = ({ result, message }) => {
  const { is_spam, spam_probability, ham_probability, important_features = [] } = result;

  // Format probabilities as percentages
  const spamPercentage = (spam_probability * 100).toFixed(2);
  const hamPercentage = (ham_probability * 100).toFixed(2);

  return (
    <div className="card mb-4">
      <div className={`card-header ${is_spam ? 'bg-danger' : 'bg-success'} text-white`}>
        <h5 className="mb-0">
          Classification Result: {is_spam ? 'SPAM' : 'HAM (Not Spam)'}
        </h5>
      </div>
      <div className="card-body">
        <div className="mb-3">
          <h6>Message:</h6>
          <p className="border p-2 rounded bg-light">{message}</p>
        </div>

        <div className="row">
          <div className="col-md-6">
            <div className="card mb-3">
              <div className="card-body">
                <h6 className="card-title text-danger">Spam Probability</h6>
                <div className="progress">
                  <div
                    className="progress-bar bg-danger"
                    role="progressbar"
                    style={{ width: `${spamPercentage}%` }}
                    aria-valuenow={spamPercentage}
                    aria-valuemin="0"
                    aria-valuemax="100"
                  >
                    {spamPercentage}%
                  </div>
                </div>
              </div>
            </div>
          </div>
          <div className="col-md-6">
            <div className="card mb-3">
              <div className="card-body">
                <h6 className="card-title text-success">Ham Probability</h6>
                <div className="progress">
                  <div
                    className="progress-bar bg-success"
                    role="progressbar"
                    style={{ width: `${hamPercentage}%` }}
                    aria-valuenow={hamPercentage}
                    aria-valuemin="0"
                    aria-valuemax="100"
                  >
                    {hamPercentage}%
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>

        {important_features.length > 0 && (
          <div className="card mb-3">
            <div className="card-header bg-primary text-white">
              <h6 className="mb-0">Important Features</h6>
            </div>
            <div className="card-body">
              <p className="mb-2">These words/phrases contributed significantly to the classification:</p>
              <div className="d-flex flex-wrap gap-2">
                {important_features.map((feature, index) => (
                  <span key={index} className={`badge ${is_spam ? 'bg-danger' : 'bg-success'} p-2`}>
                    {feature}
                  </span>
                ))}
              </div>
            </div>
          </div>
        )}

        <div className="alert alert-info">
          <h6 className="alert-heading">What does this mean?</h6>
          {is_spam ? (
            <p className="mb-0">
              This message has been classified as <strong>spam</strong> with {spamPercentage}% confidence.
              It contains patterns commonly found in unsolicited or fraudulent messages.
              {important_features.length > 0 && (
                <span> The highlighted words/phrases above are typical indicators of spam content.</span>
              )}
            </p>
          ) : (
            <p className="mb-0">
              This message has been classified as <strong>ham (not spam)</strong> with {hamPercentage}% confidence.
              It appears to be a legitimate message.
              {important_features.length > 0 && (
                <span> The highlighted words/phrases above are typical indicators of legitimate content.</span>
              )}
            </p>
          )}
        </div>
      </div>
    </div>
  );
};

export default ResultDisplay;
