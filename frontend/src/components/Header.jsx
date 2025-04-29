import React from 'react';

const Header = () => {
  return (
    <header className="bg-dark text-white py-3 mb-4">
      <div className="container">
        <div className="row align-items-center">
          <div className="col-12">
            <h1 className="mb-0">SMS Spam Detection</h1>
            <p className="mb-0">Classify messages as spam or ham using NLP</p>
          </div>

        </div>
      </div>
    </header>
  );
};

export default Header;
