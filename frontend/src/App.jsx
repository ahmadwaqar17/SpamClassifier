import React, { useState, useEffect } from 'react'
import 'bootstrap/dist/css/bootstrap.min.css'
import './App.css'
import Header from './components/Header'

import SpamClassifier from './components/SpamClassifier'
import axios from 'axios'

function App() {
  const [serverStatus, setServerStatus] = useState('checking')

  useEffect(() => {
    // Check if the backend server is running
    const checkServerStatus = async () => {
      try {
        await axios.get('http://localhost:5000/api/health')
        setServerStatus('online')
      } catch (error) {
        setServerStatus('offline')
      }
    }

    checkServerStatus()
    // Check every 10 seconds
    const interval = setInterval(checkServerStatus, 10000)

    return () => clearInterval(interval)
  }, [])

  return (
    <div className="d-flex flex-column min-vh-100">
      <Header />

      <main className="flex-grow-1 py-4">
        {serverStatus === 'offline' && (
          <div className="container mb-4">
            <div className="alert alert-danger">
              <strong>Backend server is offline!</strong> Please start the Flask server to use the application.
            </div>
          </div>
        )}

        <SpamClassifier />
      </main>
    </div>
  )
}

export default App
