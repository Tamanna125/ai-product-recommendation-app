import React from 'react'
import './App.css'
import { Routes, Route, Link } from 'react-router-dom'
import './App.css'

// We will create these two component files in the next step
import RecommendationPage from './RecommendationPage'
import AnalyticsPage from './AnalyticsPage'

function App() {
  return (
    <div className="App">
      {/* Navigation Bar */}
      <nav>
        <Link to="/">Recommendations</Link>
        <Link to="/analytics">Analytics</Link>
      </nav>

      {/* Route Definitions */}
      <Routes>
        <Route path="/" element={<RecommendationPage />} />
        <Route path="/analytics" element={<AnalyticsPage />} />
      </Routes>
    </div>
  )
}

export default App