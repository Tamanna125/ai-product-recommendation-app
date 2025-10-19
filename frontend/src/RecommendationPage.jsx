import React, { useState } from 'react'
import './RecommendationPage.css'
import axios from 'axios' // Used to make API calls
import './RecommendationPage.css' // We'll create this file next for styling

function RecommendationPage() {
  const [prompt, setPrompt] = useState('')
  const [results, setResults] = useState([])
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState('')

  // This function is called when the user clicks "Send"
  const handleSubmit = async (e) => {
    e.preventDefault() // Prevents the page from reloading
    setIsLoading(true)
    setError('')
    setResults([])

    try {
      // Call your FastAPI backend /recommend endpoint
      const response = await axios.post('http://127.0.0.1:8000/recommend', {
        prompt: prompt,
      })

      if (response.data.recommendations) {
        setResults(response.data.recommendations)
      } else {
        setError('No recommendations found.')
      }
    } catch (err) {
      console.error(err)
      setError('Error fetching recommendations. Is the backend server running?')
    }

    setIsLoading(false)
    setPrompt('') // Clear the input box
  }

  return (
    <div className="recommendation-page">
      <h1>AI Product Recommender</h1>
      <p>
  Type what you're looking for, e.g., "a comfy chair for my living room" or
  "modern patio furniture"
</p>

      {/* The input form */}
      <form onSubmit={handleSubmit}>
        <input
          type="text"
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          placeholder="Tell me what you want..."
        />
        <button type="submit" disabled={isLoading}>
          {isLoading ? 'Thinking...' : 'Send'}
        </button>
      </form>

      {/* Display errors, if any */}
      {error && <p className="error">{error}</p>}

      {/* Display the results */}
     <div className="results-grid">
        {results.map((product) => (
          <div key={product.id} className="product-card">
            {/* FIX 1: Check for an image_url. If it's missing,
              show a gray placeholder div instead.
            */}
            {product.image_url ? (
              <img src={product.image_url} alt={product.title} />
            ) : (
              <div className="img-placeholder"></div>
            )}

            <h3>{product.title}</h3>
            
            {/* FIX 2: Only show the price if it exists.
            */}
            {product.price && <p className="price">${product.price}</p>}

            {/* This displays the AI-generated description! */}
            <p className="generated-desc">
              <strong>AI Description:</strong> {product.generated_description}
            </p>
          </div>
        ))}
      </div>
    </div>
  )
}

export default RecommendationPage