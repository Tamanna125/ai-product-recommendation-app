import React, { useState, useEffect } from 'react'
import './AnalyticsPage.css'
import axios from 'axios'
// Import the charting library we installed
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
} from 'recharts'
import './AnalyticsPage.css' // We'll create this file next

function AnalyticsPage() {
  const [analytics, setAnalytics] = useState(null)
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState('')

  // This runs once when the component loads
 // This runs once when the component loads
  // This runs once when the component loads
  useEffect(() => {
    const fetchAnalytics = async () => {
      try {
        setIsLoading(true); // Set loading to true
        const response = await axios.get('http://127.0.0.1:8000/analytics');

        // --- START OF FIX ---
        // Check for a backend-specific error message FIRST
        if (response.data.error) {
          throw new Error(response.data.error);
        }

        // NOW it's safe to process the data
        const categories = response.data.top_categories
          ? Object.entries(response.data.top_categories).map(
              ([name, value]) => ({ name, value })
            )
          : [];

        const brands = response.data.top_brands
          ? Object.entries(response.data.top_brands).map(
              ([name, value]) => ({ name, value })
            )
          : [];

        const formattedData = {
          ...response.data,
          top_categories: categories,
          top_brands: brands,
        };
        // --- END OF FIX ---

        setAnalytics(formattedData);
      } catch (err) {
        console.error(err);
        setError(
          err.message || 'Error fetching analytics. Is the backend server running?'
        );
      }
      setIsLoading(false); // Set loading to false in all cases
    };

    fetchAnalytics();
  }, []); // The empty array [] means this effect runs only once

  if (isLoading) return <p>Loading analytics...</p>
  if (error) return <p className="error">{error}</p>
  if (!analytics) return <p>No analytics data.</p>

  const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884D8']

  
  return (
    <div className="analytics-page">
      <h1>Product Analytics</h1> 
      <div className="stats-header">
        {/* FIX: Use "?." and "?? 'N/A'" to prevent crash if data is missing */}
        <h2>Total Products: {analytics?.total_products ?? 'N/A'}</h2>
        <p>
          {/* FIX: Safely access 'mean' and 'toFixed' */}
          Average Price: $
          {analytics?.price_distribution?.mean?.toFixed(2) ?? '0.00'}
        </p>
      </div>

      <div className="charts-container">
        {/* Top Categories Chart */}
        <div className="chart-wrapper">
          <h3>Top 10 Categories</h3>
          <ResponsiveContainer width="100%" height={300}>
            {/* FIX: Check if data exists before rendering chart */}
            {analytics?.top_categories?.length > 0 ? (
              <BarChart data={analytics.top_categories}>
                <XAxis dataKey="name" hide />
                <YAxis />
                <Tooltip />
                <Legend />
                <Bar dataKey="value" fill="#8884d8" />
              </BarChart>
            ) : (
              <p>No category data available.</p>
            )}
          </ResponsiveContainer>
        </div>

        {/* Top Brands Chart */}
        <div className="chart-wrapper">
          <h3>Top 10 Brands</h3>
          <ResponsiveContainer width="100%" height={300}>
            {/* FIX: Check if data exists before rendering chart */}
            {analytics?.top_brands?.length > 0 ? (
              <PieChart>
                <Pie
                  data={analytics.top_brands}
                  cx="50%"
                  cy="50%"
                  labelLine={false}
                  outerRadius={100}
                  fill="#8884d8"
                  dataKey="value"
                  label={({ name, percent }) =>
                    `${name} ${(percent * 100).toFixed(0)}%`
                  }
                >
                  {analytics.top_brands.map((entry, index) => (
                    <Cell
                      key={`cell-${index}`}
                      fill={COLORS[index % COLORS.length]}
                    />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            ) : (
              <p>No brand data available.</p>
            )}
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  )
}

export default AnalyticsPage