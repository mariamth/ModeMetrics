//handles the search bar and the results page
import React, { useState } from "react";
import axios from "axios";
import "./search.css";

const SearchProduct = ({ showResults, setShowResults }) => {
  const [query, setQuery] = useState("");
  const [verdict, setVerdict] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const handleSearch = async () => {
    if (!query) {
      setError("Please enter a search term.");
      return;
    }
    setLoading(true);
    setError("");
    setVerdict(null);
    try {
      //makes a GET request to the backend with the query
      const response = await axios.get("http://127.0.0.1:8000/api/similar/", {
        params: { q: query },
      });
      const data = response.data;
      setVerdict({
        //returns the verdict based on the response
        average_trendy: data.is_overwhelmingly_trendy,
        proportion: data.trendy_proportion,
        advice: data.advice,
      });
      setShowResults(true);
    } catch (error) {
      setError("No similar products found.");
    }
    setLoading(false);
  };

  const handleBack = () => {
    setShowResults(false);
    setVerdict(null);
    setQuery("");
    setError("");
  };

  return (
    <div className="search-section">
      {/* search bar and results display functionality */}
      {!showResults && (
        <>
          <div className="search-bar">
            <input
              id="search-input"
              type="text"
              placeholder="e.g. loafers in black"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && handleSearch()}
            />
          </div>
          <button
            className="search-btn"
            onClick={handleSearch}
            disabled={loading}
          >
            {loading ? "Searching" : "Search"}
          </button>
          {error && <p className="error-msg">{error}</p>}
        </>
      )}

      {/* results display */}
      {showResults && verdict && (
        <div className="verdict-container">
          <h3 className="verdict-title">Verdict:</h3>
          <p className="verdict-main">
            {verdict.average_trendy
              ? "This style is trendy."
              : "This style is not trendy at the moment."}
          </p>
          <p className="verdict-trend">
            Trendy Rate: {(verdict.proportion * 100).toFixed(1)}%
          </p>
          {verdict.advice && (
            <p className="verdict-conclusion">
              <b>Advice:</b> {verdict.advice}
            </p>
          )}
          <button className="search-btn back-btn" onClick={handleBack}>
            Back to search
          </button>
        </div>
      )}
    </div>
  );
};

export default SearchProduct;
