import React, { useState } from "react";
import axios from "axios";

// User interacts with the search bar to find similar products 
// and get a verdict on the trendiness of the product
// this is based off the data from the Django backend and the Asos API
const SearchProduct = () => {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState([]);
  const [verdict, setVerdict] = useState(null);
  const [error, setError] = useState("");

  const handleSearch = async () => {
    if (!query) return;
    //backend url for the view that handles the similarity and overall trendiness of the product
    const baseUrl = "http://127.0.0.1:8000/api/similar/";


    try {
        //get the data from the backend based on the query
        const response = await axios.get(baseUrl, {
            params: { q: query },
        });
        const data = response.data;

        //set the results and verdict based on the data
        setResults(data.matched_products || []);
        setVerdict({
            average_trendy: data.is_overwhelmingly_trendy,
            proportion: data.trendy_proportion
        });
        setError("");
    } catch (error) {
        //error handling
        console.error("Search Error:", error.message);
        setError("No similar products found.");
        setResults([]);
        setVerdict(null);
    }
  };

  return (
    //display the search bar and the results
    <div className="search-container">
      <h2>Search for a Product</h2>
      <input
            type="text"
            placeholder="e.g. loafers in black"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            
        />
    {/* button handling */}
      <button onClick={handleSearch} >
        Search
      </button>

      |{/* error message displayed */}

      {error && <p >{error}</p>}

      {verdict && (
        <div>
            <h4>Verdict:</h4>
          {verdict.average_trendy ? (
            <p>This style is trendy.</p>
          ) : (
            <p> This style is not currently trendy.</p>
          )}
          <p>Trendy Rate: {
                (verdict.proportion * 100)
            }%
          </p>
          
        </div>
      )}

    </div>
  );
};

export default SearchProduct;
