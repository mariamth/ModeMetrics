import React, { useState, useEffect } from "react";
import './App.css';
import axios from "axios";
import SearchProduct from "./components/searchProduct";
import Header from "./components/header.js";
import Footer from "./components/footer.js";


function App() {
  useEffect(() => {
    //check if predictions have already been made
    const alreadyPredicted = sessionStorage.getItem("predicted");
    //if not, make a request to the prediction endpoint
    //and store the result in sessionStorage
    //to avoid making the request again when the page is refreshed
    if (!alreadyPredicted) {
      axios.get("http://127.0.0.1:8000/api/predict/")
        .then(() => {
          sessionStorage.setItem("predicted", "true");
        })
        .catch((err) => console.error("Prediction error:", err));
    }
  }, []);

  const [showResults, setShowResults] = useState(false);

  return (
    <div className="App">
      <Header />
      <div className="main-page">
        {!showResults && (
          <div className="welcome-container">
            <div className="headline">
              <h1>
                Welcome<br />
                to Mode<br />
                Metrics
              </h1>
            </div>
            <div className="description">
              <p>
                Type an item and we’ll let you<br />
                know if it’s trending or not.<br />
              </p>
              <p>
                Currently only available for<br />
                women's shoe styles from ASOS.<br />
              </p>
            </div>
          </div>
        )}
        <SearchProduct showResults={showResults} setShowResults={setShowResults} />
      </div>
      <Footer />
    </div>
  );
}

export default App;
