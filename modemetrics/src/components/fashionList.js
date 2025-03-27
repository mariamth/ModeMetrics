import React, { useState, useEffect } from "react";
import getApiData from "./getAPI";
import FashionItem from "./fashionItem"; // Component for single item



const FashionList = () => {
    const [products, setProducts] = useState([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);

    useEffect(() => {
        const fetchData = async () => {
            try {
                const data = await getApiData();
                setProducts(data);
            } catch (error) {
                setError(error.message);
            } finally {
                setLoading(false);
            }
        };

        fetchData();
    }, []);

    if (loading) return <p>Loading fashion data...</p>;
    if (error) return <p>Error: {error}</p>;

    return (
        <div className="fashion-container">
            <h2>ASOS Shoe Products</h2>
            <div className="fashion-grid">
                {products.map((product) => (
                    <FashionItem key={product.id} product={product} />
                ))}
            </div>
        </div>
    );
};

export default FashionList;
