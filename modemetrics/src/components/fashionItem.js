import React from "react";

const FashionItem = ({ product }) => {
    return (
        <div className="fashion-item">
            <img src={`https://${product.imageUrl}`} alt={product.name} width="100" />
            <h3>{product.name}</h3>
            <p>${product.price.current.value}</p>
        </div>
    );
};

export default FashionItem;
