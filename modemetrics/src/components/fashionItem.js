import React from "react";

const FashionItem = ({ product }) => {
    return (
        <div className="fashion-item">
            <h3>{product.name}</h3>
            <p>Price: £{product.price}</p>
        </div>
    );
};

export default FashionItem;
