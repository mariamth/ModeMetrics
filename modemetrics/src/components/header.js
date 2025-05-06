import React from 'react';
import "./style.css";
import logo from './logo.png'; //personal logo for this project

const Header = () => {
    return (
        <header className="header">
            <div className="header-left">
                <img src={logo} alt="ModeMetrics logo" className="header-logo" />
            </div>
            <div sclassName="header-right">
                <span >modemetrics by mariam</span>
            </div>
        </header>
    );
};



export default Header;
