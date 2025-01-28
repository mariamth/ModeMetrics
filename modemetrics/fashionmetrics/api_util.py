import requests
import os

API_KEY = os.getenv('ASOS_API_KEY') 

def fetch_asos_data(category_id="4209"):
    url = "https://asos2.p.rapidapi.com/products/v2/list"
    querystring = {
        "country": "US",
        "currency": "USD",
        "sort": "freshness",
        "lang": "en-US",
        "categoryId": category_id,
        "limit": "48",
        "offset": "0"
    }
    headers = {
        "X-RapidAPI-Key": API_KEY,
        "X-RapidAPI-Host": "asos2.p.rapidapi.com"
    }
    response = requests.get(url, headers=headers, params=querystring)
    return response.json()
