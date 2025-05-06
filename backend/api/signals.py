import os
import sys
import time
import requests
from django.db.models.signals import post_migrate
from django.dispatch import receiver
from api.models import FashionItem
from api.views import train_model


#signal handlers that are executed after migrations are applied. 
#it fetches data from an external API and populates the database with fashion items.
#then calls to train a model to predict trendy item and updates the database with the predictions.



@receiver(post_migrate) #manually triggered after the database migration
def fetch_data(sender, **kwargs):
    if sender.name != "api":
        return
    #this is skipped when running tests
    if 'test' in sys.argv:
        return

    #check if the database is empty
    #and if so, fetch data from the API
    if FashionItem.objects.exists():
        return

    url = "https://asos2.p.rapidapi.com/products/v2/list"
    #same logic as in fetch_api in views.py
    headers = {
        "x-rapidapi-key": os.environ.get("RAPIDAPI_KEY"), 
        "x-rapidapi-host": "asos2.p.rapidapi.com",
    }
    if not headers["x-rapidapi-key"]:
        print("The API Key, RAPIDAPI_KEY, is not set in environment.")
        return

    offset = 0
    limit = 48
    total_fetched = 0
    all_products = []
    max_products = 1000

    try:
        while True:
            params = {
                "store": "COM",
                "offset": str(offset),
                "categoryId": "4172",
                "country": "GB",
                "sort": "freshness",
                "currency": "GBP",
                "sizeSchema": "EU",
                "lang": "en-GB",
            }

            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            products = response.json().get("products", [])

            if not products:
                break
            for product in products:
                FashionItem.objects.update_or_create(
                    product_id=str(product["id"]),
                    defaults={
                        "name": product["name"],
                        "price": product["price"]["current"]["value"],
                    },
                )

            total_fetched += len(products)
            all_products.extend(products)
            if len(all_products) >= max_products or len(products) < limit:
                break
            offset += limit
            print(f"total products: {total_fetched}")

            time.sleep(0.1) 

    except Exception as e:
        print("fetch failed:", str(e))

#This function is called after the database migration
#and it trains a model to predict trendy items
#and updates the database with the predictions.
@receiver(post_migrate)
def predict_trends(sender, **kwargs):
    if sender.name != "api":
        return
    #this is skipped when running tests
    if 'test' in sys.argv:
        return 

    if FashionItem.objects.exists():
        try:
            model, _, _ = train_model()
            trendy_count = FashionItem.objects.filter(is_trendy=True).count()
        except Exception as e:
            print("Trend prediction failed:", str(e))
