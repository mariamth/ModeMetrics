from rest_framework.decorators import api_view
from rest_framework.response import Response
from .serializers import FashionItemSerializer
import random
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from api.models import FashionItem
from difflib import SequenceMatcher
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import os
import requests
from django.test import Client


#fetch data from the API and store it in the database
@csrf_exempt
def fetch_api(request):
    if request.method != "POST":
        return JsonResponse({"error": "POST required"}, status=400)

    url = "https://asos2.p.rapidapi.com/products/v2/list"
    headers = {
        "x-rapidapi-key": os.environ.get("RAPIDAPI_KEY"),
        "x-rapidapi-host": "asos2.p.rapidapi.com",
    }
    
    #pagination used to get as much data as possible without overloading the api
    offset = 0
    limit = 48
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
            formatted = []
            for product in products:
                item = {
                    "id": str(product["id"]),
                    "name": product["name"],
                    "price": product["price"]["current"]["value"]
                }
                formatted.append(item)
            all_products.extend(formatted)

            if len(all_products) >= max_products or len(products) < limit:
                break
            offset += limit

        #send to store_data view
        client = Client()
        
        client.post("/api/store_data/", 
                data={"products": all_products},
                content_type="application/json"
                )
        
        return JsonResponse({"message": "Fetched and stored products"})

    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)



#store data in the database and have a view for it
@api_view(['POST'])
def store_data(request):
    FashionItem.objects.all().delete()
    products = request.data.get('products', [])
    for product in products:
        FashionItem.objects.update_or_create(
            product_id=product['id'],
            defaults={
                'name': product['name'],
                'price': product['price']
            }
        )
    return Response({"message": "Data stored successfully", "count": len(products)})


#get all data from the database
@api_view(['GET'])
def get_data(request):
    items = FashionItem.objects.all()
    serializer = FashionItemSerializer(items, many=True)
    return Response(serializer.data)


#THE MODEL ITSELF
class FashionModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(FashionModel, self).__init__()
        self.hidden = nn.Linear(input_size, hidden_size) #liner input layer
        self.batchnorm = nn.BatchNorm1d(hidden_size) #batch normalisation
        self.relu = nn.ReLU() #activation function
        self.dropout = nn.Dropout(p=0.3) #dropout layer to prevent help prevent overfitting
        self.output = nn.Linear(hidden_size, output_size) #output layer

    #forward pass
    def forward(self, x):
        x = self.hidden(x)
        x = self.batchnorm(x)
        x = self.relu(x)
        return self.output(x)



#function to get training data
def get_training_data():
    items = list(FashionItem.objects.all())  #convert QuerySet to list
    if not items:
        return None, None, None, None, None, None

    #shuffle the items to ensure randomness
    random.shuffle(items)

    #define the variables needed to help train the model
    prices = np.array([item.price for item in items])
    median_price = np.median(prices)
    #create the labels for the model
    #if the price is above the median price, it is trendy
    trend_labels = np.array([1 if price > median_price else 0 for price in prices])
    #normalise the prices
    mean_price = np.mean(prices)
    #standard deviation of the prices
    std_price = np.std(prices)
    X_price = (prices - mean_price) / std_price
    #newest items are more trendy based on the freshness parameter of the api
    recency = np.linspace(1, 0, len(items))

    price_quantiles = np.quantile(prices, [0.33, 0.66])
    categories = []
    for price in prices:
        if price <= price_quantiles[0]:
            #cheaper category of shoes
            categories.append([1, 0, 0])  
        elif price <= price_quantiles[1]:
            #mid category of shoes
            categories.append([0, 1, 0])
        else:
            #expensive category of shoes
            categories.append([0, 0, 1])  #expensive
    X_catagories = np.array(categories)


    #create the training data
    #combine the price, freshness and the categories into one array
    X_combined = np.column_stack((X_price.reshape(-1, 1), X_catagories, recency.reshape(-1, 1))) #use of np.column_stack from:  https://numpy.org/doc/stable/reference/generated/numpy.column_stack.html   

    #split the data into training and testing data
    #80% of the data is used for training and 20% for testing
    split_id = int(0.8 * len(X_combined))
    X_train = X_combined[:split_id]
    y_train = trend_labels[:split_id]
    X_test = X_combined[split_id:]
    y_test = trend_labels[split_id:]

    return (
        #return the training data
        torch.tensor(X_train).float(),
        torch.tensor(y_train).float().view(-1, 1),
        torch.tensor(X_test).float(),
        torch.tensor(y_test).float().view(-1, 1),
        mean_price,
        std_price
    )


#train the model
def train_model(epochs=1000):
    #use function above to get training data
    X_train, y_train, X_test, y_test, mean_price, std_price = get_training_data()
    if X_train is None:
        return None, None, None
    
    #create the model 
    #input size is 5 (price, cheap, mid, expensive, freshness), hidden size is 128 and output size is 1
    model = FashionModel(5, 128, 1)
    #loss function is binary cross entropy with logits 
    loss_calc = nn.BCEWithLogitsLoss()
    #adams optimizer used
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    model.train()

    #train the model for the number of epochs
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = loss_calc(outputs, y_train)
        #backpropagation
        loss.backward()
        optimizer.step()
        #print the loss every 100 epochs
        if epoch % 100 == 0:
            print(f"Epoch {epoch} Loss: {loss.item()}")


    #model in evaluation mode
    model.eval()
    #check the accuracy of the model
    #use torch.no_grad() to prevent gradients from being calculated
    with torch.no_grad():
        #training accuracy 
        predictions = torch.sigmoid(model(X_train)) #use of sigmoid from: https://pytorch.org/docs/stable/generated/torch.sigmoid.html
        predicted_labels = (predictions > 0.5).float()
        train_acc = (predicted_labels == y_train).sum().item() / y_train.size(0)
        print(f"Training Accuracy: {train_acc:.2f}")

        #test accuracy
        test_predictions = torch.sigmoid(model(X_test))
        test_predicted_labels = (test_predictions > 0.5).float()
        test_acc = (test_predicted_labels == y_test).sum().item() / y_test.size(0)
        print(f"Test Accuracy: {test_acc:.2f}")

    return model, mean_price, std_price


#predict trends
@api_view(['GET'])
def predict_trends_view(request):
    model, _, _ = train_model()
    #ceck if there is enough data
    if model is None:
        return Response({"error": "Not enough data to train model."})

    #get all the items from the database
    items = FashionItem.objects.all()
    prices = np.array([item.price for item in items])

    #normalise the prices
    mean_price = np.mean(prices)
    #standard deviation of the prices
    std_price = np.std(prices)
    X_price = (prices - mean_price) / std_price
    #freshness parameter added as a feature
    #newest items are more trendy based on the freshness parameter of the api
    recency = np.linspace(1, 0, len(items))

    price_quantiles = np.quantile(prices, [0.33, 0.66]) #use of np.quantile() from: https://numpy.org/doc/stable/reference/generated/numpy.quantile.html 
    categories = []
    for price in prices:
        if price <= price_quantiles[0]:
            #cheaper category of shoes
            categories.append([1, 0, 0])  
        elif price <= price_quantiles[1]:
            #mid category of shoes
            categories.append([0, 1, 0])
        else:
            #expensive category of shoes
            categories.append([0, 0, 1])  #expensive
    X_catagories = np.array(categories)
    
    #combine the price and the categories into one array
    #X_price is reshaped to be 2D since categories is 2D
    X_test_combined = np.column_stack((X_price.reshape(-1, 1), X_catagories, recency.reshape(-1, 1)))
    X_test_tensor = torch.tensor(X_test_combined).float()

    model.eval()
    #response list to store the predictions that will be used to update the database
    response = []
    with torch.no_grad():
        predictions = torch.sigmoid(model(X_test_tensor))
        predictions = (predictions > 0.5).float().numpy().flatten()

        for item, prediction in zip(items, predictions):
            #update the item in the database
            item.is_trendy = bool(prediction)
            #save the item
            item.save()
            response.append({
                "product_id": item.product_id,
                "name": item.name,
                "price": item.price,
                "is_trendy": item.is_trendy
            })

    return Response(response)

#basic word matching to compare the product name with the search query
def get_matching_products(search_query, items):
    items = FashionItem.objects.all()
    response = []
    for item in items:
        #use the SequenceMatcher to compare the similarity between the search query and the product name
        similarity = SequenceMatcher(None, search_query, item.name).ratio() #sequencematcher from: https://docs.python.org/3/library/difflib.html#difflib.SequenceMatcher
        if similarity > 0.3:
            response.append({
                #format the response to contain the product id, name, price, similarity and if it is trendy
                "product_id": item.product_id,
                "name": item.name,
                "price": item.price,
                "similarity": similarity,
                "is_trendy": item.is_trendy
            })
            
    #sort the response by similarity
    response.sort(key=lambda x: x['similarity'], reverse=True)
    return response

#search for products, connects to front end where user needs to search for a product
@api_view(['GET'])
def search_products(request):
    search_query = request.query_params.get('q', '')
    if not search_query:
        return Response({"error": "Please provide an appropriate query (?q=...)"}, status=400)

    #uses the function above to get the matching products
    items = FashionItem.objects.all()
    matched = get_matching_products(search_query, items)
    #total number of matched items
    total_items = len(matched)
    #if there are no matched items return error
    if total_items == 0:
        return Response({"error": "No matching products found."}, status=404)

    #calculate the proportion of trendy items
    #is the searched item trendy or not based on the average of the matched items
    trendy_count = sum(1 for item in matched if item.get("is_trendy", False))
    trendy_proportion = trendy_count / total_items
    is_overwhelmingly_trendy = trendy_proportion > 0.5

    if trendy_proportion > 0.7:
        advice = "This style is very popular and will likely remain trendy for a while longer. Now is a great time to purchase!"
    elif trendy_proportion > 0.5:
        advice = "This style is trendy and may continue to be. It could be a good time to purchase but maybe see if you can find it cheaper."
    elif trendy_proportion > 0.3:
        advice = "This style is not trending at the momment; it may not be the best purchase."
    else:
        advice = "This style is currently not popular."

    return Response({
        #format the response to contain the matched products
        "matched_products": matched,
        "is_overwhelmingly_trendy": is_overwhelmingly_trendy,
        "trendy_proportion": round(trendy_proportion, 2),
        "advice": advice
    })
