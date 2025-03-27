from rest_framework.decorators import api_view
from rest_framework.response import Response
from .models import FashionItem
from .serializers import FashionItemSerializer
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from api.models import FashionItem
from difflib import SequenceMatcher


#store data in the database and have a view for it
@api_view(['POST'])
def store_data(request):
    FashionItem.objects.all().delete()
    products = request.data.get('products', [])
    for product in products:
        FashionItem.objects.update_or_create(
            product_id=product['id'],
            defaults={'name': product['name'], 'price': product['price']}
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
    #define the model and the layers
    def __init__(self, input_size, hidden_size, output_size):
        super(FashionModel, self).__init__()
        
        self.hidden = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        #forward pass
        x = self.hidden(x)
        x = self.relu(x)
        return self.output(x)


#function to get training data
def get_training_data():
    items = FashionItem.objects.all()
    #check if there are any items in the database
    if not items:
        return None, None, None, None

    #define the variables needed to help train the model
    prices = np.array([item.price for item in items])
    median_price = np.median(prices)
    trend_labels = np.array([1 if price > median_price else 0 for price in prices])
    #normalise the prices
    mean_price = np.mean(prices)
    std_price = np.std(prices)
    X_train = (prices - mean_price) / std_price

    return (
        #return the training data using tensor
        torch.tensor(X_train).float().view(-1, 1),
        torch.tensor(trend_labels).float().view(-1, 1),
        mean_price,
        std_price
    )


#train the model
def train_model(epochs=1000):
    #use function above to get training data
    X_train, y_train, mean_price, std_price = get_training_data()
    if X_train is None:
        return None, None, None
    
    model = FashionModel(1, 128, 1)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(epochs):
        
        optimizer.zero_grad()
        outputs = model(X_train)
        #calculate the loss
        loss = criterion(outputs, y_train)
        loss.backward()
        #use the optimizer to update the weights
        optimizer.step()
        #print the loss every 100 epochs
        if epoch % 100 == 0:
            print(f"Epoch {epoch} Loss: {loss.item()}")

    return model, mean_price, std_price

#predict trends
@api_view(['GET'])
def predict_trends_view(request):
    model, mean_price, std_price = train_model()
    #check if there is enough data to train the model
    if model is None:
        return Response({"error": "Not enough data to train model."})
    #get all the items from the database
    items = FashionItem.objects.all()
    prices = np.array([item.price for item in items])
    #normalise the prices
    X_test = (prices - mean_price) / std_price
    X_test = torch.tensor(X_test).float().view(-1, 1)
    response = [] #store the response

    with torch.no_grad(): 
        
        predictions = torch.sigmoid(model(X_test))
        predictions = (predictions > 0.5).float().numpy().flatten()
    for item, prediction in zip(items, predictions):
        #update the item in the database
        item.is_trendy = bool(prediction)
        #save the item
        item.save()
        response.append({
            #format the response to contain the product id, name, price and if it is trendy
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
        similarity = SequenceMatcher(None, search_query, item.name).ratio()
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
    return sorted(response, key=lambda x: x['similarity'], reverse=True)

#search for products, connects to front end where user needs to search for a product
@api_view(['GET'])
def search_products(request):
    search_query = request.query_params.get('q', '')
    if not search_query:
        return Response({"error": "Please provide a product name query (?q=...)"}, status=400)

    #uses the function above to get the matching products
    items = FashionItem.objects.all()
    matched = get_matching_products(search_query, items)
    #total number of matched items
    total_items = len(matched)
    #if there are no matched items return error
    if total_items == 0:
        return Response({"error": "No matching products found."}, status=404)

    #calculate the proportion of trendy items
    # is the searched item trendy or not based on the average of the matched items
    trendy_count = sum(1 for item in matched if item.get("is_trendy", False))
    trendy_proportion = trendy_count / total_items
    is_overwhelmingly_trendy = trendy_proportion > 0.5

    return Response({
        #format the response to contain the matched products
        "matched_products": matched,
        "is_overwhelmingly_trendy": is_overwhelmingly_trendy,
        "trendy_proportion": round(trendy_proportion, 2)
    })
