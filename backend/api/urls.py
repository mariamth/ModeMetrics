from django.urls import path
from .views import store_data, get_data, predict_trends_view, search_products

urlpatterns = [
    path('store_data/', store_data, name='store_data'), 
    path('get_data/', get_data, name='get_data'),           
    path('predict/', predict_trends_view, name='predict'),  
    path('similar/', search_products, name='similar'),     
]