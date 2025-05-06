from rest_framework import serializers
from .models import FashionItem

class FashionItemSerializer(serializers.ModelSerializer):
    """
    Serializer class for the FashionItem model.
    This class converts FashionItem model instances into JSON format and validates incoming data for the model.
    """

    class Meta:
        model = FashionItem
        fields = '__all__' 