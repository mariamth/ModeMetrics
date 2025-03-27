from django.db import models

# Create your models here.
class FashionItem(models.Model):
    """
    Model to store fashion items
    """
    
    product_id = models.CharField(max_length=100, unique=True)
    name = models.CharField(max_length=255)
    price = models.FloatField()
    is_trendy = models.BooleanField(default=False)
    

    def __str__(self):
        return self.name