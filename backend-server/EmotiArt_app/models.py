from django.db import models
class ProductImg(BaseModel):

    product = models.ForeignKey(Product,on_delete=models.CASCADE, related_name='images')
    image = models.ImageField(upload_to="", blank=True)