# models.py

from django.db import models

class MyImageModel(models.Model):
    image = models.ImageField(upload_to='images/')
