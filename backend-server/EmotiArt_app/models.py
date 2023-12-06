from django.db import models

class UploadedImage(models.Model):
    drawing = models.ImageField(upload_to='uploads/')

    def __str__(self):
        return self.drawing.url
