from django.db import models

class UploadedImage(models.Model):
    title = models.CharField(max_length=100)
    image = models.ImageField(upload_to='images/')  # 이미지 필드를 사용하여 이미지 저장

    def __str__(self):
        return self.title
