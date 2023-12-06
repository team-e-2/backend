from django import forms
from django.conf import settings
from .models import UploadedImage
import os
import uuid

class ImageUploadForm(forms.Form):
    drawing = forms.ImageField()

    class Meta:
        # 폼 속성 설정
        enctype = 'multipart/form-data'

    def save(self, commit=True):
        drawing = self.cleaned_data['drawing']

        # Generate a unique filename using UUID
        unique_filename = f"{uuid.uuid4()}.jpg"

        # Define the full file path
        file_path = os.path.join(settings.MEDIA_ROOT, 'uploads', unique_filename)

        # Save the file to the specified path
        with open(file_path, 'wb') as destination:
            for chunk in drawing.chunks():
                destination.write(chunk)

        # Create a model instance and return it
        instance = UploadedImage(drawing=file_path)
        if commit:
            instance.save()
        return instance
