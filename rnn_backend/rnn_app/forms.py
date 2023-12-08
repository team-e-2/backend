# forms.py

from django import forms
from .models import MyImageModel

class ImageUploadForm(forms.ModelForm):
    class Meta:
        model = MyImageModel
        fields = ['image']
