from django import forms

class ImageUploadForm(forms.Form):
    drawing = forms.ImageField()
