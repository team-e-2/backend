# rnn_app/views.py

import subprocess
from django.shortcuts import render, redirect
from .forms import ImageUploadForm
from .models import MyImageModel
from django.shortcuts import render
import os

def upload_image(request):
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            image_instance = form.save()

            # 파일 업로드 후 main.py 실행
            image_path = image_instance.image.path
            subprocess.run(['python', 'main.py', image_path])

            return redirect('rnn_app:show_images')  # 새로운 뷰로 리다이렉트
    else:
        form = ImageUploadForm()

    return render(request, 'upload_image.html', {'form': form})

def show_images(request):
    image_dir = '../similar_images'  # similar_images 디렉토리 경로
    images = os.listdir(image_dir)
    print(images)

    return render(request, 'show_images.html', {'images': images})
