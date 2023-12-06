from django.shortcuts import render
from django.http import HttpResponseRedirect
from django.http import JsonResponse
from .form import ImageUploadForm  # 이미지 업로드 폼 불러오기
from rest_framework.decorators import api_view
from rest_framework.response import Response
from .models import UploadedImage


@api_view(['GET', 'POST'])
def upload_image(request):
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            instance = form.save()  # 이미지를 저장하고 UploadedImage 모델 인스턴스를 반환
            image_url = instance.drawing.url  # 업로드된 이미지의 URL 가져오기
            return Response({'image_url': image_url})
        else:
            errors = form.errors
            return Response({'errors': errors}, status=400)

    return Response({'message': '이 엔드포인트는 GET 및 POST 메서드를 모두 허용합니다.'})