from django.shortcuts import render
from django.http import HttpResponseRedirect
from django.http import JsonResponse
from .form import ImageUploadForm  # 이미지 업로드 폼 불러오기
from rest_framework.decorators import api_view
from rest_framework.response import Response
from .models import UploadedImage
from django.core.exceptions import ValidationError


@api_view(['POST'])
def upload_image(request):
    print("Received request:", request.POST)
    print("Received files:", request.FILES)
    if request.method == 'POST':
        form = ImageUploadForm(request.POST)
        drawing_file = request.FILES.get('drawing')
        # drawing_file = request.GET.get('drawing')

        try:
            if not drawing_file:
                raise ValidationError("이미지 파일을 첨부하세요.")

            allowed_content_types = ['image/jpeg', 'image/png']
            if drawing_file.content_type not in allowed_content_types:
                raise ValidationError("지원되지 않는 이미지 형식입니다. JPEG 또는 PNG 이미지를 사용하세요.")

            # 유효성 검사 후에 폼 초기화 및 저장 수행
            if form.is_valid():
                instance = form.save()
                image_url = instance.drawing.url
                return Response({'image_url': image_url})
            else:
                errors = form.errors
                return Response({'errors': errors}, status=421)
        except ValidationError as e:
            return Response({'errors': str(e)}, status=422)

    return Response({'message': '이 엔드포인트는 POST 메서드만 허용합니다.'}, status=405)
