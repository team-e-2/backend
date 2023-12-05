from django.http import HttpResponse, Http404
from django.views.decorators.http import require_http_methods
from django.conf import settings
from django.shortcuts import render
import os

image_folder = 'C:\Users\JangHeesu\PycharmProjects\backend\create_image'

@require_http_methods(["GET"])   #이미지 보내기
def send_image(request, image_name):
    image_path = os.path.join(image_folder, image_name)

    if not os.path.isfile(image_path):
        raise Http404  # 파일이 없는 경우 404 에러 반환

    with open(image_path, 'rb') as image_file:
        return HttpResponse(image_file.read(), content_type='image/png')



image_folder = 'C:\Users\JangHeesu\PycharmProjects\backend\save_image'

@require_http_methods(["GET"]) #이미지 받기
def get_image(request, image_name):
    image_path = os.path.join(image_folder, image_name)

    if not os.path.isfile(image_path):
        raise Http404

    with open(image_path, 'rb') as image_file:
        return HttpResponse(image_file.read(), content_type='image/png')


def upload_image(request):
    if request.method == 'POST' and request.FILES['image']:
        uploaded_image = request.FILES['image']
        destination_path = 'C:\Users\JangHeesu\PycharmProjects\backend\save_image'  # 이미지를 저장할 경로

        with open(f'{destination_path}/{uploaded_image.name}', 'wb+') as destination:
            for chunk in uploaded_image.chunks():
                destination.write(chunk)

        return render(request, 'success.html')  # 이미지 업로드 성공 시 success.html 렌더링

    return render(request, 'upload.html')  # 업로드 폼 렌더링
