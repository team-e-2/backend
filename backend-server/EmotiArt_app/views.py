from django.http import JsonResponse
from django.http import FileResponse
import io
from PIL import Image
from django.views.decorators.csrf import csrf_exempt

@csrf_exempt
def receive_image(request):
    if request.method == 'POST' and request.FILES['drawing']:
        image_file = request.FILES['drawing']

        # 이미지 처리 로직
        image = Image.open(image_file)
        # 예시로 이미지를 회전시켜봅시다
        rotated_image = image.rotate(90)

        # 임시 파일로 이미지를 저장
        buffer = io.BytesIO()
        rotated_image.save(buffer, format='PNG')
        buffer.seek(0)

        # 임시 파일을 응답으로 전송
        return FileResponse(buffer, content_type='image/png')
    else:
        return JsonResponse({'error': 'No image found in the request.'}, status=400)

'''
def receive_image(request):
    if request.method == 'POST' and request.FILES['drawing']:
        image_file = request.FILES['drawing']

        # 이미지를 열어서 PIL 이미지 객체로 변환
        image = Image.open(image_file)

        # 이미지를 회전시킴 (예시로 90도 회전)
        rotated_image = image.rotate(90)

        # 회전한 이미지를 PNG 형식으로 변환
        buffer = rotated_image.convert("RGB")
        buffer.save("rotated_image.png", format='PNG')
        buffer.seek(0)

        # 회전된 이미지를 클라이언트에게 전송
        return FileResponse(buffer, content_type='image/png')
    else:
        return JsonResponse({'error': 'No file named "drawing" found in the request.'}, status=400)
    '''

