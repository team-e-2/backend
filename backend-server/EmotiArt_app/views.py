from django.http import JsonResponse, HttpResponseNotFound, FileResponse
from django.views.decorators.csrf import csrf_exempt
import os

@csrf_exempt
def change_image(request):
    if request.method == 'POST' and request.FILES.get('drawing'):
        image_file = request.FILES['drawing']

        # 이미지 저장 경로
        save_path = 'C:/Users/JangHeesu/PycharmProjects/backend/backend-server/frontend_save_image/'

        # 클라이언트로부터 받은 이미지를 저장
        with open(os.path.join(save_path, image_file.name), 'wb+') as destination:
            for chunk in image_file.chunks():
                destination.write(chunk)

        # 저장한 이미지를 읽어와서 반환
        image_data_list = []
        final_image_path = 'C:/Users/JangHeesu/PycharmProjects/backend/final_emotiart_image/'
        for filename in os.listdir(final_image_path):
            if os.path.isfile(os.path.join(final_image_path, filename)):
                with open(os.path.join(final_image_path, filename), 'rb') as file:
                    image_data = file.read()
                    image_data_list.append(image_data)

        if image_data_list:
            # 이미지 데이터를 합쳐서 반환
            combined_image_data = b''.join(image_data_list)
            return FileResponse(combined_image_data, content_type='image/png')
        else:
            return HttpResponseNotFound('No images found in the final_emotiart_image directory.')
    else:
        return JsonResponse({'error': 'No image found in the request or invalid request method.'}, status=400)