# urls.py
from django.urls import path
from .views import upload_image, show_images

app_name = 'rnn_app'

urlpatterns = [
    path('', upload_image, name='upload_image'),  # 루트 경로에 대한 패턴 추가
    path('upload/', upload_image, name='upload_image'),
    path('show_images/', show_images, name='show_images'),  # 새로운 뷰를 위한 URL 패턴
    # 다른 URL 패턴들을 추가할 수 있습니다.
]
