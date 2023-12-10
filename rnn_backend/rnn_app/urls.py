# urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('receive/', views.receive_image, name='receive_image'),
    #path('return/', views.return_images, name='return_images')
    # 다른 URL 패턴들을 추가할 수 있습니다.
]
