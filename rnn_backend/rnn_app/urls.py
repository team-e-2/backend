# urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('receive/', views.receive_image, name='receive_image'),
    path('change/', views.change_image, name='change_image'),
    #path('process/', views.process_image, name='process_image')
    # 다른 URL 패턴들을 추가할 수 있습니다.
]
