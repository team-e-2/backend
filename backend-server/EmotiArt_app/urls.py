from django.urls import path
from . import views

urlpatterns = [
    path('receive-image/', views.receive_image, name='receive_image'),
    # 다른 URL 패턴들도 추가 가능
]
