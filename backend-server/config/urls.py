from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('emotiart/', include('EmotiArt_app.urls')),  # Emotiart_app의 URL을 include
]