from django.contrib import admin
from django.urls import path, include
from django.urls import include, path

urlpatterns = [
    path('admin/', admin.site.urls),
    path('emotiart/', include('EmotiArt_app.urls'))
]

