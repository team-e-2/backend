from django.shortcuts import render
from django.http import HttpResponseRedirect
from .forms import ImageUploadForm  # 이미지 업로드 폼 불러오기


def upload_image(request):
    if request.method == "POST":
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            image = form.cleaned_data["image"]
            # 이미지를 저장하거나 다른 작업 수행
            # image 변수에는 업로드된 이미지 파일이 포함됩니다.
            return HttpResponseRedirect("/success/")  # 업로드 성공 후 리디렉션
    else:
        form = ImageUploadForm()
    return render(request, "upload.html", {"form": form})
