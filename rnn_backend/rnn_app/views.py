from django.core.files.base import ContentFile
from django.core.files.storage import default_storage
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt


@csrf_exempt
def receive_image(request):
    if request.method == 'POST' and request.FILES['drawing']:
        image_file = request.FILES['drawing']

        # 이미지를 MEDIA_ROOT에 저장
        file_path = default_storage.save('images/' + image_file.name, ContentFile(image_file.read()))

        import os
        import random
        import shutil
        import base64
        import io

        import ndjson
        import numpy as np
        from PIL import Image, ImageDraw
        from keras import layers, models
        from keras.preprocessing.image import img_to_array
        from sklearn.metrics.pairwise import cosine_similarity

        # Directory to save similar images
        save_dir = "rnn_app/similar_images"

        # Quick, Draw! 데이터셋 경로
        dataset_path = "rnn_app/rnn_tutorial_data"

        # 클래스명 정의 (원하는 클래스들로 수정)
        class_names = ["cat"]

        # 데이터 로드 함수
        def load_quickdraw_data(dataset_path, class_names):
            images = []
            labels = []

            for i, class_name in enumerate(class_names):
                file_path = os.path.join(dataset_path, "cat.ndjson")

                with open(file_path) as f:
                    data = ndjson.load(f)

                for entry in data:
                    img = Image.new("L", (256, 256), 255)  # 백색 이미지 생성
                    draw = ImageDraw.Draw(img)
                    for stroke in entry["drawing"]:
                        draw.line(list(zip(stroke[0], stroke[1])), fill=0, width=5)

                    img = img.resize((28, 28))
                    img_array = img_to_array(img)
                    images.append(img_array)
                    labels.append(i)

            return np.array(images), np.array(labels)

        # 모델 구축 함수 (간소화)
        def build_simple_model(input_shape, num_classes):
            model = models.Sequential()
            model.add(layers.Conv2D(32, (3, 3), activation="relu", input_shape=input_shape))
            model.add(layers.MaxPooling2D((2, 2)))
            model.add(layers.Flatten())
            model.add(layers.Dense(num_classes, activation="softmax"))

            model.compile(
                optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
            )

            return model

        # 유사한 그림 찾기 함수
        def find_similar_images(input_image, dataset_images, dataset_labels, save_dir):
            # 모델 학습
            input_shape = input_image.shape[1:]
            num_classes = len(np.unique(dataset_labels))

            # 저장된 모델 불러오기 또는 새로 학습
            model_path = "rnn_app/saved_model.h5"
            if os.path.exists(model_path):
                model = models.load_model(model_path)
            else:
                model = build_simple_model(input_shape, num_classes)
                model.fit(dataset_images, dataset_labels, epochs=5, validation_split=0.2)
                model.save(model_path)

            # 입력된 그림의 임베딩 벡터 얻기
            input_embedding = model.predict(input_image)

            # 데이터셋의 모든 그림에 대한 임베딩 벡터 얻기
            dataset_embeddings = model.predict(dataset_images)

            # 코사인 유사도 계산
            similarities = cosine_similarity(input_embedding, dataset_embeddings)[0]

            # 유사도가 높은 순서대로 정렬
            sorted_indices = np.argsort(similarities)[::-1]

            # 상위 몇 개의 유사한 그림 반환
            top_similar_images = [dataset_images[i] for i in sorted_indices[:5]]

            return top_similar_images

        def load_and_preprocess_input_image(image_path):
            img = Image.open(image_path)
            img = img.convert("L")  # 흑백 이미지로 변환
            img = img.resize((28, 28))
            img_array = img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0) / 255.0  # 정규화
            return img_array

        # 클래스에 해당하는 Quick, Draw! 데이터 로드
        dataset_images, dataset_labels = load_quickdraw_data(dataset_path, class_names)

        # input_image_path 디렉토리에서 무작위 이미지 파일 선택
        input_image_path = "media/images/"
        image_files = [f for f in os.listdir(input_image_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
        random_image_file = random.choice(image_files)
        selected_image_path = os.path.join(input_image_path, random_image_file)

        # 선택된 이미지를 load_and_preprocess_input_image 함수를 사용하여 처리
        input_image = load_and_preprocess_input_image(selected_image_path)

        # 유사한 그림 찾기
        similar_images = find_similar_images(input_image, dataset_images, dataset_labels, save_dir)

        # 결과 출력
        print("Input Image:")

        print("\nSimilar Images:")
        response_data = []

        for i, img_array in enumerate(similar_images):
            img = Image.fromarray((img_array.squeeze() * 255).astype(np.uint8), mode="L")

            # 이미지를 바로 클라이언트에 반환
            img_io = io.BytesIO()
            img.save(img_io, format='PNG')
            img_io.seek(0)

            # 이미지를 response_data 리스트에 추가
            response_data.append({
                'filename': f"similar_image_{i}.png",
                'content_type': 'image/png',
                'content': base64.b64encode(img_io.getvalue()).decode('utf-8')
            })

        # ../media/images/ 디렉토리에 있는 모든 이미지 삭제
        media_images_path = "media/images/"
        for file_name in os.listdir(media_images_path):
            file_path = os.path.join(media_images_path, file_name)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"Error while deleting {file_path}: {e}")

        # 오류가 발생하지 않으면 여기까지 도달
        return JsonResponse({'result': 'success', 'similar_images': response_data})
    else:
        return JsonResponse({'error': 'No image found in the request.'}, status=400)


# @csrf_exempt
# def return_images(request):
#     try:
#

