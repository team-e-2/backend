import io
import os


from django.core.files.base import ContentFile
from django.core.files.storage import default_storage
from django.http import JsonResponse, HttpResponseNotFound, FileResponse
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
            # 파일명을 생성
            filename = f"similar_image_{i}.png"

            # 이미지를 파일로 저장
            file_path = os.path.join(save_dir, filename)
            img.save(file_path, format='PNG')

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
        return JsonResponse({'images': response_data})
    else:
        return JsonResponse({'error': 'No image found in the request.'}, status=400)





@csrf_exempt
def change_image(request):
    if request.method == 'POST' and request.FILES.get('drawing'):
        image_file = request.FILES['drawing']

        from django.http import FileResponse, HttpResponseNotFound, JsonResponse
        from django.views.decorators.csrf import csrf_exempt
        from django.conf import settings
        from PIL import Image
        import os
        import random
        import numpy as np
        from keras.layers import Dense, Flatten, Reshape
        from keras.models import Sequential
        from keras.optimizers import Adam
        from keras.preprocessing.image import ImageDataGenerator

        # 이미지 저장 경로
        save_path = "rnn_app/frontended_save_image/"

        # 클라이언트로부터 받은 이미지를 저장
        with open(os.path.join(save_path, image_file.name), 'wb+') as destination:
            for chunk in image_file.chunks():
                destination.write(chunk)

        # 이미지 데이터셋 경로
        data_dir = "rnn_app/data_image"

        # 이미지 데이터를 읽고 전처리하는 ImageDataGenerator 생성
        datagen = ImageDataGenerator(rescale=1. / 255)  # 이미지를 0과 1 사이로 스케일링

        # 이미지 데이터셋 불러오기
        batch_size = 10
        image_size = (200, 200)  # 이미지 사이즈 조정
        class_mode = 'categorical'  # 분류형 레이블 사용

        train_data = datagen.flow_from_directory(
            data_dir,
            target_size=image_size,
            batch_size=batch_size,
            class_mode=class_mode
        )

        # Generator 모델 정의
        def build_generator(latent_dim, output_shape):
            model = Sequential()
            model.add(Dense(128, input_dim=latent_dim, activation='relu'))
            model.add(Dense(256, activation='relu'))
            model.add(Dense(np.prod(output_shape), activation='sigmoid'))
            model.add(Reshape(output_shape))
            return model

        # Discriminator 모델 정의
        def build_discriminator(input_shape):
            model = Sequential()
            model.add(Flatten(input_shape=input_shape))
            model.add(Dense(256, activation='relu'))
            model.add(Dense(128, activation='relu'))
            model.add(Dense(1, activation='sigmoid'))
            return model

        # GAN 모델 정의
        def build_gan(generator, discriminator):
            discriminator.trainable = False
            model = Sequential()
            model.add(generator)
            model.add(discriminator)
            return model

        # 모델 및 하이퍼파라미터 설정
        latent_dim = 100  # 잠재 공간의 차원
        img_shape = (200, 200, 3)  # 이미지의 형태

        generator = build_generator(latent_dim, img_shape)
        discriminator = build_discriminator(img_shape)

        discriminator.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
        discriminator.trainable = False

        gan = build_gan(generator, discriminator)
        gan.compile(optimizer=Adam(), loss='binary_crossentropy')

        # GAN 모델 학습
        output_dir = "rnn_app/gan_create_image"
        total_epochs = 100  # 총 에폭 수

        for epoch in range(total_epochs):
            # 각 에폭마다 이미지 데이터를 이용하여 GAN 모델 학습
            for _ in range(len(train_data)):
                real_images, labels = train_data.next()

                # 판별자 훈련
                noise = np.random.normal(0, 1, (batch_size, latent_dim))
                generated_images = generator.predict(noise)

                d_loss_real = discriminator.train_on_batch(real_images, np.ones((batch_size, 1)))
                d_loss_fake = discriminator.train_on_batch(generated_images, np.zeros((batch_size, 1)))
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

                # 생성자 훈련
                noise = np.random.normal(0, 1, (batch_size, latent_dim))
                g_loss = gan.train_on_batch(noise, np.ones((batch_size, 1)))

                # 마지막 에폭에서만 이미지를 생성하고 저장
                if epoch == (total_epochs - 1):
                    # 생성된 이미지 저장
                    noise = np.random.normal(0, 1, (1, latent_dim))
                    generated_image = generator.predict(noise)[0] * 255.0

                    image_to_save = generated_image.astype(np.uint8)
                    image = Image.fromarray(image_to_save).resize((200, 200), Image.LANCZOS)
                    image_path = os.path.join(output_dir, f"generated_image_final_epoch.png")
                    image.save(image_path)

                    print(f"generated_image_final_epoch.png 이미지를 {output_dir} 디렉토리에 저장했습니다.")
                    break

        # 새로운 이미지 로드 및 전처리
        image_files = [f for f in os.listdir(save_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
        random_image_file = random.choice(image_files)
        selected_image_path = os.path.join(save_path, random_image_file)
        new_image = Image.open(selected_image_path)
        new_image = new_image.resize((200, 200))  # 이미지 크기 조정
        new_image = np.array(new_image) / 255.0  # 이미지 스케일링

        # GAN 모델 로드 및 이미지 생성
        latent_dim = 100
        noise = np.random.normal(0, 1, (1, latent_dim))  # 랜덤 노이즈 생성
        generated_image = generator.predict(noise)[0] * 255.0  # 이미지 생성 및 스케일링

        # 생성된 이미지와 새로운 이미지의 채널 수 맞추기
        if new_image.shape[-1] != generated_image.shape[-1]:
            new_image = np.stack([new_image] * generated_image.shape[-1], axis=-1)

        # 이미지의 shape를 출력하여 확인
        print(f"Generated image shape: {generated_image.shape}")
        print(f"New image shape: {new_image.shape}")

        # 이미지의 shape가 맞지 않으면 resize를 통해 맞춤
        if generated_image.shape != new_image.shape:
            new_image = np.array(
                Image.fromarray(new_image.astype(np.uint8)).resize((generated_image.shape[1], generated_image.shape[0]),
                                                                   Image.LANCZOS))
            print(f"Resized new image shape: {new_image.shape}")

        # shape가 맞는지 다시 확인
        print(f"After resize, new image shape: {new_image.shape}")

        # 생성된 이미지와 새로운 이미지를 결합하여 새로운 이미지 생성
        blended_image = np.clip((generated_image + new_image) / 2.0, 0.0, 255.0)  # 두 이미지를 합쳐서 특정 작업 수행

        # 새로운 이미지 저장
        output_dir = "rnn_app/final_emotiart_image"
        os.makedirs(output_dir, exist_ok=True)
        modified_image = Image.fromarray(blended_image.astype(np.uint8))
        modified_image.save(os.path.join(output_dir, 'modified_image.png'))

        # 이미지 변환 작업 완료
        print("이미지 변환 작업이 완료되었습니다.")

        # 이미지를 바로 클라이언트에 반환
        img_io = io.BytesIO()
        modified_image.save(img_io, format='PNG')
        img_io.seek(0)
        # # 저장한 이미지를 읽어와서 반환
        # image_data_list = []
        # final_image_path = "rnn_app/final_emotiart_image"
        # for filename in os.listdir(final_image_path):
        #     if os.path.isfile(os.path.join(final_image_path, filename)):
        #         with open(os.path.join(final_image_path, filename), 'rb') as file:
        #             image_data = file.read()
        #             image_data_list.append(image_data)
        #
        # if image_data_list:
        #     # 이미지 데이터를 합쳐서 반환
        #     combined_image_data = b''.join(image_data_list)
        return FileResponse(img_io, content_type='image/png')
        #else:
        #return HttpResponseNotFound('No images found in the final_emotiart_image directory.')
    else:
        return JsonResponse({'error': 'No image found in the request or invalid request method.'}, status=400)

