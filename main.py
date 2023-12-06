import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import img_to_array, array_to_img
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image, ImageDraw
import ndjson

# Quick, Draw! 데이터셋 경로
dataset_path = "rnn_tutorial_data"

# 클래스명 정의 (원하는 클래스들로 수정)
class_names = ["cat"]

# 데이터 로드 함수
def load_quickdraw_data(dataset_path, class_names):
    images = []
    labels = []

    for i, class_name in enumerate(class_names):
        file_path = os.path.join(dataset_path, f"{class_name}.ndjson")

        with open(file_path) as f:
            data = ndjson.load(f)

        for entry in data:
            img = Image.new('L', (256, 256), 255)  # 백색 이미지 생성
            draw = ImageDraw.Draw(img)
            for stroke in entry['drawing']:
                draw.line(list(zip(stroke[0], stroke[1])), fill=0, width=5)

            img = img.resize((28, 28))
            img_array = img_to_array(img)
            images.append(img_array)
            labels.append(i)

    return np.array(images), np.array(labels)

# 모델 구축 함수 (간소화)
def build_simple_model(input_shape, num_classes):
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(num_classes, activation='softmax'))

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model

# 유사한 그림 찾기 함수
def find_similar_images(input_image, dataset_images, dataset_labels):
    # 모델 학습
    input_shape = input_image.shape[1:]
    num_classes = len(np.unique(dataset_labels))

    # 저장된 모델 불러오기 또는 새로 학습
    model_path = "saved_model.h5"
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

# 예시: 입력된 그림과 유사한 그림 찾기
input_image_path = "cat.png"
input_image = load_and_preprocess_input_image(input_image_path)
similar_images = find_similar_images(input_image, dataset_images, dataset_labels)

# 결과 출력
print("Input Image:")
Image.open(input_image_path).show()

print("\nSimilar Images:")
for i, img_array in enumerate(similar_images):
    img = Image.fromarray((img_array.squeeze() * 255).astype(np.uint8), mode='L')
    img.show()
