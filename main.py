import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import img_to_array, array_to_img
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image

# 데이터 다운로드 함수
def download_quickdraw_dataset():
    class_names = ["apple", "banana", "cat", "dog", "fish"]  # 사용하고자 하는 클래스 이름으로 수정
    (train_images, train_labels), (_, _) = tf.keras.datasets.fashion_mnist.load_data()

    # 간단한 전처리 (이미지 크기 조절 등)
    train_images = np.expand_dims(train_images / 255.0, axis=-1)
    return train_images, train_labels

# 모델 구축 함수
def build_model(input_shape, num_classes):
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
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
    model = build_model(input_shape, num_classes)
    model.fit(dataset_images, dataset_labels, epochs=5, validation_split=0.2)

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
    img = Image.open(image_path).convert('L')  # 이미지를 흑백으로 변환
    img = img.resize((28, 28))  # 모델의 입력 크기에 맞게 조절
    img_array = np.array(img) / 255.0  # 이미지를 0과 1 사이의 값으로 정규화
    img_array = np.expand_dims(img_array, axis=0)  # 배치 차원 추가
    img_array = np.expand_dims(img_array, axis=-1)  # 색상 채널 차원 추가
    return img_array
# 데이터 다운로드
dataset_images, dataset_labels = download_quickdraw_dataset()

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


