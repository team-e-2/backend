import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.models import Model
import numpy as np
from PIL import Image

# 미리 훈련된 InceptionV3 모델을 사용
base_model = InceptionV3(weights='imagenet')
model = Model(inputs=base_model.input, outputs=base_model.get_layer('avg_pool').output)

# 이미지를 로드하고 전처리
def load_and_preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(299, 299))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

# 이미지의 특성 벡터를 추출하는 함수
def extract_features(image_path, model):
    img_array = load_and_preprocess_image(image_path)
    features = model.predict(img_array)
    return features.flatten()

# 예시 이미지와 유사한 이미지를 찾는 함수
def find_similar_images(query_image_path, image_paths, model):
    query_features = extract_features(query_image_path, model)
    similarities = []

    for path in image_paths:
        features = extract_features(path, model)
        similarity = np.dot(query_features, features) / (np.linalg.norm(query_features) * np.linalg.norm(features))
        similarities.append((path, similarity))

    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities

# 예시 이미지와 유사한 이미지 찾기
query_image_path = 'query_image.jpg'
image_directory = 'images/'

# image_directory에 있는 모든 이미지 파일 경로 가져오기
image_paths = [image_directory + f for f in os.listdir(image_directory) if f.endswith('.jpg') or f.endswith('.png')]

similar_images = find_similar_images(query_image_path, image_paths, model)

# 상위 5개의 유사한 이미지 출력
for i in range(2):
    print(f"Similar Image #{i + 1}: {similar_images[i][0]} (Similarity Score: {similar_images[i][1]})")