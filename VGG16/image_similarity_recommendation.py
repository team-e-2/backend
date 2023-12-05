import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
import numpy as np
from PIL import Image
import requests
from io import BytesIO

# 이미지 다운로드 및 전처리
def load_and_preprocess_image(image_url):
    response = requests.get(image_url)
    img = Image.open(BytesIO(response.content))
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

# VGG16 모델 로드
base_model = VGG16(weights='imagenet')

# 이미지의 특징 추출
def extract_features(img_array):
    features = base_model.predict(img_array)
    features = np.array(features).flatten()
    return features

# 예시 이미지의 특징 추출
example_image_url = "https://img.freepik.com/free-photo/closeup-shot-of-a-fluffy-ginger-domestic-cat-looking-directly-on-a-white-background_181624-46543.jpg"
example_img_array = load_and_preprocess_image(example_image_url)
example_features = extract_features(example_img_array)

# 여러 그림 중 유사한 그림 찾기 (예시로 동일한 이미지 사용)
similar_image_url = "https://img.freepik.com/free-photo/closeup-shot-of-a-fluffy-ginger-domestic-cat-looking-directly-on-a-white-background_181624-46543.jpg"
similar_img_array = load_and_preprocess_image(similar_image_url)
similar_features = extract_features(similar_img_array)

# 유사도 계산
similarity_score = np.dot(example_features, similar_features) / (np.linalg.norm(example_features) * np.linalg.norm(similar_features))

print(f"Similarity Score: {similarity_score}")
