import epoch
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Flatten, Reshape
from keras.optimizers import Adam
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt

# 이미지 데이터셋 경로
data_dir = "C:/Users/장희수/PycharmProjects/backend/data_image"

# 이미지 데이터를 읽고 전처리하는 ImageDataGenerator 생성
datagen = ImageDataGenerator(rescale=1.0 / 255)  # 이미지를 0과 1 사이로 스케일링

# 이미지 데이터셋 불러오기
batch_size = 10
image_size = (200, 200)  # 이미지 사이즈 조정
class_mode = "categorical"  # 분류형 레이블 사용

train_data = datagen.flow_from_directory(
    data_dir, target_size=image_size, batch_size=batch_size, class_mode=class_mode
)

# Generator 모델 정의
def build_generator(latent_dim, output_shape):
    model = Sequential()
    model.add(Dense(128, input_dim=latent_dim, activation="relu"))
    model.add(Dense(256, activation="relu"))
    model.add(Dense(np.prod(output_shape), activation="sigmoid"))
    model.add(Reshape(output_shape))
    return model


# Discriminator 모델 정의
def build_discriminator(input_shape):
    model = Sequential()
    model.add(Flatten(input_shape=input_shape))
    model.add(Dense(256, activation="relu"))
    model.add(Dense(128, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))
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

discriminator.compile(
    optimizer=Adam(), loss="binary_crossentropy", metrics=["accuracy"]
)
discriminator.trainable = False

gan = build_gan(generator, discriminator)
gan.compile(optimizer=Adam(), loss="binary_crossentropy")

# GAN 모델 학습
epochs = 1000  # 학습 횟수
num_images_to_save = 1  # 저장할 이미지 개수

for epoch in range(epochs):
    for _ in range(len(train_data)):
        real_images, labels = train_data.next()

        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        generated_images = generator.predict(noise)

        real_labels = np.ones((batch_size, 1))
        fake_labels = np.zeros((batch_size, 1))

        d_loss_real = discriminator.train_on_batch(real_images, real_labels)
        d_loss_fake = discriminator.train_on_batch(generated_images, fake_labels)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        valid_labels = np.ones((batch_size, 1))
        g_loss = gan.train_on_batch(noise, valid_labels)

    # 각 epoch마다 이미지를 생성하고 저장
    for i in range(num_images_to_save):
        noise = np.random.normal(0, 1, (1, latent_dim))  # 각기 다른 노이즈 생성
        generated_image = generator.predict(noise)[0] * 255.0  # 이미지 생성 및 스케일링

        # 디렉토리 경로 설정
        output_dir = "C:/Users/장희수/PycharmProjects/backend/save_new_image"

        # 디렉토리가 존재하지 않으면 생성
        # os.makedirs(output_dir, exist_ok=True)

        # 이미지 크기 조정 및 uint8 형식으로 변환하여 저장
        image_to_save = generated_image.astype(np.uint8)
        image = Image.fromarray(image_to_save).resize(
            (200, 200), Image.LANCZOS
        )  # 이미지 크기 조정
        image_path = os.path.join(
            output_dir, f"new_image_{epoch + 1}_{i + 1}.png"
        )  # epoch와 순서대로 파일명 지정
        image.save(image_path)

        print(f"{num_images_to_save}개의 이미지를 {output_dir} 디렉토리에 64x64 크기로 저장했습니다.")


# 이후에 test 이미지에 감정을 추가하는 서비스를 구현할 수 있습니다.
# 예를 들어, 생성된 이미지를 통해 감정을 추가하거나 특정 감정을 나타내는 이미지를 생성할 수 있습니다.
# 추가적인 디테일은 이미지에 어떤 형태로 감정을 추가하고자 하는지에 따라 다를 수 있습니다.
