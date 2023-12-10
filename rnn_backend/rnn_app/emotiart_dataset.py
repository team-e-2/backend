import os

import gan
import numpy as np
from PIL import Image
from keras.layers import Dense, Flatten, Reshape
from keras.models import Sequential
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

# 이미지 데이터셋 경로
data_dir = r'data_image'

# 이미지 데이터를 읽고 전처리하는 ImageDataGenerator 생성
datagen = ImageDataGenerator(rescale=1./255)  # 이미지를 0과 1 사이로 스케일링

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

output_dir = r'gan_create_image'

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

# 새로운 이미지 경로
new_image_path = r'E:\PycharmProjects\backend\rnn_backend\rnn_app\frontended_save_image'

# 새로운 이미지 로드 및 전처리
new_image = Image.open(new_image_path)
new_image = new_image.resize((200,200))  # 이미지 크기 조정
new_image = np.array(new_image) / 255.0  # 이미지 스케일링

# GAN 모델 로드 및 이미지 생성
latent_dim = 100
noise = np.random.normal(0, 1, (1, latent_dim))  # 랜덤 노이즈 생성
generated_image = generator.predict(noise)[0] * 255.0  # 이미지 생성 및 스케일링

# 생성된 이미지와 새로운 이미지를 결합하여 새로운 이미지 생성
blended_image = np.clip((generated_image + new_image) / 2.0, 0.0, 255.0)  # 두 이미지를 합쳐서 특정 작업 수행

# 새로운 이미지 저장
output_dir = r'fianl_emotiart_image'
os.makedirs(output_dir, exist_ok=True)
modified_image = Image.fromarray(blended_image.astype(np.uint8))
modified_image.save(os.path.join(output_dir, 'modified_image.png'))

# 이미지 변환 작업 완료
print("이미지 변환 작업이 완료되었습니다.")


