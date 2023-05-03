import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import random
import time
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import pickle
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.callbacks import TensorBoard

TRAIN_DIRECTORY = r".\Train"
TEST_DIRECTORY = r".\Test"
IMG_SIZE = 100
CATEGORIES = []
NAME = f"cats_and_dogs_prediction-{int(time.time())}"

def get_categories(root):
    global CATEGORIES

    CATEGORIES = [c for c in os.listdir(root)]
    print(f"\nNúmero de classes: {len(CATEGORIES)}")
    print(f"Classes: {CATEGORIES}")

def generate_pkl_files(root, mode="train"):
    data = []
    for category in CATEGORIES:
        folder = os.path.join(root, category)
        label = CATEGORIES.index(category)
        print(f"Label {label}: {category}")
        for img in os.listdir(folder):
            img_path = os.path.join(folder, img)
            img_arr = cv.imread(img_path)
            if img_arr is None:
                print(img_path)
            else:
                img_arr = cv.resize(img_arr, (IMG_SIZE, IMG_SIZE))
                data.append([img_arr, label])
    else:
        random.shuffle(data)
        print(f"Tamanho do dataset: {len(data)}")

    x = []
    y = []
    for features, labels in data:
        x.append(features)
        y.append(labels)
    else:
        x = np.array(x)
        y = np.array(y)
        pickle.dump(x, open(f'x_{mode}.pkl', 'wb'))
        pickle.dump(y, open(f'y_{mode}.pkl', 'wb'))
        print("Arquivos pkl gerados.")


def load_pickle_train_files():
    x_train = pickle.load(open('x_train.pkl', 'rb'))
    y_train = pickle.load(open('y_train.pkl', 'rb'))
    print(f"Arquivos pkl TRAIN carregados. X_TRAIN: {x_train.shape}, Y_TRAIN: {y_train.shape}")

    return x_train, y_train

def load_pickle_test_files():
    x_test = pickle.load(open('x_test.pkl', 'rb'))
    y_test = pickle.load(open('y_test.pkl', 'rb'))
    print(f"Arquivos pkl TEST carregados. X_TEST: {x_test.shape}, Y_TEST: {y_test.shape}")

    return x_test, y_test

def train_model(x_train, y_train, x_test, y_test, board=False):
    x_train = x_train / 255
    x_test = x_test / 255

    model_ = Sequential([
        Conv2D(64, (3, 3), activation='relu', input_shape=x_train.shape[1:]),
        MaxPooling2D((2, 2)),

        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),

        Flatten(),
        Dense(128, activation='relu'),
        Dense(128, activation='relu'),
        Dense(1, activation='sigmoid')
        # Dense(2, activation='softmax')
    ])

    print("Compilando o modelo...")
    model_.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # model_.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    print("Iniciando o treinamento...")
    if board:
        tensorboard = TensorBoard(log_dir=f'Logs\\{NAME}\\')
        model_.fit(x_train, y_train, epochs=40, batch_size=64, callbacks=[tensorboard])
    else:
        model_.fit(x_train, y_train, epochs=40, batch_size=64)

    print("Salvando modelo...")
    model_.save(f'.\\Models\\cats_and_dogs')

    validation(model_, x_test, y_test)

    return model_

def validation(model_, x_test, y_test):
    print("Iniciando a validação...")
    model_.evaluate(x_test, y_test)

def prediction(model_, x, y):
    idx = random.randint(0, len(y))
    plt.imshow(x[idx, :])
    plt.show()

    t = time.time()
    y_pred = model_.predict(x[idx, :].reshape(1, IMG_SIZE, IMG_SIZE, 3))
    print(f"\nTempo de predição: {time.time() - t:.2f}s")

    idx = int(np.round(y_pred[0][0]))
    print(f"Resultado da predição: {CATEGORIES[idx]}")


if __name__ == '__main__':
    get_categories(TRAIN_DIRECTORY)
    if not {'x_train.pkl', 'y_train.pkl'}.issubset(set(os.listdir())):
        generate_pkl_files(TRAIN_DIRECTORY)
    if not {'x_test.pkl', 'y_test.pkl'}.issubset(set(os.listdir())):
        generate_pkl_files(TEST_DIRECTORY, mode='test')
    X_TEST, Y_TEST = load_pickle_test_files()
    try:
        model = load_model(f'.\\Models\\cats_and_dogs')
    except:
        X_TRAIN, Y_TRAIN = load_pickle_train_files()
        model = train_model(X_TRAIN, Y_TRAIN, X_TEST, Y_TEST)
        validation(model, X_TEST, Y_TEST)

    for i in range(10):
        prediction(model, X_TEST, Y_TEST)
