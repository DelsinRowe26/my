#подключение библиотек для обучения нейронной сети
from sys import path
from unicodedata import name
import cv2
import numpy as np
import os
from random import shuffle
from numpy import testing
from tqdm import tqdm
import datetime

#Создаем переменную пути к тренировочной папке и присваеваем путь
TRAIN_DIR = 'train'
#Создаем переменную пути к тестовой папке и присваеваем путь
TEST_DIR= 'test'
#Переменная содержащая размер картинки, если картинка будет превосходить данный размер, то будет сжиматься картинка
IMG_SIZE = 50
#Переменная обозначаюая скорость обучения нейронной сети
LR = 1e-3

#Переменная создающая модель 
MODEL_NAME = 'goodvsbad-{}-{}.model'.format(LR,'6conv-basic')


#определение меток изображений 
def label_img(img):
    word_label = img.split('.')[-3]
    if word_label == 'goodw': return [1,0]
    elif word_label == 'bad':return[0,1]

#создание набора обучающих данных, отображение изображения в оттенках серого с помощью opencv, перемешивание файлов и помещение их в массив numpy, организованный по метке/изображению
def create_train_data():
    training_data =[]
    for img in tqdm(os.listdir(TRAIN_DIR)):
        label = label_img(img)
        path = os.path.join(TRAIN_DIR,img)
        img= cv2.resize(cv2.imread(path,cv2.IMREAD_GRAYSCALE),(IMG_SIZE,IMG_SIZE))
        training_data.append([np.array(img),np.array(label)])
    shuffle(training_data)
    np.save('train_data.npy', training_data)
    return training_data

#тот же процесс с набором данных проверки, но нет необходимости перемешивать, а фотографии пронумерованы без меток, поэтому массив numpy представляет собой число/изображение
def process_test_data():
    testing_data =[]
    for img in tqdm(os.listdir(TEST_DIR)):
        img_num = img.split('.')[0]
        path = os.path.join(TEST_DIR,img)
        img= cv2.resize(cv2.imread(path,cv2.IMREAD_GRAYSCALE),(IMG_SIZE,IMG_SIZE))
        testing_data.append([np.array(img),np.array(img_num)])
    np.save('test_data.npy', testing_data)
    return testing_data

if os.path.isfile('train_data.npy'):
    #если массив numpy уже существует
    train_data = np.load('train_data.npy', allow_pickle=True)
else:
    #вызов создания набора данных изображения
    train_data = create_train_data()


#подключаем дополнительные библиотеки для обучения
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression


import tensorflow as tf
tf.compat.v1.reset_default_graph()

#Преобразование изображения в 2D массив, и обучение нейронки
convnet = input_data(shape=[None,IMG_SIZE,IMG_SIZE,1], name='input')

convnet = conv_2d(convnet,32,2, activation='relu')
convnet = max_pool_2d(convnet,2)

convnet = conv_2d(convnet,64,2, activation='relu')
convnet = max_pool_2d(convnet,2)

convnet = conv_2d(convnet,32,2, activation='relu')
convnet = max_pool_2d(convnet,2)

convnet = conv_2d(convnet,64,2, activation='relu')
convnet = max_pool_2d(convnet,2)
convnet = conv_2d(convnet,32,2, activation='relu')
convnet = max_pool_2d(convnet,2)

convnet = conv_2d(convnet,64,2, activation='relu')
convnet = max_pool_2d(convnet,2)

convnet = fully_connected(convnet,1024, activation='relu')
convnet = dropout(convnet,0.8)

convnet = fully_connected(convnet,2, activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

#Создаем лог файл после обучения
model = tflearn.DNN(convnet, tensorboard_dir='log')

#Проверяем есть ли файл можели в папке, если есть то загружаем его, если нету создаем новую
if os.path.exists('{}.meta'.format(MODEL_NAME)):
    model.load(MODEL_NAME)
    print( 'model loaded')
else:
    train = train_data[:-500]
    test = train_data[-500:]

    X = np.array([i[0] for i in train]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    Y = [i[1] for i in train]

    test_x = np.array([i[0] for i in test]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    test_y = [i[1] for i in test]

    model.fit({'input': X}, {'targets': Y}, n_epoch=5, validation_set=({'input': test_x}, {'targets': test_y}), 
    snapshot_step=500, show_metric=True, run_id=MODEL_NAME)


#Проверка существует ли файл, если да то загружает его, если нет то создает новый
if os.path.isfile('test_data.npy'):
    test_data = np.load('test_data.npy', allow_pickle=True)
else:
    test_data = process_test_data()

#Подключаем библиотеку для того чтобы мы могли выводить изображения
import matplotlib.pyplot as plt

#Создаем переменную кторой присваеваем фигуру для вывода
fig = plt.figure()

#Создаем цикл для проверки тестовой папки и вывода изобрадений и определения какие швы
for num, data in enumerate(test_data[:12]):
    img_num = data[1]
    img_data = data[0]

    y = fig.add_subplot(3,4, num+1)
    orig = img_data
    data =  img_data.reshape(IMG_SIZE,IMG_SIZE,1)

    model_out = model.predict([data])[0]

    if np.argmax(model_out) == 1: str_label='bad'
    else: str_label = 'goodw'

    y.imshow(orig,cmap='gray')
    plt.title(str_label)
    y.axes.get_xaxis().set_visible(False)
    y.axes.get_yaxis().set_visible(False)
plt.show()