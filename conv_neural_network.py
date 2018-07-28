import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import keras
from keras import backend as K
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.models import Sequential
from elastic_transformation import elastic_transform
from keras.preprocessing.image import ImageDataGenerator
from datetime import timedelta
import cv2
import time

start = time.time()

sess = tf.Session()
with tf.Session() as sess:

            train_file = './mnist_train.csv'
            test_file = './mnist_test.csv'

            trainSet = np.loadtxt(train_file, delimiter=',')
            testSet = np.loadtxt(test_file, delimiter=',')

            train_data = trainSet[:, 1:785]
            train_labels = trainSet[:, :1]

            test_data = testSet[:, 1:785]
            test_labels = testSet[:, :1]

            def one_hot(data):
                one_hot = []
                for item in data:
                    if item == 0.:
                        one_h = [1., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
                    elif item == 1.:
                        one_h = [0., 1., 0., 0., 0., 0., 0., 0., 0., 0.]
                    elif item == 2.:
                        one_h = [0., 0., 1., 0., 0., 0., 0., 0., 0., 0.]
                    elif item == 3.:
                        one_h = [0., 0., 0., 1., 0., 0., 0., 0., 0., 0.]
                    elif item == 4.:
                        one_h = [0., 0., 0., 0., 1., 0., 0., 0., 0., 0.]
                    elif item == 5.:
                        one_h = [0., 0., 0., 0., 0., 1., 0., 0., 0., 0.]
                    elif item == 6.:
                        one_h = [0., 0., 0., 0., 0., 0., 1., 0., 0., 0.]
                    elif item == 7.:
                        one_h = [0., 0., 0., 0., 0., 0., 0., 1., 0., 0.]
                    elif item == 8.:
                        one_h = [0., 0., 0., 0., 0., 0., 0., 0., 1., 0.]
                    elif item == 9.:
                        one_h = [0., 0., 0., 0., 0., 0., 0., 0., 0., 1.]

                    one_hot.append(one_h)
                one_hot=np.array(one_hot)
                return one_hot

            train_labels_one_hot = one_hot(train_labels)
            test_labels_one_hot = one_hot(test_labels)

print('Димензии на тренинг множеството: ', train_data.shape, train_labels.shape)
print('Димензии на тест множеството: ', test_data.shape, test_labels.shape)

#Печатење на бројот на класи и нивните уникатни вредности
classes = np.unique(train_labels) #0...9
numClasses = len(classes) #10
print('Вкупен број на класи: ', numClasses)
print('Уникатни вредности на излезните класи: ', classes)

#Промена на обликот на инстанците од едно димензионална низа (784) во матрица од 28x28 пиксели
train_images = train_data.reshape(train_data.shape[0],  28, 28)
test_images = test_data.reshape(test_data.shape[0], 28, 28)

plt.figure(figsize=[10, 5])


#Приказ на првите две слики од тренинг множеството заедно со точната класа
plt.subplot(221)
plt.imshow(train_images[0, :, :], cmap='gray')
plt.title("Точна класа: {}".format(int(train_labels[0])))
plt.axis('off')

plt.subplot(222)
plt.imshow(train_images[1, :, :], cmap='gray')
plt.title("Точна класа: {}".format(int(train_labels[1])))
plt.axis('off')



elastic = False
#Предобработка---------------------------------------------------------------
# -----(elastic distortions , 2D input)------
if elastic:
    print('Се применува еластична деформација над множеството.')
    for index, img in enumerate(train_images):
        img = elastic_transform(img, 36, 10)
        train_images[index] = img
    for index, img in enumerate(test_images):
        img = elastic_transform(img, 36, 10)
        test_images[index] = img
#Предобработка---------------------------------------------------------------



#Приказ на првите две слики од тренинг множеството заедно со точната класа
plt.subplot(223)
plt.imshow(train_images[0, :, :], cmap='gray')
plt.title("Точна класа: {}".format(int(train_labels[0])))
plt.axis('off')

plt.subplot(224)
plt.imshow(train_images[1, :, :], cmap='gray')
plt.title("Точна класа: {}".format(int(train_labels[1])))
plt.axis('off')
plt.show()


#димензии на влезните слики
img_rows, img_cols = 28, 28

if K.image_data_format() == 'channels_first':
    train_images = train_images.reshape(train_images.shape[0], 1, img_rows, img_cols) #train_images.shape[0]  60000
    test_images = test_images.reshape(test_images.shape[0], 1, img_rows, img_cols)    #test_images.shape[0]   10000
    input_shape = (1, img_rows, img_cols)

else:
    train_images = train_images.reshape(train_images.shape[0], img_rows, img_cols, 1)
    test_images = test_images.reshape(test_images.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

#input_shape влезот во првиот слој од невронската мрежа
print('input shape:', input_shape)
print('train shape:', train_images.shape)
print('test shape:', test_images.shape)
#print(train_data.shape[0], 'train samples')
#print(test_data.shape[0], 'test samples')


#Промена на типот на вредностите за пикселите од цели во децимални броеви
train_images = train_images.astype('float32')
test_images = test_images.astype('float32')

#Min-max нормализација на податоците (скалирање во рангот помеѓу 0 и 1)
def normalization(m):
    col_max = m.max(axis=0)
    col_min = m.min(axis=0)
    return (m-col_min)/(col_max - col_min)

train_images = np.nan_to_num(normalization(train_images))
test_images = np.nan_to_num(normalization(test_images))

#Печатење на класата на една од инстанците, како и категориската конверзија на класата
print('Класа на првата инстанца во тренинг множеството: ', train_labels[0])
print('Конверзија на класта во категориски атрибут (one-hot): ', train_labels_one_hot[0])


model = Sequential()
#padding=same го поставуваме со цел димензијата на излезот да биде едаква со онаа на влезот
#padding-от на влезната слика ќе зависи од големината на кернелот
model.add(Conv2D(15, (4, 4), padding='same', strides=1, activation='relu', input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(800, activation='relu'))
#model.add(Dropout(0.25))

model.add(Dense(numClasses, activation='softmax'))

#data_augmentation = False
sgd = keras.optimizers.SGD(lr=0.1, momentum=0.9)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
earlyStopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=0, verbose=0, mode='auto')
history = model.fit(train_images, train_labels_one_hot, batch_size=250,callbacks=[earlyStopping],  epochs=1000, verbose=2,validation_split=0.2, shuffle=True)

'''
if not data_augmentation:
    print('Not using data augmentation.')
    # FEED THE NETWORK
    history = model.fit(train_data, train_labels_one_hot,  batch_size=250, callbacks=[earlyStopping], epochs=100,verbose=2,
                       validation_split=0.1, shuffle=True)
else:
    print('Using real-time data augmentation.')
datagen = ImageDataGenerator(
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images

datagen.fit(train_data)
history = model.fit_generator(datagen.flow(train_data, train_labels_one_hot, batch_size=250), callbacks=[earlyStopping], epochs=20,verbose=1)
'''
[test_loss, test_acc] = model.evaluate(test_images, test_labels_one_hot)


#print("Evaluation result on Test Data : Loss = {}, accuracy = {}".format(test_loss, test_acc))
error = ((1.0-test_acc)*100.0)
accuracy = test_acc*100
print('Accuracy of test set:', accuracy, '%')
print('Error:', error, '%')

end = time.time()
time = end - start
print("Време на извршување: " + str(timedelta(seconds=int(round(time)))))

'''
#--------------------- Поместување на центрираните цифри -------------------------
img = test_images[4, :, :]
trueClass = testSet[4, :1]

up = np.float32([[1, 0, 3], [0, 1, -5]])
left = np.float32([[1, 0, -2], [0, 1, 0]])
leftDown = np.float32([[1, 0, -3], [0, 1, 1]])
rightUp = np.float32([[1, 0, 5], [0, 1, 0]])

img1 = cv2.warpAffine(img, up, (img.shape[1], img.shape[0]))
img2 = cv2.warpAffine(img, left, (img.shape[1], img.shape[0]))
img3 = cv2.warpAffine(img, leftDown, (img.shape[1], img.shape[0]))
img4 = cv2.warpAffine(img, rightUp, (img.shape[1], img.shape[0]))

img1Arr = (img1.reshape(1, 28, 28, 1))
img2Arr = (img2.reshape(1, 28, 28, 1))
img3Arr = (img3.reshape(1, 28, 28, 1))
img4Arr = (img4.reshape(1, 28, 28, 1))


predictedClass1 = model.predict_classes(img1Arr)
predictedClass2 = model.predict_classes(img2Arr)
predictedClass3 = model.predict_classes(img3Arr)
predictedClass4 = model.predict_classes(img4Arr)

plt.figure(figsize=[7, 5])

plt.subplot(221)
plt.imshow(img1, cmap='gray')
plt.title("Предвидена класа: {}".format(predictedClass1))
plt.axis('off')
plt.subplot(222)
plt.imshow(img2, cmap='gray')
plt.title("Предвидена класа: {}".format(predictedClass2))
plt.axis('off')
plt.subplot(223)
plt.imshow(img3, cmap='gray')
plt.title("Предвидена класа: {}".format(predictedClass3))
plt.axis('off')
plt.subplot(224)
plt.imshow(img4, cmap='gray')
plt.title("Предвидена класа: {}".format(predictedClass4))
plt.axis('off')
plt.show()
#----------------------------------------------
'''



#Приказ на график за функцијата на загуба кај тренинг и валидациското множество
plt.figure(figsize=[8, 6])
plt.plot(history.history['loss'], 'darkgreen', linewidth=3.0)
plt.plot(history.history['val_loss'], 'blue', linewidth=3.0)
plt.legend(['Тренинг множество', 'Валидациско множество'], fontsize=18)
plt.xlabel('Број на епохи', fontsize=16)
plt.ylabel('Функција на  загуба (цена на чинење)', fontsize=16)
plt.title('Функција на загуба при тренирање на моделот', fontsize=16)

#Приказ на график за точноста кај тренинг и валидациското множество
plt.figure(figsize=[8, 6])
plt.plot(history.history['acc'], 'darkgreen', linewidth=3.0)
plt.plot(history.history['val_acc'], 'blue', linewidth=3.0)
plt.legend(['Тренинг множество', 'Валидациско множество'], fontsize=18)
plt.xlabel('Број на епохи', fontsize=16)
plt.ylabel('Точност при тренирање на моделот', fontsize=16)
plt.title('', fontsize=16)

plt.show()




