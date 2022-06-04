import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import time
import os  #импортирање на оперативниот систем
import keras as keras
from keras import backend as K
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.models import Sequential
from datetime import timedelta

start = time.time()

print('Вчитување на податоците....')

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
        if item == 0:
            one_h = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        elif item == 1:
            one_h = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
        elif item == 2:
            one_h = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
        elif item == 3:
            one_h = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
        elif item == 4:
            one_h = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
        elif item == 5:
            one_h = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
        elif item == 6:
            one_h = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
        elif item == 7:
            one_h = [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
        elif item == 8:
            one_h = [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
        elif item == 9:
            one_h = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]

        one_hot.append(one_h)
    one_hot = np.array(one_hot)
    return one_hot


train_labels_one_hot = one_hot(train_labels)
test_labels_one_hot = one_hot(test_labels)


def plot_images(images, trueClass, ensembleClassPrediction=None, bestClassPrediction=None):
    fig, axes = plt.subplots(3, 3)
    if ensembleClassPrediction is None:
        hspace = 0.3
    else:
        hspace = 1.0
    fig.subplots_adjust(hspace=hspace, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        if i < len(images):
            ax.imshow(images[i, :, :], cmap='gray')
            if ensembleClassPrediction is None:
                label = "Точна класа: {0}".format(int(trueClass[i]))
            else:
                msg = "Точна класа: {0}\nАнсамбл од мрежи: {1}\nНајдобра мрежа: {2}"
                label = msg.format(format(int(trueClass[i])),
                                   ensembleClassPrediction[i],
                                   bestClassPrediction[i])
            ax.set_xlabel(label)
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()


img_rows, img_cols = 28, 28
train_images = train_data.reshape(train_data.shape[0], img_rows, img_cols)
test_images = test_data.reshape(test_data.shape[0], img_rows, img_cols)
#plot_images(train_images[0:9], train_labels[0:9])

if K.image_data_format() == 'channels_first':
    train_images = train_images.reshape(train_images.shape[0], 1, img_rows,img_cols)  # train_images.shape[0]  60000
    test_images = test_images.reshape(test_images.shape[0], 1, img_rows, img_cols)  # test_images.shape[0]   10000
    input_shape = (1, img_rows, img_cols)
else:
    train_images = train_images.reshape(train_images.shape[0], img_rows, img_cols, 1)
    test_images = test_images.reshape(test_images.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

    # Промена на типот на вредностите за пикселите од цели во децимални броеви
    train_images = train_images.astype('float32')
    test_images = test_images.astype('float32')

    # Min-max нормализација на податоците (скалирање во рангот помеѓу 0 и 1)
    def normalization(m):
        col_max = m.max(axis=0)
        col_min = m.min(axis=0)
        return (m - col_min) / (col_max - col_min)

    train_images = np.nan_to_num(normalization(train_images))
    test_images = np.nan_to_num(normalization(test_images))

    print('Класа на првата инстанца во тренинг множеството: ', format(int(train_labels[0])))
    print('Конверзија на класта во категориски атрибут (one-hot): ', train_labels_one_hot[0])

    classes = np.unique(train_labels)#0...9
    numClasses = len(classes) #10
    dataDimension = np.prod(train_data.shape[1:])#784


def newModel(numClasses, input_shape):

    print('Kреирање на мрежата')

    model = Sequential()
    model.add(Conv2D(15, (4, 4), padding='same', strides=1, activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(800, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(numClasses, activation='softmax'))

    sgd = keras.optimizers.SGD(lr=0.1, momentum=0.9)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    return model

netNumber = 3
print('----------------> Тренирање на %d мрежи <----------------' % netNumber)
directoryPath = './history/'

if not os.path.exists(directoryPath):
    os.makedirs(directoryPath)

def savePath(netNum):
    return directoryPath + 'network' + str(netNum)
    model.save(savePath(netNum))

if True:
    for i in range(1, netNumber+1):
        print("Невронска мрежа: {0}".format(i))
        model = newModel(numClasses, input_shape)
        print('Во тек е тренирање на мрежата....')
        earlyStopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=0, verbose=0, mode='auto')
        model.fit(train_images, train_labels_one_hot, callbacks=[earlyStopping], batch_size=250, epochs=1, verbose=2, validation_split=0.2, shuffle=True, class_weight=None, sample_weight=None)

        #Зачувување на моделот во HDF5 фајл
        model.save(savePath(i))
#-----------------до тука се тренира и зачувува секоја од мрежите


def ensemble_predictions():

    pred_labels = []
    test_accuracies = []

    for i in range(1, netNumber+1):

        model = keras.models.load_model(savePath(i))
        lossAcc = model.evaluate(test_images, test_labels_one_hot)
        print(i, '--функција на загуба за тест множество, точност за тест множество--', lossAcc)
        acc = lossAcc[1]
        test_accuracies.append(acc)
        predicetdLabels = model.predict(test_images)
        pred_labels.append(predicetdLabels)

    return np.array(test_accuracies),\
           np.array(pred_labels)

test_accuracies, pred_labels = ensemble_predictions()

end = time.time()
time = end - start
print("Време на извршување: " + str(timedelta(seconds=int(round(time)))))

print("Облик од предвидени лабели од сите мрежи", pred_labels.shape) # netNumber, 10000, 10
print("Облик од предвидена точност од сите мрежи", test_accuracies.shape)


#Ансамбл предвидување
ensemble_labelsPrediction = np.mean(pred_labels, axis=0)  #10000, 10
ensemble_classPrediction = np.argmax(ensemble_labelsPrediction, axis=1) #10000, 1

#print("test class", test_labels.shape)
#print(test_labels)
test_labels = test_labels.reshape((10000,))
#print("test class", test_labels.shape)
#print(test_labels)

ensemble_correct = (ensemble_classPrediction == test_labels)  #boolean array
ensemble_incorrect = np.logical_not(ensemble_correct)


#Најдобра-мрежа предвидување
test_accuracies #array([ 0.9893,  0.988 ,  0.9893,  0.9889,  0.9892])
print(test_accuracies)
best_net = np.argmax(test_accuracies)
print('Број на најдобра невронска мрежа:', best_net + 1)
bestNet_labelsPrediction = pred_labels[best_net, :, :]  #0, 10000, 10
bestNet_classPrediction = np.argmax(bestNet_labelsPrediction, axis=1)

bestNet_correct = (bestNet_classPrediction == test_labels) #boolean array
bestNet_incorrect = np.logical_not(bestNet_correct)

#Печатење на резултатите
print('Број на точно класифицирани инстанци преку ансамбл:', np.sum(ensemble_correct))
print('Број на точно класифицирани инстанци преку најдобраа мрежа:', np.sum(bestNet_correct))

ensemble_better = np.logical_and(bestNet_incorrect, ensemble_correct)   #[1,0,1,0,0]
print('Број на инстанци каде што ансаблот на мрежи се покажал како подобар класификатор од најдобрата мрежа:', ensemble_better.sum())

bestNet_better = np.logical_and(bestNet_correct, ensemble_incorrect)    #[1,0,1,0,0]
print('Број на инстанци каде што најдобрата мрежа се покажала како подобар класификатор од ансамблот на мрежи:', bestNet_better.sum())

print('Точност на тест множество (најдобра мрежа):', test_accuracies[best_net])
print('Точност на тест множество (ансамбл):', ensemble_correct.mean())

test_images = test_data.reshape(test_data.shape[0], img_rows, img_cols)

#Прикажување на слики_____________________________________________________________________________
def plot_images_comparison(idx):
    plot_images(images=test_images[idx, :],
                trueClass=test_labels[idx],
                ensembleClassPrediction=ensemble_classPrediction[idx],
                bestClassPrediction=bestNet_classPrediction[idx])

#Прикажување на лабели_____________________________________________________________________________
def print_labels(labels, idx, num=1):
    labels = labels[idx, :]
    labels = labels[0:num, :]
    labels_rounded = np.round(labels, 2)
    print(labels_rounded)

def print_labels_ensemble(idx, **kwargs):
    print_labels(labels=ensemble_labelsPrediction, idx=idx, **kwargs) #labels= сите излезни лабели предвидени од ансамблот
                                                                      #idx = [1,0,1,0]
                                                                      #**kwargs = можеме да го смениме бројот на лабели што ќе ги печатиме
def print_labels_best_net(idx, **kwargs):
    print_labels(labels=bestNet_labelsPrediction, idx=idx, **kwargs)  #labels= сите излезни лабели предвидени од најдобрата мрежа
                                                                      #idx = [1,0,1,0]
                                                                      #**kwargs = можеме да го смениме бројот на лабели што ќе ги печатиме
    
print_labels_ensemble(idx=ensemble_better, num=1)  #num = колкав број на лабели да печатиме (**kwargs instead)
print_labels_best_net(idx=ensemble_better, num=1)

print_labels_ensemble(idx=bestNet_better, num=1)
print_labels_best_net(idx=bestNet_better, num=1)

plot_images_comparison(idx=ensemble_better)
plot_images_comparison(idx=bestNet_better)


