import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from keras.preprocessing.image import img_to_array, load_img
from keras.models import load_model
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.utils import to_categorical

from sklearn.model_selection import train_test_split

import os
import re


# list_picturesがimportできなかったのでコピペ
def list_pictures(directory, ext='jpg|jpeg|bmp|png|ppm'):
    return [os.path.join(root, f)
            for root, _, files in os.walk(directory) for f in files
            if re.match(r'([\w]+\.(?:' + ext + '))', f.lower())]


# 制度の計算
def calculate_accuracy(X_test, y_test):
    network = load_model("./my_model.h5")

    y_pred_one_hot = network.predict(X_test)
    y_pred = np.argmax(y_pred_one_hot, axis=1)

    from sklearn.metrics import confusion_matrix, accuracy_score
    CM = confusion_matrix(y_test, y_pred)
    print(CM)
    acc = accuracy_score(y_test, y_pred)
    print(acc)


def my_model(input_shape, output_size):
    # CNNを構築
    model = Sequential()

    model.add(Conv2D(32, (3, 3), padding='same',
                     input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(output_size))
    model.add(Activation('softmax'))

    # コンパイル
    model.compile(loss='categorical_crossentropy',
                  optimizer='SGD',
                  metrics=['accuracy'])

    model.summary()

    return model


def learn():
    genres = ["actress", "actor", "comedian"]
    genre_size = len(genres)

    X_data = []
    Y_data = []
    for i in range(genre_size):
        print('now processing: ' + genres[i])
        for picture in list_pictures('./data/' + genres[i]):
            img = img_to_array(load_img(picture, target_size=(64, 64)))
            X_data.append(img)
            Y_data.append(i)

    X_data = np.asarray(X_data)
    Y_data = np.asarray(Y_data)

    # 画素値を0から1の範囲に変換
    X = X_data.astype('float32')
    X = X / 255.0

    # 学習用データとテストデータ
    X_train, X_test, y_train, y_test = train_test_split(X, Y_data, test_size=0.20, random_state=111)

    model = my_model(X_train.shape[1:], genre_size)

    y_train_one_hot = to_categorical(y_train)
    y_test_one_hot = to_categorical(y_test)

    # 学習
    early_stopping_cb = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=2, verbose=1, mode='auto')
    model_checkpoint_cb = ModelCheckpoint("best_network.hdf5", monitor='val_loss', verbose=1,
                                          save_best_only=True, save_weights_only=False, mode='auto', period=1)

    history = model.fit(X_train, y_train_one_hot, batch_size=100,
                        epochs=1, verbose=1, validation_data=(X_test, y_test_one_hot),
                        callbacks=[early_stopping_cb, model_checkpoint_cb], )
    # モデルを保存
    model.save("my_model.h5")

    # 汎化制度の評価・表示
    score = model.evaluate(X_test, y_test_one_hot, batch_size=32, verbose=0)
    print('validation loss:{0[0]}\nvalidation accuracy:{0[1]}'.format(score))

    # acc, val_accのプロット
    plt.plot(history.history["acc"], label="acc", ls="-", marker="o")
    plt.plot(history.history["val_acc"], label="val_acc", ls="-", marker="x")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(loc="best")
    plt.show()


if __name__ == '__main__':
    learn()
