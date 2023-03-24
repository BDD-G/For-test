
import tensorflow as tf
from keras import backend as K
from tensorflow.keras.layers import Input, Conv2D, Dense
from tensorflow.keras.layers import MaxPool2D, AvgPool2D, GlobalAvgPool2D, GlobalMaxPool2D, ZeroPadding2D
from tensorflow.keras.layers import Dropout, BatchNormalization, Flatten, Concatenate
from tensorflow.keras.activations import relu, sigmoid, tanh, elu, softmax, linear
from tensorflow.keras.models import Model, save_model, load_model
from tensorflow.keras.optimizers import SGD, Adam, RMSprop
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.losses import categorical_crossentropy, binary_crossentropy
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.metrics import categorical_accuracy as accuracy
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score, auc, precision_recall_curve
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input as pre_mobnet
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input as pre_vgg



def my_model(height, width, channels=3):
    lyr_in = Input(shape=(height, width, channels))
    # lyr_in = Input(shape=(height, width, channels))
    # lyr = ZeroPadding2D(padding=(1, 1))(lyr_in)
    lyr = Conv2D(filters=128, kernel_size=(3, 3), activation='relu', strides=(2, 2))(lyr_in)
    # lyr = MaxPool2D(pool_size=2)(lyr)
    # lyr = AvgPool2D(pool_size=2)(lyr)

    # lyr = ZeroPadding2D(padding=(1, 1))(lyr)
    lyr = Conv2D(filters=256, kernel_size=3, activation='relu', strides=(2, 2))(lyr)
    # lyr = MaxPool2D(pool_size=2)(lyr)
    # lyr = AvgPool2D(pool_size=2)(lyr)

    # lyr = ZeroPadding2D(padding=(1, 1))(lyr)
    lyr = Conv2D(filters=256, kernel_size=3, activation='relu', strides=(1, 1), padding='same')(lyr)
    # lyr = Dropout(rate=0.5)(lyr)
    lyr = MaxPool2D(pool_size=2)(lyr)
    # lyr = AvgPool2D(pool_size=2)(lyr)

    lyr = Flatten()(lyr)
    # lyr = GlobalAvgPool2D()(lyr)
    # lyr = GlobalMaxPool2D()(lyr)
    # lyr_mp = Flatten()(lyr_mp)
    # lyr = Concatenate()([lyr_mp, lyr])
    # lyr = Dense(units=512, activation='relu')(lyr)
    # lyr = Dropout(rate=0.5)(lyr)
    lyr_out = Dense(units=1, activation='linear', dtype=tf.float64)(lyr)

    m = Model(lyr_in, lyr_out)
    return m

def vgg16(height, width, num_class):
    # base = MobileNetV2(input_shape=(height, width, 3), weights='imagenet', classes=num_class, include_top=False)
    base = VGG16(input_shape=(height, width, 3), classes=num_class, include_top=False)
    # for L in base.layers:
    #     L.trainable = False
    # lyr = Conv2D(filters=3, activation='relu', kernel_size=3, padding='same')(lyr_in)
    base_out = base.output
    lyr = GlobalAvgPool2D()(base_out)
    lyr = Dense(512, activation='relu')(lyr)
    lyr_out = Dense(units=num_class, activation='softmax')(lyr)

    return Model(base.inputs, lyr_out)



def mob_net2(num_class, height, width, chan):
    base = MobileNetV2(input_shape=(height, width, 3), weights='imagenet', classes=num_class, include_top=False)
    base_out = base.output
    # lyr = Conv2D(filters=3, activation='relu', kernel_size=3, padding='same')(lyr_in)
    lyr = GlobalAvgPool2D()(base_out)
    lyr = Dense(512, activation='relu')(lyr)
    lyr_out = Dense(units=num_class, activation='softmax')(lyr)

    return Model(base.inputs, lyr_out)



def vit():
    pass

def cnn_model(model_name, height, width, num_class):
    if model_name == 'vgg':
        return vgg16(height, width, num_class)
    elif model_name == 'mobnet':
        return mob_net2(num_class=num_class, height=height, width=width, chan=3)
    elif model_name == 'densenet':
        pass


def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall_keras = true_positives / (possible_positives + K.epsilon())
    return recall_keras


def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision_keras = true_positives / (predicted_positives + K.epsilon())
    return precision_keras


def specificity(y_true, y_pred):
    tn = K.sum(K.round(K.clip((1 - y_true) * (1 - y_pred), 0, 1)))
    fp = K.sum(K.round(K.clip((1 - y_true) * y_pred, 0, 1)))
    return tn / (tn + fp + K.epsilon())


def negative_predictive_value(y_true, y_pred):
    tn = K.sum(K.round(K.clip((1 - y_true) * (1 - y_pred), 0, 1)))
    fn = K.sum(K.round(K.clip(y_true * (1 - y_pred), 0, 1)))
    return tn / (tn + fn + K.epsilon())


def f1(y_true, y_pred):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return 2 * ((p * r) / (p + r + K.epsilon()))

