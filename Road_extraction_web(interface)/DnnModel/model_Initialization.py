from datetime import datetime
import tensorflow as tf
import os

from os.path import realpath, join, dirname

MODEL_PATH = join(dirname(realpath(__file__)), "model")


def initializing_model():
    print()
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    print(str(datetime.now()) + ': initializing model...')
    featureColumns = [tf.contrib.layers.real_valued_column("", dimension=75)]
    hiddenUnits = [100, 150, 100, 50]
    classes = 3
    classifier = tf.contrib.learn.DNNClassifier(feature_columns=featureColumns,
                                                hidden_units=hiddenUnits,
                                                n_classes=classes,
                                                model_dir=MODEL_PATH)

    return classifier
