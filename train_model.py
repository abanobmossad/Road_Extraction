from datetime import datetime
import numpy as np
import tensorflow as tf
import time

# calculate the total time for training 800000 line
stratTotalTime=time.time()

#print every step
tf.logging.set_verbosity(tf.logging.INFO)

print(str(datetime.now()) + ': loading data files')
# Data sets
trainDataFileName = 'airs-dataset/train.csv'
testDataFileName = 'airs-dataset/test.csv'
validationDataFileName = 'airs-dataset/valid.csv'
# Load datasets.
trainData = tf.contrib.learn.datasets.base.load_csv_without_header(
    filename=trainDataFileName,
    target_dtype=np.int,
    features_dtype=np.int)
testData = tf.contrib.learn.datasets.base.load_csv_without_header(
    filename=testDataFileName,
    target_dtype=np.int,
    features_dtype=np.int)

trainingSteps = 1000
validation_monitor = tf.contrib.learn.monitors.ValidationMonitor(
    testData.data,
    testData.target,
    every_n_steps=50)

featureColumns = [tf.contrib.layers.real_valued_column("", dimension=79)]
hiddenUnits = [100, 150, 100, 50]
classes = 3
modelDir = 'model'
classifierConfig = tf.contrib.learn.RunConfig(save_checkpoints_secs = None, save_checkpoints_steps = trainingSteps)

classifier = tf.contrib.learn.DNNClassifier(feature_columns = featureColumns,
                                                hidden_units = hiddenUnits,
                                                n_classes = classes,
                                                model_dir = modelDir,
                                                config = classifierConfig)

# Define the training inputs

x_train = trainData.data
y_train = trainData.target


# Define the test inputs

x_test = testData.data
y_test = testData.target



print(str(datetime.now()) + ': training...')
classifier.fit(x=x_train,y=y_train,batch_size=200000,steps=20000,monitors=[validation_monitor])
print(str(datetime.now()) + ': testing...')
accuracy = classifier.evaluate(x=x_test,y=y_test,batch_size=200000, steps=1)['accuracy']
print(str(datetime.now()) + ': accuracy of testing:', accuracy*100)

# calculate the total time for training 800000 line
endTotalTime = time.time()
print("Total time for training : ",(endTotalTime-stratTotalTime)/60)
