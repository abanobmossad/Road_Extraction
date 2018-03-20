import os
from time import time

from flask import Flask, render_template
from os.path import realpath, dirname, join
from DnnModel.remoteSensing import RemoteSensing
from UploadImages import upload_file
from DnnModel.model_Initialization import initializing_model

app = Flask(__name__)
classifier = initializing_model()
SAVED_LOCATION = join(dirname(realpath(__file__)), 'static') + "\Results\\"
SAVED_IMAGE = "Results/"


@app.route('/')
def web():
    remove_dir_files()
    return render_template('index.html')


def remove_dir_files():
    re_path = join(dirname(realpath(__file__)), 'static') + "\Results\\"
    up_path = join(dirname(realpath(__file__)), 'static') + "\\Uploads\\"
    re = os.listdir(re_path)
    up = os.listdir(up_path)
    if re:
        for i in re:
            os.remove(re_path + i)
    if up:
        for i in up:
            os.remove(up_path + i)


@app.route('/process', methods=['GET', 'POST'])
def upload_aerial_image():
    global aerial_image_path
    aerial_image, aerial_image_path, image_name = upload_file()
    # predicting
    t1 = time()
    predict = RemoteSensing(aerial_image_path, image_name, classifier, SAVED_LOCATION)
    predict.aerial_predicting()
    print(aerial_image_path)
    print(SAVED_LOCATION + image_name)
    t2 = time()
    print("Total time is: ", (t2 - t1) / 60, " minute")
    return render_template('process.html', aerial_image=aerial_image, result_image=SAVED_IMAGE + image_name)


if __name__ == '__main__':
    app.run()
