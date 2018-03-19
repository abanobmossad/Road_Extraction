from os.path import dirname, join, realpath
from flask import Flask, request, flash
from werkzeug.utils import secure_filename, redirect

UPLOAD_FOLDER = 'static\\Uploads\\'
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'tiff', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}
UPLOADS_PATH = join(dirname(realpath(__file__)), app.config['UPLOAD_FOLDER'])


def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):

            filename = secure_filename(file.filename)
            filename = filename.split(".")
            file.save(UPLOADS_PATH + filename[0]+'.png')
    return "Uploads/"+filename[0]+'.png',UPLOADS_PATH+filename[0]+'.png',filename[0]+'.png'



def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
