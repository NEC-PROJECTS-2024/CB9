from flask import Flask, render_template, request
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import os
from werkzeug.utils import secure_filename

app = Flask(__name__, template_folder='./templates')

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model1 = load_model("densenet_model_epoch_v902.h5")
model2 = load_model("inception_model_epoch_06.h5")
model3 = load_model("xception_model_epoch_v102.h5")

classnames_path = 'classnames.txt'

def read_class_names(file_path):
    with open(file_path, 'r') as file:
        return [line.strip() for line in file]

classes = read_class_names(classnames_path)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_label(img_path, threshold=0.1):
    test_image = image.load_img(img_path, target_size=(224, 224))
    test_image = image.img_to_array(test_image) / 255.0
    test_image = np.expand_dims(test_image, axis=0)
    predict_x = (model1.predict(test_image) + model2.predict(test_image) + model3.predict(test_image)) / 3
    max_probability = np.max(predict_x)
    if max_probability < threshold:
        return "This image does not contain a dog."
    classes_x = np.argmax(predict_x, axis=1)
    return classes[classes_x[0]]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/aboutus')
def aboutus():
    return render_template('aboutus.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('predict.html', error='No file part')

        file = request.files['file']

        if file.filename == '':
            return render_template('predict.html', error='No selected file')

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            predicted_class_name = predict_label(file_path, threshold=0.5)

            return render_template('result.html', predicted_breed=predicted_class_name)
    return render_template('predict.html', error=None)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
