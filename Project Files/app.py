from flask import Flask, render_template, request
import os
from werkzeug.utils import secure_filename
from keras.models import load_model
from keras.preprocessing import image
import numpy as np

# Create Flask app
app = Flask(__name__)

# Set upload folder path
UPLOAD_FOLDER = os.path.join('static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load model once
model = load_model("healthy_vs_rotten.h5")

# Class labels
class_labels = [
    "Apple_Healthy", "Apple_Rotten",
    "Banana_Healthy", "Banana_Rotten",
    "Bellpepper_Healthy", "Bellpepper_Rotten",
    "Carrot_Healthy", "Carrot_Rotten",
    "Cucumber_Healthy", "Cucumber_Rotten",
    "Grape_Healthy", "Grape_Rotten",
    "Guava_Healthy", "Guava_Rotten",
    "Jujube_Healthy", "Jujube_Rotten",
    "Mango_Healthy", "Mango_Rotten",
    "Orange_Healthy", "Orange_Rotten",
    "Pomegranate_Healthy", "Pomegranate_Rotten",
    "Potato_Healthy", "Potato_Rotten",
    "Strawberry_Healthy", "Strawberry_Rotten",
    "Tomato_Healthy", "Tomato_Rotten"
]

# Home route
@app.route('/')
def index():
    return render_template('index.html')

# Predict route (GET and POST supported)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        if 'file' not in request.files:
            return "⚠️ No file part"

        file = request.files['file']
        if file.filename == '':
            return "⚠️ No selected file"

        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # Preprocess image
            img = image.load_img(file_path, target_size=(224, 224))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0) / 255.0

            # Predict
            prediction = model.predict(img_array)
            predicted_index = int(np.argmax(prediction))
            predicted_class = class_labels[predicted_index]

            return render_template(
                'predict.html',
                prediction=predicted_class,
                index=predicted_index
            )

    return render_template('predict.html')


# About page
@app.route('/about')
def about():
    return render_template('about.html')

# Portfolio details route
@app.route('/portfolio-details')
def portfolio_details():
    return render_template('portfolio-details.html')

# Run app
if __name__ == '__main__':
    app.run(debug=True)
