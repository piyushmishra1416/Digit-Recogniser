from flask import Flask, render_template, request, jsonify
from PIL import Image
import io
import base64
import numpy as np
import pickle

app = Flask(__name__)

# Load the trained SVM model
with open("model.pkl", "rb") as f:
    svm_model = pickle.load(f)

# Function to convert base64 string to image
def base64_to_image(base64_string):
    base64_bytes = base64_string.encode('utf-8')
    image_data = base64.b64decode(base64_bytes)
    image = Image.open(io.BytesIO(image_data))
    return image

# Function to preprocess the image for prediction
def preprocess_image(image):
    # Convert image to grayscale
    image = image.convert('L')
    # Resize image to 28x28 (MNIST dataset size)
    image = image.resize((28, 28))
    # Convert image to numpy array
    image_array = np.array(image)
    # Flatten image array
    image_flattened = image_array.flatten()
    # Normalize image
    image_normalized = image_flattened / 255.0
    return image_normalized

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict_digit', methods=['POST'])
def predict_digit():
    # Get base64-encoded image data from the request
    image_data = request.form['image_data']
    # Convert base64 string to image
    img = base64_to_image(image_data)
    # Preprocess image
    img_processed = preprocess_image(img)
    # Reshape image for prediction
    img_reshaped = img_processed.reshape(1, -1)
    # Predict digit using the trained SVM model
    prediction = svm_model.predict(img_reshaped)[0]
    return jsonify({'prediction': str(prediction)})

if __name__ == '__main__':
    app.run(debug=True)
