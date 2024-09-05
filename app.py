from flask import Flask, render_template, request, jsonify
import tensorflow as tf
from PIL import Image
import io
import base64
import numpy as np

app = Flask(__name__)

# Load your trained model
model = tf.keras.models.load_model('path_to_your_model.h5')

def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    image_data = request.json['image']
    image_data = base64.b64decode(image_data.split(',')[1])
    image = Image.open(io.BytesIO(image_data))
    processed_image = preprocess_image(image)
    
    prediction = model.predict(processed_image)
    result = 'Defective' if prediction[0][0] > 0.5 else 'Non-Defective'
    
    return jsonify({'result': result})

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['file']
    image = Image.open(file.stream)
    processed_image = preprocess_image(image)
    
    prediction = model.predict(processed_image)
    result = 'Defective' if prediction[0][0] > 0.5 else 'Non-Defective'
    
    return jsonify({'result': result})

if __name__ == '__main__':
    app.run(debug=True)
