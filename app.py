from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import io
import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash
import logging
from tensorflow.keras.preprocessing import image
import os
import tempfile

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize SQLite database
def init_db():
    try:
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS users 
                     (email TEXT PRIMARY KEY, password TEXT)''')
        conn.commit()
        conn.close()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Database initialization failed: {str(e)}")

init_db()

# Load the trained MobileNet model
try:
    model = tf.keras.models.load_model('models/MobNet_classic.h5')
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model: {str(e)}")
    raise

# Define class labels for almond varieties
class_labels = ['Independence', 'Lauranne', 'Mazetto']

@app.route('/signup', methods=['POST'])
def signup():
    logger.info("Received signup request")
    try:
        data = request.get_json()
        email = data.get('email')
        password = data.get('password')

        # Validate email format
        if not email or '@' not in email or '.' not in email:
            logger.warning(f"Invalid email format: {email}")
            return jsonify({'error': 'Invalid email format'}), 400

        # Validate password length
        if not password or len(password) < 6:
            logger.warning("Password too short")
            return jsonify({'error': 'Password must be at least 6 characters'}), 400

        # Check if email already exists
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        c.execute("SELECT email FROM users WHERE email = ?", (email,))
        if c.fetchone():
            conn.close()
            logger.warning(f"Email already exists: {email}")
            return jsonify({'error': 'Email already exists'}), 400

        # Save new user
        hashed_password = generate_password_hash(password)
        c.execute("INSERT INTO users (email, password) VALUES (?, ?)", (email, hashed_password))
        conn.commit()
        conn.close()
        logger.info(f"User signed up: {email}")
        return jsonify({'message': 'Sign up successful'}), 201

    except Exception as e:
        logger.error(f"Error during signup: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/login', methods=['POST'])
def login():
    logger.info("Received login request")
    try:
        data = request.get_json()
        email = data.get('email')
        password = data.get('password')

        # Validate email format
        if not email or '@' not in email or '.' not in email:
            logger.warning(f"Invalid email format: {email}")
            return jsonify({'error': 'Invalid email format'}), 400

        # Check credentials
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        c.execute("SELECT password FROM users WHERE email = ?", (email,))
        result = c.fetchone()
        conn.close()

        if result and check_password_hash(result[0], password):
            logger.info(f"Login successful for: {email}")
            return jsonify({'message': 'Login successful'}), 200
        else:
            logger.warning(f"Invalid login attempt for: {email}")
            return jsonify({'error': 'Invalid email or password'}), 401

    except Exception as e:
        logger.error(f"Login error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    logger.info("Received predict request")
    try:
        if 'file' not in request.files:
            logger.warning("No file provided in predict request")
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['file']
        if file.filename == '':
            logger.warning("No file selected in predict request")
            return jsonify({'error': 'No file selected'}), 400

        # Save the uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
            file.save(temp_file.name)
            temp_file_path = temp_file.name
            logger.debug(f"Temporary file saved at: {temp_file_path}")

        # Load and preprocess the image
        img = image.load_img(temp_file_path, target_size=(224, 224), interpolation='lanczos')
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array.astype('float32') / 255.0  # Normalize

        # Make prediction
        predictions = model.predict(img_array)
        logger.debug(f"Predictions: {predictions[0]}")
        predicted_class_index = np.argmax(predictions[0])
        probability = float(predictions[0][predicted_class_index])
        predicted_class = class_labels[predicted_class_index]

        # Delete the temporary file after processing
        os.unlink(temp_file_path)
        logger.debug(f"Temporary file deleted: {temp_file_path}")

        logger.info(f"Prediction result: {predicted_class}, Probability: {probability}")
        return jsonify({
            'variety': predicted_class,
            'probability': probability
        })

    except Exception as e:
        logger.error(f"Predict error: {str(e)}")
        # Ensure temporary file is deleted if an error occurs
        if 'temp_file_path' in locals():
            try:
                os.unlink(temp_file_path)
                logger.debug(f"Cleaned up temporary file on error: {temp_file_path}")
            except Exception as cleanup_error:
                logger.error(f"Failed to clean up temporary file: {str(cleanup_error)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    logger.info("Starting Flask server")
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
