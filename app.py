from flask import Flask, render_template, request, redirect, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array
import numpy as np
import tensorflow as tf
import warnings
import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Suppress warnings
warnings.filterwarnings("ignore")

# Initialize Flask app
app = Flask(__name__)

# Set up file directories
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the trained model
def load_trained_model():
    return tf.keras.models.load_model(r"C:\Users\AMRATANSHU GUPTA\Desktop\Foodify\Food-Recognition\foodRecognition.h5")

model = load_trained_model()

# Reverse mapping for class indices
class_indices =  {0: 'apple_pie',
 1: 'baby_back_ribs',
 2: 'baklava',
 3: 'beef_carpaccio',
 4: 'beef_tartare',
 5: 'beet_salad',
 6: 'beignets',
 7: 'bibimbap',
 8: 'bread_pudding',
 9: 'breakfast_burrito',
 10: 'bruschetta',
 11: 'caesar_salad',
 12: 'cannoli',
 13: 'caprese_salad',
 14: 'carrot_cake',
 15: 'ceviche',
 16: 'cheese_plate',
 17: 'cheesecake',
 18: 'chicken_curry',
 19: 'chicken_quesadilla',
 20: 'chicken_wings',
 21: 'chocolate_cake',
 22: 'chocolate_mousse',
 23: 'churros',
 24: 'clam_chowder',
 25: 'club_sandwich',
 26: 'crab_cakes',
 27: 'creme_brulee',
 28: 'croque_madame',
 29: 'cup_cakes',
 30: 'deviled_eggs',
 31: 'donuts',
 32: 'dumplings',
 33: 'edamame',
 34: 'eggs_benedict',
 35: 'escargots',
 36: 'falafel',
 37: 'filet_mignon',
 38: 'fish_and_chips',
 39: 'foie_gras',
 40: 'french_fries',
 41: 'french_onion_soup',
 42: 'french_toast',
 43: 'fried_calamari',
 44: 'fried_rice',
 45: 'frozen_yogurt',
 46: 'garlic_bread',
 47: 'gnocchi',
 48: 'greek_salad',
 49: 'grilled_cheese_sandwich',
 50: 'grilled_salmon',
 51: 'guacamole',
 52: 'gyoza',
 53: 'hamburger',
 54: 'hot_and_sour_soup',
 55: 'hot_dog',
 56: 'huevos_rancheros',
 57: 'hummus',
 58: 'ice_cream',
 59: 'lasagna',
 60: 'lobster_bisque',
 61: 'lobster_roll_sandwich',
 62: 'macaroni_and_cheese',
 63: 'macarons',
 64: 'miso_soup',
 65: 'mussels',
 66: 'nachos',
 67: 'omelette',
 68: 'onion_rings',
 69: 'oysters',
 70: 'pad_thai',
 71: 'paella',
 72: 'pancakes',
 73: 'panna_cotta',
 74: 'peking_duck',
 75: 'pho',
 76: 'pizza',
 77: 'pork_chop',
 78: 'poutine',
 79: 'prime_rib',
 80: 'pulled_pork_sandwich',
 81: 'ramen',
 82: 'ravioli',
 83: 'red_velvet_cake',
 84: 'risotto',
 85: 'samosa',
 86: 'sashimi',
 87: 'scallops',
 88: 'seaweed_salad',
 89: 'shrimp_and_grits',
 90: 'spaghetti_bolognese',
 91: 'spaghetti_carbonara',
 92: 'spring_rolls',
 93: 'steak',
 94: 'strawberry_shortcake',
 95: 'sushi',
 96: 'tacos',
 97: 'takoyaki',
 98: 'tiramisu',
 99: 'tuna_tartare',
 100: 'waffles'}
# Function to get a recipe using Gemini API
def get_recipe_from_gemini(predicted_class):
    my_api_key = os.getenv('GEMINI_API_KEY')
    genai.configure(api_key=my_api_key)

    model = genai.GenerativeModel('gemini-2.0-pro-exp')
    response = model.generate_content(f"Give me a step-by-step recipe for {predicted_class} and give proper steps.")

    return response.text if hasattr(response, 'text') else str(response)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    try:
        # Save the uploaded file
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        # Process the uploaded image
        image = load_img(file_path, target_size=(224, 224))
        image_array = img_to_array(image)
        preprocessed_image = tf.keras.applications.mobilenet_v2.preprocess_input(image_array)
        image_batch = np.expand_dims(preprocessed_image, axis=0)

        # Make predictions
        predicted_probabilities = model.predict(image_batch)
        predicted_class_index = np.argmax(predicted_probabilities, axis=1)[0]
        predicted_label = class_indices[predicted_class_index].capitalize()
        predicted_label = predicted_label.replace('_', " ")
        # Return the predicted label
        return jsonify({"predicted_label": predicted_label})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/recipe')
def recipe():
    predicted_label = request.args.get('label', default='', type=str)

    if not predicted_label:
        return "Error: No label provided", 400

    try:
        # Get the recipe using Gemini API
        recipe = get_recipe_from_gemini(predicted_label)
        recipe = recipe.replace("*", '')
        recipe = recipe.replace("#", '')
        return render_template('recipe.html', label=predicted_label, recipe=recipe)
    except Exception as e:
        return f"Error fetching recipe: {str(e)}", 500

if __name__ == '__main__':
    app.run(debug=True)
