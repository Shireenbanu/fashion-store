from flask import Flask, render_template, request, jsonify
import os
import random
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from ecommerce_classifier import train_model, predict_image
from visual_search import VisualSearch
import base64
from flask import send_from_directory

app = Flask(__name__)

# Initialize both models at startup
def initialize_models():
    # Initialize classification model
    base_dir = 'ECOMMERCE_PRODUCT_IMAGES'
    train_dir = os.path.join(base_dir, 'train')
    val_dir = os.path.join(base_dir, 'val')
    model_path = 'ecommerce_classifier.h5'
    
    if not os.path.exists(model_path):
        print("Training classification model...")
        model = train_model(train_dir=train_dir, val_dir=val_dir, epochs=20, batch_size=32)
        model.save(model_path)
    else:
        print("Loading classification model...")
        model = load_model(model_path)
    
    # Initialize visual search
    print("Initializing visual search...")
    visual_searcher = VisualSearch()
    visual_searcher.build_indexes()
    
    return model, visual_searcher

# Global variables for models
CLASSIFICATION_MODEL, VISUAL_SEARCHER = initialize_models()

# Define the base directory and category folders
BASE_FOLDER = 'static/products'
CATEGORY_FOLDERS = [
    'BABY_PRODUCTS', 'BEAUTY_HEALTH', 'CLOTHING_ACCESSORIES_JEWELLERY', 
    'ELECTRONICS', 'GROCERY', 'HOBBY_ARTS_STATIONERY', 
    'HOME_KITCHEN_TOOLS', 'PET_SUPPLIES', 'SPORTS_OUTDOOR'
]

def get_random_image():
    """Get a random image from a random category folder"""
    try:
        category = random.choice(CATEGORY_FOLDERS)
        category_path = os.path.join(BASE_FOLDER, category)
        
        valid_extensions = ('.png', '.jpg', '.jpeg', '.gif')
        images = [f for f in os.listdir(category_path) 
                 if f.lower().endswith(valid_extensions)]
        
        if images:
            chosen_image = random.choice(images)
            return {
                'status': 'success',
                'image_url': f'/static/products/{category}/{chosen_image}',
                'category': category
            }
        return {
            'status': 'error',
            'message': f'No images found in category {category}'
        }
    except Exception as e:
        return {
            'status': 'error',
            'message': str(e)
        }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_random_image')
def random_image():
    return jsonify(get_random_image())

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        if 'image' not in request.files:
            return jsonify({'status': 'error', 'message': 'No image provided'})
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'status': 'error', 'message': 'No selected file'})
        
        if file:
            # Save uploaded file
            filename = secure_filename(file.filename)
            temp_folder = 'static/temp'
            os.makedirs(temp_folder, exist_ok=True)
            filepath = os.path.join(temp_folder, filename)
            file.save(filepath)
            
            # Make classification prediction
            predicted_class, probabilities = predict_image(CLASSIFICATION_MODEL, filepath)
            
            # Get all predictions with probabilities, sorted by confidence
            all_predictions = sorted(
                probabilities.items(), 
                key=lambda x: x[1], 
                reverse=True
            )
            
            # Find similar images using visual search
            try:
                similar_images, distances = VISUAL_SEARCHER.search(
                    filepath, 
                    predicted_class, 
                    num_results=4
                )
                
                # Process similar images with their distances
                similar_items = []
                for img_path, distance in zip(similar_images, distances):
                    path_parts = img_path.split('ECOMMERCE_PRODUCT_IMAGES/')
                    if len(path_parts) > 1:
                        url = f'/ECOMMERCE_PRODUCT_IMAGES/{path_parts[1]}'
                        similar_items.append({
                            'url': url,
                            'distance': float(distance),
                            'filename': os.path.basename(img_path)
                        })
                    
            except Exception as e:
                print(f"Visual search error: {e}")
                similar_items = []
            
            # Prepare detailed results
            results = {
                'classification': {
                    'predicted_class': predicted_class,
                    'confidence': float(probabilities[predicted_class]),
                    'all_probabilities': [
                        {
                            'category': cat,
                            'probability': float(prob),
                            'percentage': f"{float(prob) * 100:.2f}%"
                        }
                        for cat, prob in all_predictions
                    ]
                },
                'similar_images': similar_items
            }
            
            # Clean up temporary file
            os.remove(filepath)
            
            return jsonify({
                'status': 'success',
                'results': results
            })
            
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        })
		
@app.route('/ECOMMERCE_PRODUCT_IMAGES/<path:filename>')
def serve_image(filename):
    return send_from_directory('ECOMMERCE_PRODUCT_IMAGES', filename)


if __name__ == '__main__':
    app.run(debug=True)