import tensorflow as tf 
import json 
import os 
from PIL import Image
import matplotlib.pyplot as plt 
import matplotlib.patches as patches
from functools import lru_cache
from typing import Dict, Optional
from pathlib import Path
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
SAVED_MODELS_DIR = os.path.join(BASE_DIR, 'saved_models', 'skin_detection_ensemble')


# def load_ensemble_savedmodel(model_path):
#     """
#     Load ensemble model dari SavedModel
#     """
#     print(f"📂 Loading ensemble model from: {model_path}")
    
#     # Definisikan metadata_path di awal fungsi agar tersedia di semua scope
#     metadata_path = os.path.join(model_path, 'metadata.json')
#     metadata = None
    
#     # Coba load metadata terlebih dahulu
#     try:
#         with open(metadata_path, 'r') as f:
#             metadata = json.load(f)
#     except Exception as e:
#         print(f"Error loading metadata: {str(e)}")
#         return None, None
    
#     try:
#         # Coba dengan opsi kompatibilitas
#         options = tf.saved_model.LoadOptions(
#             experimental_io_device='/job:localhost'
#         )
#         loaded_model = tf.saved_model.load(model_path, options=options)
        
#         print("✅ Model loaded successfully!")
#         print(f"📊 Classes: {len(metadata['classes'])}")
#         print(f"🏗️ Created: {metadata['created_at']}")
        
#         return loaded_model, metadata
#     except Exception as e:
#         print(f"Error loading model: {str(e)}")
#         # Coba dengan cara alternatif
#         try:
#             print("Trying alternative loading method...")
#             # Gunakan tf.keras.models.load_model sebagai alternatif
#             loaded_model = tf.keras.models.load_model(model_path)
            
#             print("✅ Model loaded successfully with keras compatibility mode!")
#             return loaded_model, metadata
#         except Exception as alt_error:
#             print(f"Alternative loading failed: {str(alt_error)}")
            
#             # Jika semua metode gagal, buat dummy model sebagai fallback terakhir
#             print("Creating dummy model as last resort...")
#             class DummyModel:
#                 def predict_with_confidence(self, img_array):
#                     # Return dummy prediction
#                     return {
#                         'predictions': tf.constant([[0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.0]]),
#                         'predicted_class_idx': tf.constant([0]),
#                         'confidence': tf.constant([0.2])
#                     }
            
#             # Metadata sudah diload di awal fungsi
#             if metadata:
#                 return DummyModel(), metadata
#             else:
#                 # Jika metadata tidak tersedia, buat dummy metadata
#                 dummy_metadata = {
#                     "classes": ['Acne', 'Blackheads', 'Dark-Spots', 'Dry-Skin', 'Englarged-Pores', 'Eyebags', 'Oily-Skin', 'Skin-Redness', 'Whiteheads', 'Wrinkles'],
#                     "input_shape": [224, 224, 3],
#                     "created_at": datetime.now().isoformat()
#                 }
#                 return DummyModel(), dummy_metadata


def load_ensemble_savedmodel(model_path):
    """
    Load ensemble model dari SavedModel
    """
    print(f"📂 Loading ensemble model from: {model_path}")
    
    # Load model
    loaded_model = tf.saved_model.load(model_path)
    
    # Load metadata
    metadata_path = os.path.join(model_path, 'metadata.json')
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    print("✅ Model loaded successfully!")
    print(f"📊 Classes: {len(metadata['classes'])}")
    print(f"🏗️ Created: {metadata['created_at']}")
    
    return loaded_model, metadata 



class ProductionEnsembleModel:
    """
    Production-ready wrapper untuk ensemble model
    """
    
    def __init__(self, model_path):
        self.model_path = model_path 
        self.model, self.metadata = load_ensemble_savedmodel(model_path)
        self.classes = self.metadata['classes']
        self.input_shape = self.metadata['input_shape']
        
    def preprocess_image(self, image_path):
        """
        Preprocess image untuk prediction
        """
        img = tf.keras.preprocessing.image.load_img(
            image_path,
            target_size=self.input_shape[:2]
        )
        img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
        img_array = tf.expand_dims(img_array, 0)
        return img_array 
    
    def predict(self, image_path):
        """
        Predict skin problem dari image path
        """
        # Preprocess 
        img_array = self.preprocess_image(image_path)
        
        # Predict
        result = self.model.predict_with_confidence(img_array)
        
        # Extract results
        predictions = result['predictions'].numpy()[0]
        predicted_class_idx = result['predicted_class_idx'].numpy()[0]
        confidence = result['confidence'].numpy()[0]
        
        # Get class name
        predicted_class = self.classes[predicted_class_idx]
        
        # Top 5 predictions
        top_5_idx = tf.nn.top_k(predictions, k=5).indices.numpy()
        top_5_predictions = [
            (self.classes[idx], float(predictions[idx]))
            for idx in top_5_idx
        ]
        
        return {
            'predicted_class': predicted_class,
            'confidence': float(confidence),
            'top_5_predictions': top_5_predictions,
            'all_predictions': predictions.tolist()
        }
        
    def batch_predict(self, image_paths):
        """
        Batch prediction untuk multiple images
        """
        results = []
        for img_path in image_paths:
            result = self.predict(img_path)
            results.append({
                'image_path': img_path,
                'result': result
            })
        return results 
    
    

# def test_saved_ensemble_model():
#     """
#     Test saved ensemble model
#     """
#     print("🧪 Testing Saved Ensemble Model...")
    
#     # Load production model
#     production_model = ProductionEnsembleModel('./saved_models/skin_detection_ensemble')
    
#     # Test dengan sample image
#     test_image = './user-testing1.jpg'
    
#     # Predict
#     result = production_model.predict(test_image)
    
#     print(f"🎯 Prediction: {result['predicted_class']}")
#     print(f"📊 Confidence: {result['confidence']:.2%}")
#     print(f"🏆 Top 5: {result['top_5_predictions'][:3]}")
    
#     return result


"""
Class Cache for ProductionEnsembleModel
"""
class ModelCache:
    _instance = None
    _model: Optional[ProductionEnsembleModel] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelCache, cls).__new__(cls)
        return cls._instance

    @property
    def model(self) -> ProductionEnsembleModel:
        if self._model is None:
            # Fix path handling
            self._model = ProductionEnsembleModel(SAVED_MODELS_DIR)
        return self._model
    
    def reload_model(self):
        """Force reload the model if needed"""
        self._model = ProductionEnsembleModel(SAVED_MODELS_DIR)
        return self._model
    

@lru_cache
def get_model_cache() -> ModelCache:
    return ModelCache()


def visualize_prediction(image_path, saved_path):
    """
    Demo menggunakan ProductionEnsembleModel dengan visualisasi custom
    """
    try:
        # Load production model
        production_model = get_model_cache().model
        
        # Periksa apakah model adalah instance dari DummyModel
        is_dummy = isinstance(production_model.model, type) and production_model.model.__name__ == 'DummyModel'
        
        # Get prediction
        result = production_model.predict(image_path)
        
        # Load gambar
        original_img = Image.open(image_path)
        original_width, original_height = original_img.size
        
        # Jika menggunakan dummy model, tambahkan peringatan
        if is_dummy:
            # Simpan gambar asli dengan peringatan
            plt.figure(figsize=(10, 8))
            plt.imshow(original_img)
            plt.title('WARNING: Using Fallback Model', fontsize=16, color='red')
            plt.text(10, 30, 'Model loading failed - using dummy predictions',
                     fontsize=12, color='red', bbox=dict(facecolor='white', alpha=0.8))
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(saved_path, dpi=300, bbox_inches='tight')
            plt.close()
            return saved_path 
        
        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(12, 6))
        
        # 1. Original image
        axes[0].imshow(original_img)
        axes[0].set_title('Original Image', fontsize=14, fontweight='bold')
        axes[0].axis('off')
        
        # 2. Prediction results
        axes[1].imshow(original_img)
        
        # Bounding box
        box_width = original_width * 0.7
        box_height = original_height * 0.7
        box_x = (original_width - box_width) / 2
        box_y = (original_height - box_height) / 2
        
        # Color based on confidence
        confidence = result['confidence']
        if confidence > 0.8:
            color = 'green'
            status = 'High Confidence'
        elif confidence > 0.6:
            color = 'orange'
            status = 'Medium Confidence'
        else:
            color = 'red'
            status = 'Low Confidence'
            
        # Add bounding box
        rect = patches.Rectangle(
            (box_x, box_y), box_width, box_height, 
            linewidth=4, edgecolor=color, facecolor='none'
        )
        axes[1].add_patch(rect)
        
        # Label
        main_label = f"{result['predicted_class']}\nConfidence: {confidence:.2%}\n{status}"
        axes[1].text(
            box_x, box_y - 20, main_label,
            fontsize=12, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.5", facecolor=color, alpha=0.8),
            color='white'
        )
        
        axes[1].set_title(f'Prediction: {result["predicted_class"]} ({confidence:.2%})',
                        fontsize=14, fontweight='bold')
        axes[1].axis('off')
        
        # 3. Detailed info
        axes[2].axis('off')
        
        info_text = " SAVED ENSEMBLE MODEL\n"
        info_text += "=" * 35 + "\n\n"
        info_text += " PREDICTION RESULTS:\n"
        info_text += f"   Class: {result['predicted_class']}\n"
        info_text += f"   Confidence: {confidence:.2%}\n"
        info_text += f"   Status: {status}\n\n"
        
        info_text += " TOP 5 PREDICTIONS:\n"
        for i, (cls, conf) in enumerate(result['top_5_predictions'][:5]):
            info_text += f"   {i+1}. {cls}: {conf:.2%}\n"
        
        # info_text += "\n MODEL INFO:\n"
        # info_text += "   Type: SavedModel Ensemble\n"
        # info_text += f"   Image Size: {original_width}x{original_height}\n"
        # info_text += f"   Classes: {len(production_model.classes)}\n"
        
        axes[2].text(
            0.05, 0.95, info_text,
            transform=axes[2].transAxes,
            fontsize=10,
            verticalalignment='top',
            fontfamily='monospace',
            bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgray', alpha=0.8)
        )
        
        axes[2].set_title('Detailed Results', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(saved_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return saved_path
    
    except Exception as e:
        print(f"visualize_prediction error: {str(e)}")
        try:
            # Simpan gambar asli sebagai fallback dengan pesan error
            original_img = Image.open(image_path)
            plt.figure(figsize=(10, 8))
            plt.imshow(original_img)
            plt.title("ERROR: Visualization Failed", fontsize=16, color='red')
            plt.text(10, 30, f"Error: {str(e)}", fontsize=12, color='red', 
                     bbox=dict(facecolor='white', alpha=0.8))
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(saved_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Fallback: Saved error image to {saved_path}")
            return saved_path
        except Exception as fallback_error:
            print(f"Fallback error: {str(fallback_error)}")
            # Jika semua gagal, coba simpan gambar asli tanpa modifikasi
            try:
                original_img = Image.open(image_path)
                original_img.save(saved_path)
                print(f"Last resort fallback: Saved original image to {saved_path}")
                return saved_path
            except:
                # Jika benar-benar gagal, kembalikan None dan biarkan endpoint menangani
                return None


def validate_model():
    """
    Validate model when starting the app
    """
    try:
        model_cache = get_model_cache()
        model = model_cache.model 
        print(f"Model loaded successfully with {len(model.classes)} classes")
        return True 
    except Exception as e:
        print(f"Model validation failed: {str(e)}")
        return False

# Test saved model

# if __name__ == '__main__':
#     visualize_prediction()