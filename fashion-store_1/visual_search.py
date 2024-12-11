import torch
from torchvision import models, transforms
from PIL import Image
import os
from pathlib import Path
import numpy as np
import faiss
import pickle

class VisualSearch:
    def __init__(self, base_path="ECOMMERCE_PRODUCT_IMAGES"):
        self.base_path = base_path
        self.model = models.resnet50(pretrained=True)
        self.model.eval()
        self.model = torch.nn.Sequential(*(list(self.model.children())[:-1]))
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.index_dict = {}
        self.image_paths = {}
    
    def search(self, query_image_path, category, num_results=10):
        if category not in self.index_dict:
            print(f"Available categories: {list(self.index_dict.keys())}")
            raise ValueError(f"Category {category} not found")
            
        query_features = self.extract_features(query_image_path)
        query_features = query_features.reshape(1, -1).astype('float32')
        
        D, I = self.index_dict[category].search(query_features, num_results)
        similar_images = [self.image_paths[category][i] for i in I[0]]
        return similar_images, D[0]

    def load_indexes(self, cache_dir="cache"):
        if not os.path.exists(cache_dir):
            return False
            
        try:
            with open(f"{cache_dir}/image_paths.pkl", "rb") as f:
                self.image_paths = pickle.load(f)
            
            for category in self.image_paths.keys():
                index_path = f"{cache_dir}/{category}.index"
                if os.path.exists(index_path):
                    self.index_dict[category] = faiss.read_index(index_path)
            return True
        except Exception as e:
            print(f"Error loading cache: {e}")
            return False

    def save_indexes(self, cache_dir="cache"):
        """Save the computed indexes to cache"""
        os.makedirs(cache_dir, exist_ok=True)
        
        # Save image paths
        with open(f"{cache_dir}/image_paths.pkl", "wb") as f:
            pickle.dump(self.image_paths, f)
        
        # Save indexes
        for category, index in self.index_dict.items():
            index_path = f"{cache_dir}/{category}.index"
            faiss.write_index(index, index_path)

    def extract_features(self, image_path):
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        image = image.unsqueeze(0)
        
        with torch.no_grad():
            features = self.model(image)
        return features.numpy().reshape(-1)

    def build_indexes(self):
        if self.load_indexes():
            print("Loaded indexes from cache")
            return
            
        train_dir = Path(self.base_path) / 'train'
        categories = [d for d in train_dir.iterdir() if d.is_dir()]
        
        for category_path in categories:
            category_name = category_path.name
            print(f"\nProcessing {category_name}")
            
            features_list = []
            image_paths_list = []
            
            images = list(category_path.glob("*.jpeg"))
            print(f"Found {len(images)} images")
            
            for img_path in images:
                try:
                    features = self.extract_features(img_path)
                    features_list.append(features)
                    image_paths_list.append(str(img_path))
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
            
            if features_list:
                features_array = np.array(features_list).astype('float32')
                index = faiss.IndexFlatL2(features_array.shape[1])
                index.add(features_array)
                
                self.index_dict[category_name] = index
                self.image_paths[category_name] = image_paths_list
                print(f"Successfully indexed {len(features_list)} images")
        
        self.save_indexes()
