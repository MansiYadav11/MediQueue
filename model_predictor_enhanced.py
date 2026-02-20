# model_predictor_enhanced.py
import torch
import numpy as np
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import re
from typing import Dict, Any

class MedicalAIPredictor:
    def __init__(self, model_path='./medical_ai_model_enhanced'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🚀 Loading enhanced medical AI model from {model_path}...")
        
        try:
            self.tokenizer = DistilBertTokenizer.from_pretrained(model_path)
            self.model = DistilBertForSequenceClassification.from_pretrained(model_path)
            self.model.to(self.device)
            self.model.eval()
            
            # Your exact disease labels from training
            self.disease_labels = [
                'Psoriasis', 'Varicose Veins', 'Asthma', 'Chronic Kidney Disease', 
                'Migraine', 'Gastritis', 'Anemia', 'Osteoarthritis', 'Chickenpox', 
                'Diabetes', 'Hypertension', 'Flu', 'COVID-19', 'Tuberculosis', 
                'Allergy', 'Depression', 'Heart Attack', 'Stroke', 'Kidney Stones', 
                'General Physician'
            ]
            
            # Emergency conditions
            self.emergency_conditions = ['Heart Attack', 'Stroke', 'COVID-19']
            
            print("✅ Enhanced AI model loaded successfully!")
            
        except Exception as e:
            print(f"❌ Error loading enhanced model: {e}")
            raise

    def clean_medical_text(self, text):
        """Enhanced medical text preprocessing"""
        text = str(text).lower().strip()
        
        # Medical abbreviation expansion (same as your training)
        medical_abbr = {
            'hr': 'heart rate', 'bp': 'blood pressure', 'temp': 'temperature',
            'c/o': 'complains of', 'sob': 'shortness of breath',
            'cp': 'chest pain', 'ha': 'headache', 'n/v': 'nausea vomiting'
        }
        
        for abbr, full in medical_abbr.items():
            text = text.replace(abbr, full)
        
        # Clean text
        text = re.sub(r'[^\w\s]', ' ', text)
        text = ' '.join(text.split())
        
        return text

    def predict(self, symptoms_text, top_k=3):
        """Get top-K predictions with confidence scores"""
        try:
            processed_text = self.clean_medical_text(symptoms_text)
            
            # Tokenize
            inputs = self.tokenizer(
                processed_text,
                truncation=True,
                padding=True,
                max_length=256,
                return_tensors='pt'
            )
            
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            # Get top-K predictions
            probs, indices = torch.topk(predictions, top_k)
            
            results = []
            for i in range(top_k):
                disease = self.disease_labels[indices[0][i].item()]
                confidence = probs[0][i].item()
                is_emergency = disease in self.emergency_conditions
                
                results.append({
                    'condition': disease,
                    'confidence': confidence,
                    'emergency': is_emergency
                })
            
            return results
            
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            return []

    def ai_recommend(self, symptoms: str) -> Dict[str, Any]:
        try:
            print(f"🔍 AI analyzing symptoms: '{symptoms}'")

            predictions = self.predict(symptoms, top_k=3)
            
            if not predictions:
                return {
                    'success': False,
                    'condition': 'General Physician',
                    'confidence': 0.0,
                    'message': 'AI model prediction failed'
                }
            
            top_prediction = predictions[0]
            
            # Use confidence threshold (adjustable)
            confidence_threshold = 0.5
            
            if top_prediction['confidence'] >= confidence_threshold:
                print(f"✅ AI Success: {top_prediction['condition']} (Confidence: {top_prediction['confidence']:.2f})")
                
                return {
                    'success': True,
                    'condition': top_prediction['condition'],
                    'confidence': round(top_prediction['confidence'], 2),
                    'emergency': top_prediction['emergency'],
                    'alternative_suggestions': predictions[1:],  # Other top predictions
                    'message': 'High confidence AI recommendation'
                }
            else:
                print(f"❌ AI Low confidence: {top_prediction['condition']} (Confidence: {top_prediction['confidence']:.2f})")
                
                return {
                    'success': False,
                    'condition': top_prediction['condition'],
                    'confidence': round(top_prediction['confidence'], 2),
                    'message': 'Low confidence in AI recommendation',
                    'suggestions': predictions  # Return all suggestions for fallback
                }
                
        except Exception as e:
            print(f"❌ AI Model Error: {str(e)}")
            return {
                'success': False,
                'condition': 'General Physician',
                'confidence': 0.0,
                'message': f'AI processing error: {str(e)}'
            }

# Global instance (same as your original structure)
enhanced_predictor = MedicalAIPredictor()

def ai_recommend(symptoms: str) -> Dict[str, Any]:
    """
    Main interface function - EXACT same as your original
    """
    return enhanced_predictor.ai_recommend(symptoms)

# Test function (same as your original)
def test_ai_model():
    """Test the enhanced AI model with various symptoms"""
    test_cases = [
        "I have sharp chest pain and sweating",
        "My skin has red patches and it's itching", 
        "Headache and dizziness for 2 days",
        "Stomach pain and nausea after eating",
        "Cough and breathing difficulty",
        "Joint pain and swelling in knees",
    ]
    
    print("🧪 Testing ENHANCED AI Model...")
    for symptoms in test_cases:
        result = ai_recommend(symptoms)
        print(f"\nSymptoms: '{symptoms}'")
        print(f"AI Recommendation: {result['condition']} (Confidence: {result.get('confidence', 0)})")
        print(f"Success: {result['success']}")
        if result.get('alternative_suggestions'):
            print(f"Alternatives: {[s['condition'] for s in result['alternative_suggestions']]}")

if __name__ == "__main__":
    test_ai_model()