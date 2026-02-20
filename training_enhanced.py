
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback
import torch
from datasets import Dataset
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import warnings
import re
from sklearn.utils.class_weight import compute_class_weight

warnings.filterwarnings('ignore')

print("📁 Loading medical dataset...")
df = pd.read_csv('medical_training_data.csv')

# FIX: Check and correct column order if needed
if 'disease' not in df.columns or 'symptoms' not in df.columns:
    print("🔄 Adjusting column names...")
    if len(df.columns) == 2:
        df.columns = ['disease', 'symptoms']  # Assuming disease first, symptoms second
    else:
        print("❌ CSV must have exactly 2 columns: disease and symptoms")
        exit()

print(f"📊 Dataset Overview:")
print(f"Total samples: {len(df)}")
print(f"Number of unique diseases: {df['disease'].nunique()}")
print(f"\nDisease Distribution:")
print(df['disease'].value_counts())

# ===== SIMPLIFIED DATA AUGMENTATION =====
def augment_medical_data(df, augmentation_factor=2):
    """Simplified data augmentation"""
    print("🔄 Augmenting medical data...")
    augmented_rows = []
    
    medical_synonyms = {
        'pain': ['discomfort', 'ache'],
        'fever': ['high temperature'],
        'cough': ['coughing'],
        'headache': ['migraine'],
        'nausea': ['sickness'],
        'vomiting': ['throwing up'],
        'fatigue': ['tiredness'],
        'dizziness': ['vertigo'],
        'rash': ['skin eruption'],
        'swelling': ['edema']
    }
    
    for _, row in df.iterrows():
        augmented_rows.append(row)  # Keep original
        
        symptoms = str(row['symptoms']).lower()
        
        # Create augmented versions
        for i in range(augmentation_factor):
            new_symptoms = symptoms
            
            # Simple synonym replacement
            for word, synonyms in medical_synonyms.items():
                if word in new_symptoms and np.random.random() > 0.7:
                    new_symptoms = new_symptoms.replace(word, np.random.choice(synonyms))
            
            augmented_row = row.copy()
            augmented_row['symptoms'] = new_symptoms
            augmented_rows.append(augmented_row)
    
    augmented_df = pd.DataFrame(augmented_rows)
    print(f"✅ Data augmented: {len(df)} → {len(augmented_df)} samples")
    return augmented_df

# Apply data augmentation
df = augment_medical_data(df, augmentation_factor=2)

# ===== ENHANCED PREPROCESSING =====
def clean_medical_text(text):
    """Medical text cleaning"""
    text = str(text).lower()
    
    # Medical abbreviation expansion
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

# Apply cleaning
df['symptoms'] = df['symptoms'].apply(clean_medical_text)

# Convert labels to numerical format
label_names = df['disease'].unique()
label_to_id = {label: idx for idx, label in enumerate(label_names)}
id_to_label = {idx: label for label, idx in label_to_id.items()}

df['label'] = df['disease'].map(label_to_id)

print(f"\n🏷️ Label Classes: {list(label_names)}")

# Split data
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df['symptoms'].values,
    df['label'].values,
    test_size=0.2,
    random_state=42,
    stratify=df['label']
)

print(f"\n📈 Data Split:")
print(f"Training samples: {len(train_texts)}")
print(f"Validation samples: {len(val_texts)}")

# ===== TOKENIZATION =====
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

def tokenize_function(examples):
    return tokenizer(
        examples['text'], 
        padding="max_length", 
        truncation=True, 
        max_length=256
    )

print("🔤 Tokenizing datasets...")
train_dataset = Dataset.from_dict({'text': train_texts, 'labels': train_labels})
val_dataset = Dataset.from_dict({'text': val_texts, 'labels': val_labels})

train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)

# ===== MODEL INITIALIZATION =====
model = DistilBertForSequenceClassification.from_pretrained(
    'distilbert-base-uncased',
    num_labels=len(label_names),
    id2label=id_to_label,
    label2id=label_to_id
)

print(f"\n🤖 Model initialized with {len(label_names)} classes")

# ===== FIXED TRAINING ARGUMENTS =====
training_args = TrainingArguments(
    output_dir='./medical_ai_model_enhanced',
    num_train_epochs=8,  # Reduced for stability
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=200,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=50,
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=100,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    learning_rate=3e-5,
    fp16=False,  # Disabled for stability
    dataloader_pin_memory=False,
)

# ===== SIMPLIFIED METRICS =====
def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=1)
    
    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='weighted')
    
    return {
        'accuracy': accuracy,
        'f1_score': f1
    }

# ===== SIMPLIFIED TRAINER (NO CUSTOM CLASS) =====
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)

print("🚀 Starting enhanced training...")
trainer.train()

print("💾 Saving enhanced model...")
trainer.save_model('./medical_ai_model_enhanced')
tokenizer.save_pretrained('./medical_ai_model_enhanced')

# Final evaluation
eval_results = trainer.evaluate()
print(f"📊 Final validation accuracy: {eval_results['eval_accuracy']:.3f}")
print(f"📊 Final validation F1 score: {eval_results['eval_f1_score']:.3f}")

print("✅ Enhanced training completed! Model saved to './medical_ai_model_enhanced'")