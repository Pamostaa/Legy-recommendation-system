import os
os.environ["TRANSFORMERS_NO_TF"] = "1"  # Force PyTorch-only

import pandas as pd
import torch
from datasets import Dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split

# Check device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# STEP 1: Load your labeled dataset
df = pd.read_csv("C:/Users/21655/OneDrive - Ministere de l'Enseignement Superieur et de la Recherche Scientifique/Desktop/Legy Data/Sentiment analysis/labeled_reviews.csv")

# Train-validation split
df_train, df_val = train_test_split(df, test_size=0.1, random_state=42)

# STEP 2: Prepare the HuggingFace Datasets
dataset_train = Dataset.from_pandas(df_train)
dataset_val = Dataset.from_pandas(df_val)

# STEP 3: Load BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

# Tokenization function
def preprocess_function(examples):
    return tokenizer(examples['Review'], truncation=True, padding='max_length', max_length=128)

# Apply tokenizer
dataset_train = dataset_train.map(preprocess_function, batched=True)
dataset_val = dataset_val.map(preprocess_function, batched=True)

# Merge labels into a single 'labels' field with floats
def merge_labels(example):
    example['labels'] = [
        float(example['cares_about_food_quality']),
        float(example['cares_about_service_speed']),
        float(example['cares_about_price']),
        float(example['cares_about_cleanliness'])
    ]
    return example

dataset_train = dataset_train.map(merge_labels)
dataset_val = dataset_val.map(merge_labels)

# Set format for PyTorch
dataset_train.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
dataset_val.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

# STEP 4: Prepare model for multi-label classification
model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-cased', num_labels=4, problem_type="multi_label_classification")
model.to(device)

# Metrics
def compute_metrics(pred):
    logits, labels = pred
    predictions = torch.sigmoid(torch.tensor(logits)) > 0.5
    labels = torch.tensor(labels)
    acc = (predictions == labels).float().mean()
    return {"accuracy": acc.item()}

# STEP 5: Set Training Arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=4,  # Increased to 4 epochs
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
)

# STEP 6: Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset_train,
    eval_dataset=dataset_val,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# STEP 7: Start training
trainer.train()

# STEP 8: Save the model
model.save_pretrained("./bert_preference_model")
tokenizer.save_pretrained("./bert_preference_model")

print("\n✅ Training completed and model saved in ./bert_preference_model")
