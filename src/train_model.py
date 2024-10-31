import pandas as pd
from datasets import Dataset
from transformers import DistilBertTokenizerFast
from transformers import DistilBertForSequenceClassification, Trainer, TrainingArguments
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import resample  # Import resample for oversampling
from transformers import pipeline
from transformers import DataCollatorWithPadding  # Import DataCollatorWithPadding

# Load the data
df = pd.read_csv('data/faq_data.csv')

# Map intents to numeric labels
intent_counts = df['Intent'].value_counts()
print("Intent counts before balancing:")
print(intent_counts)

# Oversample minority classes to balance the dataset
majority_class_size = intent_counts.max()

# List to hold the oversampled dataframes
oversampled_dfs = []

# For each intent, oversample to match the majority class size
for intent in intent_counts.index:
    df_intent = df[df['Intent'] == intent]
    if len(df_intent) < majority_class_size:
        # Oversample minority class
        df_intent_oversampled = resample(
            df_intent,
            replace=True,                 # Sample with replacement
            n_samples=majority_class_size, # Match majority class size
            random_state=42               # For reproducibility
        )
        oversampled_dfs.append(df_intent_oversampled)
    else:
        # Majority class remains as is
        oversampled_dfs.append(df_intent)

# Combine all oversampled dataframes
df_balanced = pd.concat(oversampled_dfs)

# Shuffle the dataset
df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

# Check intent counts after balancing
balanced_intent_counts = df_balanced['Intent'].value_counts()
print("\nIntent counts after balancing:")
print(balanced_intent_counts)

# Proceed with mapping intents to labels
intent_labels = {intent: idx for idx, intent in enumerate(df_balanced['Intent'].unique())}
label_intents = {idx: intent for intent, idx in intent_labels.items()}
df_balanced['Intent_Label'] = df_balanced['Intent'].map(intent_labels)

print("\nIntent to Label mapping:")
for intent, label in intent_labels.items():
    print(f"{intent}: {label}")

# Split the data into training and validation sets
train_df, val_df = train_test_split(
    df_balanced,
    test_size=0.1,
    stratify=df_balanced['Intent_Label'],
    random_state=42
)

# Create Hugging Face Datasets
train_dataset = Dataset.from_pandas(train_df.reset_index(drop=True))
val_dataset = Dataset.from_pandas(val_df.reset_index(drop=True))

# Initialize the tokenizer
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

# Tokenize the questions
def tokenize(batch):
    return tokenizer(batch['Question'], padding=True, truncation=True)

train_dataset = train_dataset.map(tokenize, batched=True)
val_dataset = val_dataset.map(tokenize, batched=True)

# Rename 'Intent_Label' to 'labels' as the model expects 'labels' key
train_dataset = train_dataset.rename_column('Intent_Label', 'labels')
val_dataset = val_dataset.rename_column('Intent_Label', 'labels')

# Set the format for PyTorch
train_dataset.set_format(
    type='torch',
    columns=['input_ids', 'attention_mask', 'labels']
)
val_dataset.set_format(
    type='torch',
    columns=['input_ids', 'attention_mask', 'labels']
)

# Load the model and specify the number of labels
model = DistilBertForSequenceClassification.from_pretrained(
    'distilbert-base-uncased',
    num_labels=len(intent_labels)
)

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',             # Output directory
    num_train_epochs=5,                 # Number of training epochs
    per_device_train_batch_size=8,      # Batch size per device during training
    per_device_eval_batch_size=8,       # Batch size per device during evaluation
    warmup_steps=50,                    # Warmup steps
    weight_decay=0.01,                  # Weight decay
    eval_strategy='epoch',              # Evaluate every epoch
    save_strategy='epoch',              # Save the model every epoch
    logging_dir='./logs',               # Directory for logs
    logging_steps=10,
    load_best_model_at_end=True,        # Load the best model at the end
    metric_for_best_model='accuracy',
    greater_is_better=True,
)

# Define metrics for evaluation
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='weighted'
    )
    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

# Initialize the Data Collator
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    data_collator=data_collator,  # Use the data collator
)

# Train the model
trainer.train()

# Evaluate the model on the validation set
eval_metrics = trainer.evaluate()

# Print the evaluation metrics
print("\nEvaluation Metrics:")
for key, value in eval_metrics.items():
    print(f"{key}: {value}")

# Save the trained model
model.save_pretrained('fine_tuned_model')
tokenizer.save_pretrained('fine_tuned_model')

print("\nModel training complete and saved as 'fine_tuned_model'.")

# Load the fine-tuned model and tokenizer
model = DistilBertForSequenceClassification.from_pretrained('fine_tuned_model')
tokenizer = DistilBertTokenizerFast.from_pretrained('fine_tuned_model')

# Create a pipeline for text classification
classifier = pipeline('text-classification', model=model, tokenizer=tokenizer)

# Reconstruct label mapping
label_intents = {v: k for k, v in intent_labels.items()}

# Test inputs
test_inputs = [
    "What unique features does Clifftop Cabin offer for families?",
    "Is there a scenic view from Wildside Cabin?",
    "Can you describe the layout of Rondebos Retreat?",
    "What kind of entertainment options are available in Hill Penthouse Plett?",
    "What distinguishes Ichibi Luxury Lodge from other accommodations?",
    "What is the check-in process for each property?",
    "Are there any hidden fees I should be aware of before booking?",
    "Can I get a detailed list of amenities for each rental property?",
    "What is the cancellation policy for my reservation?",
    "Are there any special promotions or discounts available currently?",
    "How do I modify my booking for Dassen Eiland Home?",
    "What documents do I need to provide when checking in?",
    "Can I book multiple properties for the same dates?",
    "What is the maximum number of guests allowed at Arrowood Apartment?",
    "How do I check the availability of Keurbooms River Apartment?",
    "What local attractions are within walking distance of Panorama Seaview Apart?",
    "Are there any recommended restaurants near the properties?",
    "What outdoor activities can I enjoy while staying at Robberg Ridge?",
    "Can you suggest some family-friendly activities in Plettenberg Bay?",
    "What are the best hiking trails nearby?",
    "How can I reach out for support during my stay?",
    "Is there a way to contact the property manager if I have an issue?",
    "What should I do if I forget the lockbox code?",
    "Are there any specific instructions for using the amenities?",
    "Can you guide me on how to leave feedback after my stay?",
    "What are the benefits of booking directly through Calm Waters Plett?",
    "How do I find the nearest grocery store from my accommodation?",
    "Are pets allowed at any of the properties?",
    "What is the minimum stay requirement for each property?",
    "Can I request special accommodations, such as extra beds or cribs?"
]

for input_text in test_inputs:
    prediction = classifier(input_text)[0]
    label = prediction['label']
    score = prediction['score']
    # Extract the label number
    label_num = int(label.split('_')[-1])
    intent = label_intents[label_num]
    print(f"Input: {input_text}")
    print(f"Predicted Intent: {intent}, Score: {score}")
    print("---")