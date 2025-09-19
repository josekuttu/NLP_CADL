# -------------------------------
# Install required libraries (uncomment if needed)
# -------------------------------
# pip install transformers datasets torch spacy
# python -m spacy download en_core_web_sm

# -------------------------------
# Imports
# -------------------------------
from transformers import BartTokenizer, BartForConditionalGeneration, Trainer, TrainingArguments
from datasets import load_dataset

# -------------------------------
# Step 0: Load Dataset
# -------------------------------
# Using CNN/DailyMail dataset as an example
dataset = load_dataset("cnn_dailymail", "3.0.0")
print("Sample data:", dataset['train'][0])

# -------------------------------
# Step 1: Load Pre-trained Tokenizer and Model
# -------------------------------
tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")

# -------------------------------
# Step 2: Tokenization Function
# -------------------------------
max_input_length = 1024
max_target_length = 128

def preprocess_function(examples):
    inputs = examples["article"]
    targets = examples["highlights"]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)

    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=max_target_length, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_datasets = dataset.map(preprocess_function, batched=True)

# -------------------------------
# Step 3: Fine-Tuning Setup
# -------------------------------
training_args = TrainingArguments(
    output_dir="./bart_summarizer",
    evaluation_strategy="steps",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    save_steps=500,
    eval_steps=500,
    logging_steps=200,
    save_total_limit=2,
    num_train_epochs=1,   # Increase for better results
    learning_rate=5e-5,
    weight_decay=0.01,
    fp16=True,
    push_to_hub=False
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"].select(range(1000)),  # small subset for testing
    eval_dataset=tokenized_datasets["validation"].select(range(200))
)

# -------------------------------
# Step 4: Train the Model
# -------------------------------
trainer.train()

# -------------------------------
# Step 5: Test the Model
# -------------------------------
text = """
Artificial intelligence (AI) is transforming industries across the globe. Companies are using AI 
for predictive analytics, natural language processing, and automation to improve efficiency and reduce costs.
"""

inputs = tokenizer([text], max_length=1024, return_tensors="pt", truncation=True)
summary_ids = model.generate(inputs["input_ids"], num_beams=4, max_length=60, min_length=20, early_stopping=True)
summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
print("\nGenerated Summary:\n", summary)
