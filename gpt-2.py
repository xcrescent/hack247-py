import torch
import torch.nn as nn
from transformers import GPT2Tokenizer, GPT2ForSequenceClassification, AdamW, get_linear_schedule_with_warmup

# Load pre-trained model and tokenizer
model_name = "gpt2"  # You can choose different models like "gpt2-medium", "gpt2-large", etc.
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Create a simple text classification dataset
texts = ["This is a positive example.", "This is a negative example."]
labels = [1, 0]

# Tokenize input texts and create PyTorch DataLoader
inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
labels = torch.tensor(labels)
dataset = torch.utils.data.TensorDataset(inputs["input_ids"], inputs["attention_mask"], labels)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True)


# Create a GPT-2 model for sequence classification
class GPT2ForSequenceClassificationCustom(nn.Module):
    def __init__(self, config):
        super(GPT2ForSequenceClassificationCustom, self).__init__()
        self.num_labels = config.num_labels
        self.gpt2 = GPT2ForSequenceClassification(config)

    def forward(self, input_ids, attention_mask=None, labels=None):
        return self.gpt2(input_ids, attention_mask=attention_mask, labels=labels)


# Load the GPT-2 model with a classification head
model = GPT2ForSequenceClassificationCustom.from_pretrained(model_name, num_labels=2)

# Set device (CPU or CUDA)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Fine-tuning setup
optimizer = AdamW(model.parameters(), lr=1e-5)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(dataloader) * 5)

# Fine-tuning loop
for epoch in range(5):  # Fine-tune for 5 epochs as an example
    model.train()
    for batch in dataloader:
        batch = tuple(t.to(device) for t in batch)
        inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[2]}
        loss = model(**inputs).loss
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

# Save the fine-tuned model
model.save_pretrained("fine_tuned_gpt2_classification")

import torch
from transformers import GPT2Tokenizer, GPT2ForSequenceClassification

# Load the fine-tuned GPT-2 model and tokenizer
model_path = "fine_tuned_gpt2_classification"  # Path to the saved model
model = GPT2ForSequenceClassification.from_pretrained(model_path)
tokenizer = GPT2Tokenizer.from_pretrained(model_path)

# Set device (CPU or CUDA)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Text classification example
text = "This is a positive statement."
input_ids = tokenizer.encode(text, return_tensors="pt").to(device)
with torch.no_grad():
    logits = model(input_ids).logits
predicted_class = torch.argmax(logits).item()

if predicted_class == 1:
    print("Predicted class: Positive")
else:
    print("Predicted class: Negative")


from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load the fine-tuned GPT-2 model and tokenizer for text generation
model_path = "fine_tuned_gpt2_generation"  # Path to the saved model
model = GPT2LMHeadModel.from_pretrained(model_path)
tokenizer = GPT2Tokenizer.from_pretrained(model_path)

# Set device (CPU or CUDA)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Text generation example
prompt = "Once upon a time,"
input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
with torch.no_grad():
    output = model.generate(input_ids, max_length=50, num_return_sequences=1, no_repeat_ngram_size=2, top_k=50, top_p=0.95)

generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print("Generated Text:", generated_text)
