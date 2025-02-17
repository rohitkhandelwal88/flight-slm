import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import Dataset

# Load Phi-2 tokenizer and model
model_name = "microsoft/phi-2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
# Load Phi-2 model on the correct device
device = "cpu"
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)


tokenizer.pad_token = tokenizer.eos_token

# Load flight schedule data
df = pd.read_csv("data/flight_schedule.csv")

# Format data into text prompts
df['formatted_text'] = df.apply(lambda row: f"Flight ID: {row['Flight ID']} | Departure: {row['Departure Airport']} | "
                                            f"Arrival: {row['Arrival Airport']} | Departure Time: {row['Departure Time']} | "
                                            f"Arrival Time: {row['Arrival Time']}", axis=1)

# Convert DataFrame to Hugging Face Dataset
dataset = Dataset.from_pandas(df[['formatted_text']])

# Tokenize function
def tokenize_function(examples):
    return tokenizer(examples["formatted_text"], padding="max_length", truncation=True, max_length=128)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Training arguments
# training_args = TrainingArguments(
#     output_dir="model",
#     #evaluation_strategy="no",
#     learning_rate=2e-5,
#     per_device_train_batch_size=1,
#     num_train_epochs=3,
#     weight_decay=0.01,
#     save_strategy="epoch",
#     #push_to_hub=False
# )

training_args = TrainingArguments(
    output_dir="phi2_flight_model",
    per_device_train_batch_size=1,  # Small batch size to prevent OOM
    gradient_accumulation_steps=4,  # Accumulate gradients to simulate larger batch size
    learning_rate=2e-5,
    num_train_epochs=3,
    weight_decay=0.01,
    fp16=False,  # Mac MPS does not support fp16 well
    bf16=torch.backends.mps.is_available(),  # Use bf16 if MPS is available
    save_strategy="epoch",
    logging_dir="/mnt/data/logs",
    logging_steps=10,
    report_to="none"
)

# Data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
    data_collator=data_collator,
    tokenizer=tokenizer
)

# Train model
trainer.train()

trainer.save_model("model/phi2_flight_model")
tokenizer.save_pretrained("model/phi2_flight_model")
