import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load fine-tuned Phi-2 model
model_path = "model/phi2_flight_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
# Load Phi-2 model on the correct device
device = "cpu"
model = AutoModelForCausalLM.from_pretrained(model_path).to(device)

tokenizer.pad_token = tokenizer.eos_token

def generate_response(source, destination, flight_type=None):
    if flight_type == "direct":
        prompt = f"Find direct flights from {source} to {destination}."
    elif flight_type == "one-stop":
        prompt = f"Find one-stop flights from {source} to {destination}."
    else:
        prompt = f"Find all flights (direct or one-stop) from {source} to {destination}."

    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=100)
    input_ids = inputs["input_ids"]
    input_ids = input_ids.to(device)
    attention_mask = inputs["attention_mask"]
    attention_mask = attention_mask.to(device)

    with torch.no_grad():
        output = model.generate(input_ids, attention_mask=attention_mask, max_length=200, pad_token_id=tokenizer.eos_token_id)

    return tokenizer.decode(output[0], skip_special_tokens=True)

# Interactive chatbot
print("Flight Schedule Assistant")
print("Type 'exit' to stop.")

while True:
    source = input("\nEnter Source Airport: ").strip().upper()
    if source.lower() == "exit":
        break
    
    destination = input("Enter Destination Airport: ").strip().upper()
    if destination.lower() == "exit":
        break

    flight_type = input("Enter Flight Type (direct/one-stop, press Enter for both): ").strip().lower()
    if flight_type not in ["direct", "one-stop", ""]:
        print("Invalid option. Showing both direct and one-stop flights.")
        flight_type = None

    print("\nGenerating response...\n")
    response = generate_response(source, destination, flight_type)
    print(response)
