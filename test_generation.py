import torch
from src.diffusion.dima import DiMAModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = DiMAModel(config_path="src/configs", device=device)

model.load_pretrained()

sequences = model.generate_samples(num_texts=10)

print("\nGenerated sequences:")
for i, seq in enumerate(sequences, 1):
    print(f"{i}: {seq}")
