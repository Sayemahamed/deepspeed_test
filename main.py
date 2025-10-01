# train_cpu.py
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import argparse
import deepspeed
import os

# 1. Define a Simple Toy Model
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(128, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

# 2. Create a Dummy Dataset
class RandomDataset(Dataset):
    def __init__(self, num_samples, input_size):
        self.num_samples = num_samples
        self.input_size = input_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        data = torch.randn(self.input_size)
        label = torch.randint(0, 10, (1,)).squeeze()
        return data, label

def main():
    # 3. Argument Parsing for DeepSpeed
    parser = argparse.ArgumentParser(description="DeepSpeed CPU-only Test")
    parser.add_argument('--local_rank', type=int, default=-1, help='local rank passed from distributed launcher')
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    # 4. Initialize DeepSpeed for CPU
    model = SimpleModel()
    
    # DeepSpeed will automatically detect there's no GPU and configure for CPU
    model_engine, optimizer, _, _ = deepspeed.initialize(
        args=args,
        model=model,
        model_parameters=model.parameters()
    )

    # Print a message to confirm which rank is on which machine
    # Using hostname is a good way to verify processes are on different laptops
    rank = model_engine.global_rank
    hostname = os.uname()[1]
    print(f"--> Rank {rank} is running on host: {hostname} on device: {model_engine.device}")

    # 5. Prepare Dataloader
    batch_size = model_engine.train_micro_batch_size_per_gpu()
    train_dataset = RandomDataset(num_samples=1000, input_size=128)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # 6. Training Loop
    criterion = nn.CrossEntropyLoss()
    num_epochs = 2

    for epoch in range(num_epochs):
        for step, (data, labels) in enumerate(train_loader):
            # No need for .to(device) - data stays on CPU
            
            # Forward pass
            outputs = model_engine(data)
            loss = criterion(outputs, labels)

            # Backward pass
            model_engine.backward(loss)

            # Optimizer step
            model_engine.step()

            # Print loss only from the master process (rank 0)
            if rank == 0 and step % 10 == 0:
                print(f"Epoch: {epoch+1}/{num_epochs}, Step: {step}, Loss: {loss.item():.4f}")
            
            if step >= 50:
                break

    if rank == 0:
        print("\n--- CPU-only distributed test finished successfully! ---")

if __name__ == '__main__':
    main()