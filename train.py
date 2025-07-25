# import torch
# from torch.utils.data import DataLoader, TensorDataset
# import torch.optim as optim
# from model import Pong
# import torch.nn as nn
# import torch.nn.functional as F

# observations = torch.load('pong_observations.pt')  # Image frames as tensors
# actions = torch.load('pong_actions.pt')            # Action labels as tensors
# # Combine the observations and actions into a dataset and DataLoader
# dataset = TensorDataset(observations, actions)
# dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# # Initialize the model, loss function, and optimizer
# model = Pong()
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)

# # Training loop
# epochs = 10
# for epoch in range(epochs):
#     for images, labels in dataloader:
#         # Forward pass
#         outputs = model(images)
#         loss = criterion(outputs, labels)
        
#         # Backward pass and optimization
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
    
#     print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")


# import torch
# from torch.utils.data import DataLoader, TensorDataset
# import torch.optim as optim
# from model import Pong
# import torch.nn as nn
# import torch.nn.functional as F
# # Load the data
# observations = torch.load('pong_observations.pt')  # Shape [15261, 4, 84, 84]
# actions = torch.load('pong_actions.pt')  # Shape [15261]

# # Create a DataLoader for batching
# print("Observations shape:", observations.shape)  # Should be [15261, 4, 84, 84] or [15261, 84, 84, 4]
# batch_size = 64
# dataset = TensorDataset(observations, actions)
# train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
# print(train_loader)

# # Instantiate the model, define the loss function and optimizer
# device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
# model = Pong().to(device)
# # model = Pong()
# # model = model.to('cuda' if torch.cuda.is_available() else 'cpu')
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=1e-3)

# # Training loop
# epochs = 10
# # device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')


# for images, labels in train_loader:
#     print(images.shape)
#     images = images.permute(0, 3, 1, 2)
#     print(images.shape)  # Should print [batch_size, 4, 84, 84]
#     break
# torch.autograd.set_detect_anomaly(True)

# # model.train()
# model = model.to(device)
# # Test a forward pass
# with torch.no_grad():
#     for images, labels in train_loader:
#         images, labels = images.to(device), labels.to(device)
#         outputs = model(images)
#         print("Output shape:", outputs.shape)  # Should be [batch_size, num_actions]
#         break  # Only test with one batch for now

# for epoch in range(epochs):
#     running_loss = 0.0
#     for i, (images, labels) in enumerate(train_loader):
#         images = images.permute(0, 3, 1, 2)
#         print(f"Batch {i+1}: images shape after permute: {images.shape}")

#         images, labels = images.to(device), labels.to(device)

#         # Zero the parameter gradients
#         optimizer.zero_grad()

#         # Forward pass
#         outputs = model(images)
#         loss = criterion(outputs, labels)

#         # Backward pass and optimize
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         running_loss += loss.item()
#         if (i + 1) % 100 == 0:
#             print(f"Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(train_loader)}], Loss: {running_loss / 100:.4f}")
#             running_loss = 0.0

# print("Training completed.")

# # Save the trained model
# torch.save(model.state_dict(), 'pong_model.pth')

import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from model import Pong 
import torch.nn as nn

# Load the data
observations = torch.load('pong_observations.pt')  # Shape: [15261, 4, 84, 84]
actions = torch.load('pong_actions.pt')  # Shape: [15261]

# Create a DataLoader for batching
print("Observations shape:", observations.shape)
batch_size = 64
dataset = TensorDataset(observations, actions)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Instantiate the model, define the loss function, and optimizer
device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
model = Pong().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Training loop
epochs = 10

# Ensure the data is in the right format and perform a forward pass for verification
for images, labels in train_loader:
    print("Original images shape:", images.shape)
    images = images.permute(0, 3, 1, 2)  # Ensure images are [batch_size, 4, 84, 84]
    print("Permuted images shape:", images.shape)
    break

# Set anomaly detection
torch.autograd.set_detect_anomaly(True)

# Model warmup with one batch (for shape verification)
model.eval()
with torch.no_grad():
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        images = images.permute(0, 3, 1, 2)  # Change to [batch_size, 4, 84, 84]
        outputs = model(images)
        print("Output shape:", outputs.shape)  # Expected: [batch_size, num_actions]
        break  # Only test with one batch for verification

# Training
model.train()
for epoch in range(epochs):
    running_loss = 0.0
    for i, (images, labels) in enumerate(train_loader):
        # Prepare inputs
        images = images.permute(0, 3, 1, 2).to(device)
        labels = labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if (i + 1) % 100 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(train_loader)}], Loss: {running_loss / 100:.4f}")
            running_loss = 0.0

print("Training completed.")

# Save the trained model
torch.save(model.state_dict(), 'pong_model.pth')
