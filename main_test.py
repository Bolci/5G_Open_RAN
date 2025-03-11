import torch

tensor = torch.zeros(3, 3, dtype=torch.float32)  # Make sure to use a consistent dtype
torch.save(tensor, 'tensor.pth')

loaded_tensor = torch.load('tensor.pth')
print(loaded_tensor)