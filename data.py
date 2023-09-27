import torch
import numpy as np
import torch.nn as nn


# def generate_data(num_samples=1000, input_dim=10):
#     # Generate random one-hot encoded vectors
#     inputs = torch.eye(input_dim)
#     data = inputs[torch.randint(0, input_dim, (num_samples,))]

#     # If a stimulus appears on left, it's [data, zeros]; if on right, it's [zeros, data]
#     left_data = torch.cat((data, torch.zeros_like(data)), dim=1)
#     right_data = torch.cat((torch.zeros_like(data), data), dim=1)

#     return left_data, right_data

def generate_data(num_samples=1000, input_dim=10, embedding_dim=20):
    embedding = nn.Embedding(input_dim, embedding_dim)

    # Generate random indices for embeddings
    indices = torch.randint(0, input_dim, (num_samples,))
    data = embedding(indices)

    # Introducing random noise
    noise = torch.randn_like(data) * 0.05
    data += noise

    # If a stimulus appears on left, it's [data, zeros]; if on right, it's [zeros, data]
    left_data = torch.cat((data, torch.zeros_like(data)), dim=1)
    right_data = torch.cat((torch.zeros_like(data), data), dim=1)

    return left_data, right_data
