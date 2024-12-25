import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import transforms
from torch_geometric.nn import GraphConv, global_max_pool
import networkx as nx
from torch_geometric.data import Data
from torch.utils.data import DataLoader, Dataset
import qutip as qt
from torch.utils.data import random_split, DataLoader
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Generate clean surface code using numpy
def generate_surface_code(size):
    return np.array(np.random.randint(0, 2, size=(size, size)))

def plot_graphs(ground_truth, noisy, confidence, syndromes, corrected):
    """
    Visualizes ground truth, noisy input, GNN confidence predictions, syndromes, and corrected output.
    """
    def to_numpy(data):
        if isinstance(data, torch.Tensor):
            return data.cpu().detach().numpy()
        return data

    ground_truth = to_numpy(ground_truth)
    noisy = to_numpy(noisy)
    confidence = to_numpy(confidence)
    syndromes = to_numpy(syndromes)
    corrected = to_numpy(corrected)

    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    titles = [
        "Ground Truth (Clean Surface Code)",
        "Noisy Surface Code",
        "Predicted Output (Confidence)",
        "Syndromes (Errors)",
        "Corrected Surface Code"
    ]
    data_list = [ground_truth, noisy, confidence, syndromes, corrected]

    for ax, data, title in zip(axes, data_list, titles):
        im = ax.imshow(data, cmap="gray", vmin=0, vmax=1)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
        fig.colorbar(im, ax=ax, orientation='vertical')

    plt.tight_layout()
    plt.show()

class SurfaceCodeDataset(Dataset):
    def __init__(self, size, temperature, noise_level, num_samples):
        self.size = size
        self.temperature = temperature
        self.noise_level = noise_level
        self.num_samples = num_samples
        self.data = self.generate_data()

    def generate_data(self):
        # Generate clean and noisy surface codes for all samples in one go
        data = []
        for _ in range(self.num_samples):
            clean_surface_code = generate_surface_code(self.size)
            noisy_surface_code = simulate_surface_code_with_noise(
                clean_surface_code, self.size, self.temperature, self.noise_level
            )
            cnn_input = preprocess_data(noisy_surface_code)
            data.append((cnn_input, torch.tensor(clean_surface_code, dtype=torch.float), noisy_surface_code))
        return data

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx]

#TODO: make accurate bosanic bath model

# Simulate surface code with noise
def simulate_surface_code_with_noise(surface_code, size, temperature, noise_level):
    noisy_surface_code = surface_code.copy()  # Create a copy to avoid modifying the original surface code

    for i in range(size):
        for j in range(size):
            if random.random() < noise_level:  # Apply noise based on the noise level
                if temperature > 0:  # If there's a temperature effect, apply probabilistic flipping
                    flip_probability = 1.0 / (1.0 + np.exp(-2 * temperature))  # Logistic function for flip probability
                    if random.random() < flip_probability:
                        noisy_surface_code[i][j] = 1 - noisy_surface_code[i][j]  # Flip the bit
                else:
                    noisy_surface_code[i][j] = 1 - noisy_surface_code[i][j]  # Flip the bit randomly

    return noisy_surface_code

# Generate syndromes for error detection
import numpy as np

def generate_syndromes(noisy_surface_code):
    """
    Generates syndromes based on parity checks in the noisy surface code,
    handling boundary conditions.

    Args:
    - noisy_surface_code (np.array): The noisy 2D surface code.

    Returns:
    - syndromes (np.array): A binary array indicating syndrome locations.
    """
    size = len(noisy_surface_code)
    syndromes = np.zeros_like(noisy_surface_code)

    for i in range(1, size - 1):
        for j in range(1, size - 1):

            if (i % 2 == 1 and j % 2 == 0):
                parity = (
                    noisy_surface_code[i - 1, j] ^
                    noisy_surface_code[i + 1, j] ^
                    noisy_surface_code[i, j - 1] ^
                    noisy_surface_code[i, j + 1]
                )

            elif (i % 2 == 0 and j % 2 == 1):
                parity = (
                    noisy_surface_code[i - 1, j - 1] ^
                    noisy_surface_code[i - 1, j + 1] ^
                    noisy_surface_code[i + 1, j - 1] ^
                    noisy_surface_code[i + 1, j + 1]
                )

    # Handle boundary cases (edges)
    for i in range(1, size - 1, 2):  # Iterate over odd rows (X-stabilizers on edges)
        # Left edge
        parity = noisy_surface_code[i - 1, 0] ^ noisy_surface_code[i + 1, 0] ^ noisy_surface_code[i, 1]
        syndromes[i, 0] = parity

        # Right edge
        parity = noisy_surface_code[i - 1, size - 1] ^ noisy_surface_code[i + 1, size - 1] ^ noisy_surface_code[i, size - 2]
        syndromes[i, size - 1] = parity

    for j in range(1, size - 1, 2):  # Iterate over odd columns (Z-stabilizers on edges)
        # Top edge
        parity = noisy_surface_code[0, j - 1] ^ noisy_surface_code[0, j + 1] ^ noisy_surface_code[1, j]
        syndromes[0, j] = parity

        # Bottom edge
        parity = noisy_surface_code[size - 1, j - 1] ^ noisy_surface_code[size - 1, j + 1] ^ noisy_surface_code[size - 2, j]
        syndromes[size - 1, j] = parity

    return syndromes

# MWPM Decoder to correct errors based on syndrome
def mwpm_decoder_with_weights(syndromes, weights, size):
    """
    MWPM decoder using weighted syndromes with GNN confidence.

    Args:
    - syndromes (np.array): The syndrome array (binary array with 1s at error locations).
    - weights (np.array): The weight array (confidence values for each qubit based on CNN/GNN predictions).
    - size (int): The size of the surface code (size x size lattice).

    Returns:
    - corrections (np.array): The corrected surface code.
    """
    # Create a graph where each syndrome is a node
    G = nx.Graph()
    syndrome_positions = np.argwhere(syndromes == 1)  # Find the positions of the 1s in the syndrome

    # Normalize weights once for the entire array
    normalized_weights = (weights - np.min(weights)) / (np.max(weights) - np.min(weights))  # Normalize to [0, 1]

    # Add nodes to the graph
    for idx, (i, j) in enumerate(syndrome_positions):
        G.add_node(idx, pos=(i, j))

    # Add edges between all pairs of syndrome nodes, with weights based on CNN/GNN predictions
    for i in range(len(syndrome_positions)):
        for j in range(i + 1, len(syndrome_positions)):
            (i1, j1) = syndrome_positions[i]
            (i2, j2) = syndrome_positions[j]
            dist = np.abs(i1 - i2) + np.abs(j1 - j2)  # Manhattan distance between the two syndromes

            # Combine the Manhattan distance with the normalized GNN weights
            weight = dist - (normalized_weights[i1, j1] + normalized_weights[i2, j2])  # Subtract GNN confidence to prioritize lower-weight matches
            G.add_edge(i, j, weight=weight)

    # Use MWPM (minimum weight perfect matching) to find the best matching of syndrome nodes
    matching = nx.algorithms.matching.min_weight_matching(G, weight='weight')

    # Create an array for corrections
    corrections = np.zeros_like(syndromes)

    # Apply the corrections based on the MWPM matching
    for (i, j) in matching:
        (i1, j1) = syndrome_positions[i]
        (i2, j2) = syndrome_positions[j]
        corrections[i1, j1] = corrections[i2, j2] = 1  # Flip these syndromes to correct the error

    return corrections


# Preprocess data for CNN
def preprocess_data(data):
    data = data.astype(np.float32)
    data = torch.tensor(data).unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, size, size)
    return data

class ConfigurableCNN(nn.Module):
    def __init__(self, size, conv_layers=2, fc_layers=2, initial_channels=32):
        super(ConfigurableCNN, self).__init__()
        self.size = size

        # Create convolutional layers
        self.conv_layers = nn.ModuleList()
        in_channels = 1
        out_channels = initial_channels

        for i in range(conv_layers):
            # Add padding to preserve spatial dimensions
            self.conv_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=2))  # Padding = 2
            in_channels = out_channels
            out_channels = out_channels * 2  # Double channels in each layer

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Calculate input dimension for first FC layer
        self.fc1_input_dim = self._get_conv_output(size)

        # Create fully connected layers
        self.fc_layers = nn.ModuleList()
        fc_in_dim = self.fc1_input_dim

        for i in range(fc_layers - 1):  # -1 because last layer is fixed size
            fc_out_dim = fc_in_dim // 2  # Reduce dimensions by half
            self.fc_layers.append(nn.Linear(fc_in_dim, fc_out_dim))
            fc_in_dim = fc_out_dim

        # Final layer to output original image size
        self.fc_layers.append(nn.Linear(fc_in_dim, size * size))

    def _get_conv_output(self, size):
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, size, size)
            x = dummy_input

            # Pass through all conv layers
            for conv in self.conv_layers:
                x = self.pool(F.relu(conv(x)))

            return x.numel()

    def forward(self, x):
        # Conv layers
        for conv in self.conv_layers:
            x = self.pool(F.relu(conv(x)))

        # Flatten
        x = x.view(x.size(0), -1)

        # FC layers
        for i, fc in enumerate(self.fc_layers):
            x = fc(x)
            # Apply ReLU to all but the last layer
            if i < len(self.fc_layers) - 1:
                x = F.relu(x)

        return x


# Construct graph for GNN
def construct_graph(surface_code):
    G = nx.Graph()
    nodes = []
    size = len(surface_code)
    for i in range(size):
        for j in range(size):
            node_index = i * size + j
            G.add_node(node_index, state=surface_code[i][j])
            nodes.append([surface_code[i][j]])
            if i > 0:
                G.add_edge(node_index, node_index - size)
            if j > 0:
                G.add_edge(node_index, node_index - 1)
    edge_index = torch.tensor(list(G.edges), dtype=torch.long).t().contiguous()
    x = torch.tensor(nodes, dtype=torch.float)
    return Data(x=x, edge_index=edge_index)

# Define GNN model
class GNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GNN, self).__init__()
        self.conv1 = GraphConv(in_channels, hidden_channels)
        self.conv2 = GraphConv(hidden_channels, hidden_channels)
        self.fc1 = nn.Linear(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = global_max_pool(x, data.batch)
        x = self.fc1(x)
        return x

# Training function for CNN + GNN hybrid approach
def train_with_hybrid_mwpm(cnn_model, gnn_model, optimizer, scheduler, criterion, device, train_loader, num_epochs, size):
    cnn_model.train()
    gnn_model.train()

    train_losses = []
    train_accuracies = []

    for epoch in range(num_epochs):
        epoch_loss = 0
        correct = 0
        total = 0
        start_time = time.time()

        for cnn_inputs, clean_surface_codes, noisy_surface_codes in train_loader:
            cnn_inputs = cnn_inputs.to(device).squeeze(1)  # (batch_size, 1, size, size)
            clean_surface_codes = clean_surface_codes.to(device)

            optimizer.zero_grad()

            # Forward pass through CNN
            cnn_outputs = cnn_model(cnn_inputs).view(-1, size, size)

            batch_loss = 0
            for batch_idx in range(cnn_inputs.size(0)):
                # Pass CNN output to GNN to refine the error predictions
                graph_data = construct_graph(cnn_outputs[batch_idx].cpu().detach().numpy())
                graph_data = graph_data.to(device)
                gnn_output = gnn_model(graph_data).view(size, size)


                # Generate classical syndromes from noisy surface code
                syndromes = generate_syndromes(noisy_surface_codes[batch_idx])

                # Use the GNN output as the weight for MWPM
                weights = torch.sigmoid(gnn_output).cpu().detach().numpy()  # Convert to numpy for MWPM

                # Decode with MWPM
                corrections = mwpm_decoder_with_weights(syndromes, weights, size)

                # Clone the noisy surface code and apply the corrections
                corrected_surface_code = noisy_surface_codes[batch_idx].clone()  # Use .clone() instead of .copy()
                corrected_surface_code[corrections == 1] = 1 - corrected_surface_code[corrections == 1]

                loss = criterion(gnn_output, clean_surface_codes[batch_idx])
                batch_loss += loss  # Accumulate batch loss
                corrected_surface_code_tensor = corrected_surface_code.clone().detach().to(device).float().requires_grad_(True)

                # Calculate accuracy
                # accuracy isn't changing because gnn/cnn output isn't weighed enough in decoder (to-fix)
                predicted = (corrected_surface_code_tensor)
                correct += (predicted == clean_surface_codes[batch_idx]).sum().item()
                total += clean_surface_codes[batch_idx].numel()

            batch_loss.backward()
            optimizer.step()

            epoch_loss += batch_loss.item() # Accumulate total batch loss for the epoch

        avg_train_loss = epoch_loss / len(train_loader.dataset)
        train_losses.append(avg_train_loss)

        train_accuracy = 100 * correct / total
        train_accuracies.append(train_accuracy)

        print(f"Epoch {epoch + 1}/{num_epochs}: Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")
        end_time = time.time()
        print(f"Elapsed time of epoch {epoch+1}: {round(end_time-start_time,2)} seconds")
        scheduler.step()

        # Plot and visualize predictions every x epochs

        if (epoch + 1) % 1 == 0:

            # Call the updated plotting function
            plot_graphs(
                ground_truth=clean_surface_codes[batch_idx].cpu().detach(),
                noisy=noisy_surface_codes[batch_idx],
                confidence=torch.sigmoid(gnn_output.cpu().detach()),
                syndromes=syndromes,
                corrected=corrected_surface_code
            )


    # Final plots for loss and accuracy
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss', color='blue')
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(train_accuracies) + 1), train_accuracies, label='Train Accuracy', color='green')
    plt.title('Training Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    plt.tight_layout()
    plt.show()
    end_time = time.time()

# Usage
size = 5
temperature = 0.1
noise_level = 0.1

batch_size = 32
num_samples = 2000
num_epochs = 40
'''
batch_size = 1
num_samples = 5
num_epochs = 10
'''
learning_rate = 0.01
print(generate_surface_code(5))
cnn_model = ConfigurableCNN(size).to(device)
gnn_model = GNN(in_channels=1, hidden_channels=64, out_channels=size * size).to(device)
surface_codes = generate_surface_code(5)
optimizer = optim.Adam(list(cnn_model.parameters()) + list(gnn_model.parameters()), lr=learning_rate)
criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([2.0]).to(device))
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.7)

#Load dataset and train
dataset = SurfaceCodeDataset(size, temperature, noise_level, num_samples)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
train_with_hybrid_mwpm(cnn_model, gnn_model, optimizer, scheduler, criterion, device, train_loader, num_epochs, size)
print("nothing happened")