import torch
import numpy as np
import umap
from concurrent.futures import ThreadPoolExecutor

# Subgraph node information - read the file and extract the numeric part
def read_numeric_data(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            # Split each line and filter out the numeric part
            numeric_part = line.strip().split('\t')[0]
            numeric_part = [int(num) for num in numeric_part.split('-')]
            data.append(numeric_part)
    return data

# Convert the extracted numeric data into a 2D tensor
def convert_to_tensor(data):
    max_len = max(len(row) for row in data)  # Get the max length to create a 2D tensor
    padded_data = [row + [-1] * (max_len - len(row)) for row in data]  # Pad data with -1
    return torch.tensor(padded_data)

# Parallel connection check
def parallel_has_connection(subgraph, edge_tensor_data):
    connections = [False] * len(subgraph)
    for i, node1 in enumerate(subgraph):
        for j, node2 in enumerate(subgraph[i + 1:], start=i + 1):  # Start from i+1
            if (edge_tensor_data == torch.tensor([node1, node2], dtype=torch.float).cuda()).all(dim=1).any() or \
               (edge_tensor_data == torch.tensor([node2, node1], dtype=torch.float).cuda()).all(dim=1).any():
                connections[i] = True
                connections[j] = True
                break
    return connections

# Read subgraph - numeric part from the file
file_path = 'dataset/em_user/subgraphs.pth'
numeric_data = read_numeric_data(file_path)
A = subgraph_tensor_data = convert_to_tensor(numeric_data).cuda()

print("Shape of the 2D tensor for subgraphs:", subgraph_tensor_data.shape)  # (324, 499)
print("Subgraph data:")
print(subgraph_tensor_data)

# Read edge information - text file and convert to numpy array
data_array = np.loadtxt("dataset/em_user/edge_list.txt", dtype=int)
B = edge_tensor_data = torch.tensor(data_array).cuda()

print("Edge tensor shape:", edge_tensor_data.shape)  # (4573417, 2)
print("Edge tensor data:", edge_tensor_data)

# Create adjacency matrix tensor C
C = torch.zeros(324, 324).cuda()

# Count connections
def count_connections(nodes1, nodes2):
    count = 0
    for node1 in nodes1:
        for node2 in nodes2:
            if node1 != -1 and node2 != -1 and \
               (B[:, 0] == node1).any() and (B[:, 1] == node2).any():
                count += 1
    return count

# Fill the adjacency matrix
for i in range(324):
    for j in range(i + 1, 324):
        connections = count_connections(A[i], A[j])
        if connections >= 3:
            C[i, j] = 1

# Save the adjacency matrix tensor C
torch.save(C, 'subgraph_adjacency_em_user.pt')

# Output and save the adjacency matrix
print("Adjacency matrix tensor C:")
print(C)

# The above code is to obtain the adjacency matrix between subgraphs, where 1 indicates a connection and 0 indicates no connection.
# Based on the obtained subgraph adjacency matrix, calculate the longest and average distances achievable by the subgraphs.
# Used to calculate the farthest point that a node can reach
def floyd_warshall_cuda(adj_matrix):
    """
    Use the Floyd-Warshall algorithm to calculate the shortest path distance matrix, accelerated with CUDA.

    Parameters:
    adj_matrix (torch.Tensor): Adjacency matrix (CUDA tensor).

    Returns:
    torch.Tensor: Shortest path distance matrix (CUDA tensor).
    """
    len_nodes = adj_matrix.size(0)
    dist_matrix = torch.where(adj_matrix == 0, float('inf'), adj_matrix).float()
    dist_matrix = dist_matrix.to('cuda')
    torch.diagonal(dist_matrix).fill_(0)

    for k in range(len_nodes):
        dist_matrix = torch.min(dist_matrix, dist_matrix[:, k].view(-1, 1) + dist_matrix[k, :])

    return dist_matrix

def longest_and_average_distances_cuda(adj_matrix):
    """
    Calculate the longest and average distances each node can reach, accelerated with CUDA.

    Parameters:
    adj_matrix (torch.Tensor): Adjacency matrix (CUDA tensor).

    Returns:
    (torch.Tensor, torch.Tensor): Longest and average distances for each node (CUDA tensor).
    """
    dist_matrix = floyd_warshall_cuda(adj_matrix)

    # Calculate the longest distance: excluding self and unreachable nodes
    valid_distances = torch.where(dist_matrix == float('inf'), -float('inf'), dist_matrix)
    longest_distances = torch.max(valid_distances, dim=1)[0]

    # Calculate the average distance: excluding unreachable nodes
    valid_distances = torch.where(dist_matrix == float('inf'), 0, dist_matrix)
    reachability_counts = torch.sum(dist_matrix != float('inf'), dim=1) - 1  # Excluding self
    average_distances = torch.sum(valid_distances, dim=1) / reachability_counts

    return longest_distances, average_distances

# Calculate degrees
def compute_degrees(adj_matrix):
    """
    Calculate the degree of each node.

    Parameters:
    adj_matrix (torch.Tensor): Adjacency matrix.

    Returns:
    torch.Tensor: Degree of each node.
    """
    return torch.sum(adj_matrix, dim=1)

# Calculate the longest and average distances
longest_distances, average_distances = longest_and_average_distances_cuda(C)

# Round to one decimal place
longest_distances = torch.round(longest_distances * 10) / 10
average_distances = torch.round(average_distances * 10) / 10

# Calculate the degree of each node
degrees = compute_degrees(C)

# Calculate the product of the degree and the average distance
new_distances = degrees * average_distances
new_distances = torch.round(new_distances * 10) / 10

# Print results
print("Longest distances:", longest_distances)
print("Average distances:", average_distances)
print("Degree average distances:", new_distances)

# Calculate the denominator value for each node i
denominators = longest_distances + new_distances

# Create denominator matrix with the same shape as A
denominator_matrix = denominators.view(-1, 1).expand(-1, C.size(1))

# Calculate the parts of C1 where C[i,j] == 1
C1 = torch.sin((torch.pi / denominator_matrix) * C)

# Set the parts of C1 where C[i,j] == 0 to -1.5
C1[C == 0] = -1.5

# Print results
print("sin(C1)", C1)

# Convert the tensor A to a PyTorch tensor
C1_tensor = torch.tensor(C1, dtype=torch.float32).cuda()

# Use UMAP to reduce the dimensions of tensor A
umap_model = umap.UMAP(n_components=16, n_neighbors=15, min_dist=0.1, metric='euclidean')
C2_tensor = umap_model.fit_transform(C1_tensor)
# Save subgraph positional encoding
torch.save(C2_tensor, 'subgraph_position_code_em_user.pt')
print("Subgraph positional encoding:", C2_tensor)
