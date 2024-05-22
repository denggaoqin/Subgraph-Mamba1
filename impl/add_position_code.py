import torch

def add_positional_encoding(A, B, C, D):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    A = A.to(device)
    B = B.to(device)
    C = C.to(device)
    D = D.to(device)

    c_in_a_indices = []

    for c_subgraph in C:
        for idx, a_subgraph in enumerate(A):
            if torch.equal(c_subgraph, a_subgraph):
                c_in_a_indices.append(idx)
                break


    c_in_a_indices = torch.tensor(c_in_a_indices, dtype=torch.long).to(device)

    C_positions = B[c_in_a_indices]

    D_with_positional_encoding = torch.zeros((D.size(0), D.size(1) + B.size(1)), device=device)

    current_subgraph_idx = 0

    d_idx = 0

    for subgraph in C:
        positional_encoding = C_positions[current_subgraph_idx]

        for node in subgraph:
            if node != -1:
                if d_idx >= D.size(0):
                    print(f"Warning: d_idx {d_idx} exceeds the number of rows in D.")
                    continue

                node_feature = D[d_idx]


                D_with_positional_encoding[d_idx] = torch.cat((node_feature, positional_encoding))


                d_idx += 1

        current_subgraph_idx += 1

    return D_with_positional_encoding


if __name__ == "__main__":
    torch.manual_seed(42)

    A = torch.randint(0, 100, (400, 160), dtype=torch.int)

    B = torch.rand(400, 16)

    D = torch.rand(801, 64)

    indices = torch.randperm(A.size(0))[:50]
    C = A[indices]

    D_with_positional_encoding = add_positional_encoding(A, B, C, D)


    print("D_with_positional_encoding shape:", D_with_positional_encoding.shape)
    print(D_with_positional_encoding)
