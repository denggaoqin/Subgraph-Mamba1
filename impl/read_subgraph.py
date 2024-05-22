import torch

def load_data_as_tensor(file_path):
    # Read the file and extract the numeric parts
    def read_numeric_data(file_path):
        data = []
        with open(file_path, 'r') as file:
            for line in file:
                # Split each line and filter out the numeric parts
                numeric_part = line.strip().split('\t')[0]
                numeric_part = [int(num) for num in numeric_part.split('-')]
                data.append(numeric_part)
        return data

    # Convert the extracted numeric data to a 2D tensor
    def convert_to_tensor(data):
        max_len = max(len(row) for row in data)  # Get the max length to create a 2D tensor
        padded_data = [row + [-1] * (max_len - len(row)) for row in data]  # Pad the data with -1
        return torch.tensor(padded_data)

    # Read the numeric parts from the file
    numeric_data = read_numeric_data(file_path)

    # Convert the numeric data to a 2D tensor
    tensor_data = convert_to_tensor(numeric_data)

    return tensor_data

# Example usage
if __name__ == "__main__":
    # File path
    file_path = '../dataset/ppi_bp/subgraphs.pth'

    # Call the function to load data and convert it to a tensor
    tensor_data = load_data_as_tensor(file_path)

    print("Shape of the 2D tensor:", tensor_data.shape)
    print("Sample data:")
    print(tensor_data)
