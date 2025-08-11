import torch
import torch.utils.data as data_utils
import scipy.io
import numpy as np

# --- 1. Define Dataset class with integrated data loading ---
class CustomDataset(data_utils.Dataset):
    def __init__(self, args, split_type='train'):
        """
        Dataset class that handles data loading, preprocessing and splitting internally.
        
        Args:
            args (dict): Configuration dictionary containing:
                - data_path (str): Path to the .mat file
                - input_length (int): Length of input sequence
                - target_length (int): Length of target sequence
                - downsample_factor (int, optional): Downsampling factor
                - variables_input (list): List of input variable indices
                - variables_output (list): List of output variable indices
                - train_split_ratio (float): Ratio for training set
                - val_split_ratio (float): Ratio for validation set
            split_type (str): One of 'train', 'val', or 'test'
        """
        super(CustomDataset, self).__init__()
        self.args = args
        self.input_length = args['input_length']
        self.target_length = args['target_length']
        self.downsample_factor = args.get('downsample_factor', 1)
        self.split_type = split_type

        # Load and preprocess data
        self._load_and_preprocess_data()

        # Split data according to split_type
        self._split_data()

        # Create sample indices
        self._create_sample_indices()

    def _load_and_preprocess_data(self):
        """Load data from .mat file and perform preprocessing"""
        # Load data
        mat_data = scipy.io.loadmat(self.args['data_path'])
        print("Variables in file:", mat_data.keys())

        if 'u' not in mat_data:
            raise ValueError("Error: 'u' key not found in data")

        # Original shape: (1200, 64, 64, 20) -> (N, H, W, T_total)
        U_original = mat_data['u']
        print("Original U shape:", U_original.shape)

        # Adjust dimension order: (N, H, W, T_total) -> (N, T_total, H, W)
        U_permuted = np.transpose(U_original, (0, 3, 1, 2))
        print("U shape after dimension adjustment (N, T_total, H, W):", U_permuted.shape)

        # Convert to PyTorch Tensor and add channel dimension
        # [N, T_total, H, W] -> [N, T_total, 1, H, W]
        self.data_full = torch.from_numpy(U_permuted).float().unsqueeze(2)

    def _split_data(self):
        """Split data into train/val/test sets"""
        total_samples = self.data_full.shape[0]
        
        if self.split_type == 'train':
            self.start_index = 0
            self.end_index = int(self.args.get('train_split_ratio', 0.8) * total_samples)
        elif self.split_type == 'val':
            train_end_idx = int(self.args.get('train_split_ratio', 0.8) * total_samples)
            self.start_index = train_end_idx
            self.end_index = int((self.args.get('train_split_ratio', 0.8) + 
                                self.args.get('val_split_ratio', 0.1)) * total_samples)
        elif self.split_type == 'test':
            val_end_idx = int((self.args.get('train_split_ratio', 0.8) + 
                              self.args.get('val_split_ratio', 0.1)) * total_samples)
            self.start_index = val_end_idx
            self.end_index = total_samples
        else:
            raise ValueError("split_type must be 'train', 'val', or 'test'")

        self.data = self.data_full[self.start_index:self.end_index]
        self.num_samples_split = self.data.shape[0]
        self.num_time_steps_total = self.data.shape[1]
        self.num_variables = self.data.shape[2]  # Should be 1

        # Validate input/output variables
        self.variables_input = self.args.get('variables_input', [0])
        self.variables_output = self.args.get('variables_output', [0])
        if not isinstance(self.variables_input, list): 
            self.variables_input = [self.variables_input]
        if not isinstance(self.variables_output, list): 
            self.variables_output = [self.variables_output]
        
        for var_idx in self.variables_input + self.variables_output:
            if var_idx >= self.num_variables:
                raise ValueError(
                    f"Variable index {var_idx} out of range. "
                    f"Available channels: {self.num_variables}"
                )

    def _create_sample_indices(self):
        """Create indices for sliding window sampling"""
        self.sample_indices = []
        max_start_of_input_seq = self.num_time_steps_total - (self.input_length + self.target_length)

        if max_start_of_input_seq < 0:
            raise ValueError(
                f"Total timesteps {self.num_time_steps_total} (in samples {self.start_index}-{self.end_index-1}) "
                f"are insufficient for input_length ({self.input_length}) and target_length ({self.target_length})."
            )

        for s_idx_split in range(self.num_samples_split):
            for t_start_input in range(max_start_of_input_seq + 1):
                t_start_target = t_start_input + self.input_length
                self.sample_indices.append((s_idx_split, t_start_target))

    def __len__(self):
        if not self.sample_indices and self.num_samples_split > 0 and \
           (self.num_time_steps_total < (self.input_length + self.target_length)):
            print(
                f"Warning: For split_type (sample indices {self.start_index}-{self.end_index-1}), "
                f"total timesteps {self.num_time_steps_total} are insufficient to create any sequences of length "
                f"{self.input_length}+{self.target_length}. Dataset will be empty."
            )
        return len(self.sample_indices)

    def __getitem__(self, idx):
        s_idx_split, t_start_target = self.sample_indices[idx]

        input_seq = self.data[s_idx_split,
                             t_start_target - self.input_length : t_start_target,
                             self.variables_input, :, :]
        target_seq = self.data[s_idx_split,
                              t_start_target : t_start_target + self.target_length,
                              self.variables_output, :, :]

        if self.downsample_factor > 1:
            input_seq = input_seq[:, :, ::self.downsample_factor, ::self.downsample_factor]
            target_seq = target_seq[:, :, ::self.downsample_factor, ::self.downsample_factor]

        return input_seq.float(), target_seq.float()

# --- 2. Prepare Dataset and DataLoader ---
if __name__ == '__main__':
    # Define configuration
    config = {
        'data_path': 'NavierStokes_V1e-5_N1200_T20.mat',  # Only need to specify data path here
        'input_length': 1,
        'target_length': 1,
        'downsample_factor': 1,
        'variables_input': [0],
        'variables_output': [0],
        'train_split_ratio': 0.8, 
        'val_split_ratio': 0.1,   
        'batch_size': 20
    }

    # Create datasets
    train_dataset = CustomDataset(config, split_type='train')
    val_dataset = CustomDataset(config, split_type='val')
    test_dataset = CustomDataset(config, split_type='test')

    print(f"Training samples (generated via sliding window): {len(train_dataset)}")
    print(f"Validation samples (generated via sliding window): {len(val_dataset)}")
    print(f"Test samples (generated via sliding window): {len(test_dataset)}")

    # Create dataloaders
    train_loader = data_utils.DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True, 
        num_workers=0
    )
    val_loader = data_utils.DataLoader(
        val_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False, 
        num_workers=0
    )
    test_loader = data_utils.DataLoader(
        test_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False, 
        num_workers=0
    )

    # Test the dataloaders
    print("\n--- Testing DataLoaders ---")
    if len(train_loader) > 0:
        sample_batch_input, sample_batch_target = next(iter(train_loader))
        print("Input batch shape from Train DataLoader (B, T_input, C, H, W):", sample_batch_input.shape)
        print("Target batch shape from Train DataLoader (B, T_target, C, H, W):", sample_batch_target.shape)
        print(f"  Batch size (B): {sample_batch_input.shape[0]}")
        print(f"  Input timesteps (T_input): {sample_batch_input.shape[1]}")
        print(f"  Channels (C): {sample_batch_input.shape[2]}")  # Should be 1
        print(f"  Height (H): {sample_batch_input.shape[3]}")
        print(f"  Width (W): {sample_batch_input.shape[4]}")
    else:
        print("Training set is empty. Check data splitting and sequence length parameters.")

    if len(val_loader) > 0:
        sample_batch_input_val, _ = next(iter(val_loader))
        print("Input batch shape from Val DataLoader (B, T_input, C, H, W):", sample_batch_input_val.shape)
    else:
        print("Validation set is empty.")

    if len(test_loader) > 0:
        sample_batch_input_test, _ = next(iter(test_loader))
        print("Input batch shape from Test DataLoader (B, T_input, C, H, W):", sample_batch_input_test.shape)
    else:
        print("Test set is empty.")