class Config:
    def __init__(self):
        # GPU Configuration
        self.gpus = "0"  # GPU device to use
        
        # Dataset Configuration
        self.data = 'PaviaU'  # Options: 'PaviaU', 'Indian', 'Houston2018', 'Houston2013'
        
        # Data Processing Parameters
        self.test_ratio = 0.9  # Proportion of test samples
        self.patch_size = 15   # Patch size for image cubes
        self.pca_components = 30  # Number of PCA components
        
        # Model Architecture Parameters
        self.embed_dim = 96    # Embedding dimension
        self.depth = 4         # Number of transformer layers
        self.num_classes = 9   # Number of classes (adjust based on dataset)
        
        # VideoMamba specific parameters
        self.group_type = 'groupScan'  # Type of grouping
        self.scan_type = 'globalBiScan'  # Scanning type
        self.k_group = 4       # Number of groups
        self.dt_rank = 32      # Rank of the dynamic routing tensor
        self.d_state = 16      # State dimension of the model
        self.dim_inner = 384   # Inner dimension of the model
        self.pos = True        # Use positional encoding
        self.cls = True        # Use classification token
        
        # 3D Convolution Parameters
        self.conv3D_channel = 16   # 3D convolution channels
        self.conv3D_kernel = (3, 3, 3)  # 3D convolution kernel size
        self.dim_patch = 15 * 15   # Patch dimension (patch_size * patch_size)
        self.dim_linear = 512      # Linear layer dimension
        
        # Training Parameters
        self.train_epoch = 100     # Number of training epochs
        self.test_epoch = 10       # Number of test runs
        self.BATCH_SIZE_TRAIN = 64  # Training batch size
        
        # Directory Configurations
        self.checkpoint_path = './checkpoints/'
        self.logs = './logs/'
        
        # Dataset-specific class configurations
        self.dataset_classes = {
            'PaviaU': 9,
            'Indian': 16,
            'Houston2018': 20,
            'Houston2013': 15
        }
        
        # Update num_classes based on selected dataset
        if self.data in self.dataset_classes:
            self.num_classes = self.dataset_classes[self.data]
    
    def update_dataset(self, dataset_name):
        """Update configuration for different datasets"""
        self.data = dataset_name
        if dataset_name in self.dataset_classes:
            self.num_classes = self.dataset_classes[dataset_name]
        
        # Dataset-specific optimizations
        if dataset_name == 'PaviaU':
            self.pca_components = 30
            self.patch_size = 15
        elif dataset_name == 'Indian':
            self.pca_components = 25
            self.patch_size = 13
        elif dataset_name == 'Houston2018':
            self.pca_components = 35
            self.patch_size = 17
        elif dataset_name == 'Houston2013':
            self.pca_components = 30
            self.patch_size = 15
        
        # Update dependent parameters
        self.dim_patch = self.patch_size * self.patch_size

# Create global config instance
config = Config()