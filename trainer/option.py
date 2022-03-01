import argparse

def basic_opt(description=''):
    parser = argparse.ArgumentParser(description=description)
    
    # Model specifications
    parser.add_argument('--prefix', '-p', type=str, default='cnn')
    
    # Training specifications
    parser.add_argument('--batchSize', '-b', type=int, default=16, help='Training batch size.')
    parser.add_argument('--nEpochs', '-n', type=int, default=800, help='Training epoch.')
    parser.add_argument('--dir', '-d', type=str, default='/media/exthdd/datasets/hsi/lzy_data/CAVE_22_10/CAVE64_31_22.db', help='Training Data.')
    
    # Optimization specifications
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate.')
    parser.add_argument('--min-lr', '-mlr', type=float, default=5e-5, help='Minimal learning rate.')

    # Checkpoint specifications
    parser.add_argument('--ri', type=int, default=50, help='Record interval.')
    parser.add_argument('--resume', '-r', action='store_true', help='Resume from checkpoint.')
    parser.add_argument('--resumePath', '-rp', type=str, help='Checkpoint to use.')

    # Hardware specifications
    parser.add_argument('--threads', type=int, default=8, help='Number of threads for data loader.')
    parser.add_argument('--no-cuda', action='store_true', help='Disable cuda?')
    
    # Log specifications
    parser.add_argument('--no-log', action='store_true', help='Disable logger?')
    
    opt = parser.parse_args()
    return opt