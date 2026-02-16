#!/usr/bin/env python3
"""
Auto-detect optimal device (GPU/CPU) for PyTorch inference.
Returns 'cuda' if GPU is available and functioning, otherwise 'cpu'.
"""
import sys
try:
    import torch
    
    if torch.cuda.is_available():
        # Test if GPU actually works
        try:
            device = torch.device('cuda')
            test_tensor = torch.zeros(1).to(device)
            print('cuda')
            sys.exit(0)
        except Exception as e:
            print(f'cpu  # GPU detected but failed test: {e}', file=sys.stderr)
            print('cpu')
            sys.exit(0)
    else:
        print('cpu')
        sys.exit(0)
except ImportError:
    print('cpu  # torch not available')
    sys.exit(0)
