import torch
if torch.cuda.is_available():
    print("CUDA is available!")
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    # Perform a small tensor operation on GPU
    x = torch.tensor([1.0, 2.0, 3.0], device='cuda')
    print(f"Tensor on GPU: {x}")
else:
    print("CUDA is not available. Running on CPU.")
