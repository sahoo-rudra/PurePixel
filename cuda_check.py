import sys
import time
import torch

def main():
    print("=== CUDA VERIFICATION ===")
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")
    
    if not cuda_available:
        print("CUDA is not available. GPU environment is not ready.")
        sys.exit(1)
        
    device_count = torch.cuda.device_count()
    print(f"Device count: {device_count}")
    current_device = torch.cuda.current_device()
    device_name = torch.cuda.get_device_name(current_device)
    print(f"Device name: {device_name}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"cuDNN version: {torch.backends.cudnn.version()}")
    
    # Try enabling cudnn benchmark
    try:
        torch.backends.cudnn.benchmark = True
        print("torch.backends.cudnn.benchmark enabled successfully.")
    except Exception as e:
        print(f"Failed to enable cudnn.benchmark: {e}")
        
    print("\n--- BENCHMARK ---")
    size = 4096
    print(f"Performing {size}x{size} matrix multiplication benchmark...")
    
    # CPU calculation
    cpu_tensor = torch.randn(size, size)
    start_cpu = time.time()
    _ = torch.matmul(cpu_tensor, cpu_tensor)
    cpu_time = time.time() - start_cpu
    print(f"CPU time: {cpu_time:.4f} seconds")
    
    # GPU calculation
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gpu_tensor = torch.randn(size, size).to(device)
    
    # Warmup
    _ = torch.matmul(gpu_tensor, gpu_tensor)
    if torch.cuda.is_available():
        torch.cuda.synchronize(device)
        
    start_gpu = time.time()
    _ = torch.matmul(gpu_tensor, gpu_tensor)
    if torch.cuda.is_available():
        torch.cuda.synchronize(device)
    gpu_time = time.time() - start_gpu
    print(f"GPU time: {gpu_time:.4f} seconds")
    
    if gpu_time > 0:
        speedup = cpu_time / gpu_time
        print(f"Speedup: {speedup:.2f}x faster on GPU")
        
    print("\nEnvironment is ready for GPU training and inference")
    sys.exit(0)

if __name__ == "__main__":
    main()
