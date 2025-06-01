# full_cuda_test.py
import time
import torch
import numpy as np


def full_cuda_test():
    """Kapsamlı CUDA testi"""
    print("🚀 KAPSAMLI CUDA TESTİ")
    print("=" * 60)

    # 1. Sistem Bilgileri
    print("📊 Sistem Bilgileri:")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU count: {torch.cuda.device_count()}")

    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"GPU {i}: {props.name}")
        print(f"  Memory: {props.total_memory / 1e9:.1f} GB")
        print(f"  Compute: {props.major}.{props.minor}")

    # 2. CuPy kontrolü
    try:
        import cupy as cp
        print(f"\n✅ CuPy version: {cp.__version__}")
        print(f"CuPy CUDA version: {cp.cuda.runtime.runtimeGetVersion()}")
    except ImportError:
        print("\n❌ CuPy yüklü değil")

    # 3. Numba CUDA kontrolü
    try:
        from numba import cuda
        print(f"\n✅ Numba CUDA available")
        print(f"Numba GPU count: {len(cuda.gpus)}")
        for i, gpu in enumerate(cuda.gpus):
            print(f"  GPU {i}: {gpu.name}")
    except Exception as e:
        print(f"\n❌ Numba CUDA hatası: {e}")

    # 4. Performans Testleri
    print(f"\n⚡ PERFORMANS TESTLERİ")
    print("-" * 40)

    # Test boyutları
    sizes = [1000, 2000, 5000]

    for size in sizes:
        print(f"\n📏 Test boyutu: {size}x{size}")

        # CPU test
        a_cpu = torch.randn(size, size)
        b_cpu = torch.randn(size, size)

        start = time.time()
        c_cpu = torch.matmul(a_cpu, b_cpu)
        cpu_time = time.time() - start
        print(f"  CPU: {cpu_time:.3f}s")

        # GPU test
        device = torch.device('cuda')
        a_gpu = torch.randn(size, size, device=device)
        b_gpu = torch.randn(size, size, device=device)

        # Warm-up
        torch.cuda.synchronize()
        _ = torch.matmul(a_gpu, b_gpu)
        torch.cuda.synchronize()

        # Actual test
        start = time.time()
        c_gpu = torch.matmul(a_gpu, b_gpu)
        torch.cuda.synchronize()
        gpu_time = time.time() - start

        speedup = cpu_time / gpu_time if gpu_time > 0 else 0
        print(f"  GPU: {gpu_time:.3f}s")
        print(f"  🚀 Hızlanma: {speedup:.1f}x")

        # Doğruluk kontrolü
        c_gpu_cpu = c_gpu.cpu()
        max_diff = torch.max(torch.abs(c_cpu - c_gpu_cpu)).item()
        print(f"  ✓ Doğruluk: {max_diff:.2e}")

    # 5. Memory Test
    print(f"\n💾 MEMORY TESTİ")
    print("-" * 20)

    device = torch.device('cuda')

    # Mevcut memory
    allocated = torch.cuda.memory_allocated(device) / 1e6
    cached = torch.cuda.memory_reserved(device) / 1e6
    print(f"Allocated: {allocated:.1f} MB")
    print(f"Cached: {cached:.1f} MB")

    # Büyük tensor oluştur
    try:
        big_tensor = torch.randn(10000, 10000, device=device)
        allocated_after = torch.cuda.memory_allocated(device) / 1e6
        print(f"Big tensor allocated: {allocated_after - allocated:.1f} MB")
        del big_tensor
        torch.cuda.empty_cache()
        print("✅ Memory test başarılı")
    except RuntimeError as e:
        print(f"❌ Memory test hatası: {e}")

    # 6. Genel Değerlendirme
    print(f"\n🎯 SONUÇ")
    print("-" * 15)

    if torch.cuda.is_available():
        print("✅ CUDA tam olarak çalışıyor!")
        print("✅ GPU hesaplamalar için hazır")
        print("🚀 Drone projesi CUDA ile hızlandırılabilir")

        # Öneriler
        print(f"\n💡 ÖNERİLER:")
        print("1. Genetik algoritma popülasyon boyutunu artırın (100 → 200)")
        print("2. Mesafe hesaplamalarını GPU'ya taşıyın")
        print("3. Fitness hesaplamalarını batch olarak yapın")

        return True
    else:
        print("❌ CUDA problemi var")
        return False


if __name__ == "__main__":
    success = full_cuda_test()

    if success:
        print(f"\n🎉 CUDA hazır! Projenizde aşağıdaki kodu kullanabilirsiniz:")
        print("""
# CUDA device ayarlama
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Tensor'ları GPU'ya taşıma
tensor_gpu = torch.tensor(data, device=device)
        """)