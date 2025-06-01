import numpy as np
import cupy as cp
from numba import cuda

# CuPy kernel for pairwise distance calculation
distance_kernel = cp.RawKernel(r'''
extern "C" __global__
void distance_kernel(const float* points_x, const float* points_y, 
                     float* result_matrix, int num_points) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;

    if (i < num_points && j < num_points) {
        float dx = points_x[i] - points_x[j];
        float dy = points_y[i] - points_y[j];
        result_matrix[i * num_points + j] = sqrtf(dx*dx + dy*dy);
    }
}
''', 'distance_kernel')


def calculate_distances_gpu(points):
    """
    CuPy kullanarak GPU üzerinde tüm noktalar arasındaki mesafeleri hesaplar

    Parametreler:
    -------------
    points : list
        (x, y) koordinatları listesi

    Dönüş:
    ------
    numpy.ndarray
        Mesafe matrisi (CPU'ya geri aktarılmış)
    """
    num_points = len(points)

    # Noktaları x ve y koordinatlarına ayır
    points_x = np.array([p[0] for p in points], dtype=np.float32)
    points_y = np.array([p[1] for p in points], dtype=np.float32)

    # GPU'ya kopyala
    d_points_x = cp.array(points_x)
    d_points_y = cp.array(points_y)

    # Sonuç matrisini oluştur
    d_result = cp.zeros((num_points, num_points), dtype=cp.float32)

    # Grid ve blok boyutlarını hesapla
    threads_per_block = (16, 16)
    blocks_per_grid_x = (num_points + threads_per_block[0] - 1) // threads_per_block[0]
    blocks_per_grid_y = (num_points + threads_per_block[1] - 1) // threads_per_block[1]
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

    # Kernel'i çalıştır
    distance_kernel(blocks_per_grid, threads_per_block,
                    (d_points_x, d_points_y, d_result, num_points))

    # Sonucu CPU'ya geri aktar
    result = d_result.get()

    return result


# Numba ile CUDA kerneli
@cuda.jit
def numba_distance_kernel(points_x, points_y, result_matrix):
    """
    Numba ile CUDA kullanarak mesafe hesaplama kerneli

    Parametreler:
    -------------
    points_x : array
        x koordinatları dizisi
    points_y : array
        y koordinatları dizisi
    result_matrix : array
        Sonuç matrisi
    """
    i, j = cuda.grid(2)
    if i < points_x.size and j < points_x.size:
        dx = points_x[i] - points_x[j]
        dy = points_y[i] - points_y[j]
        result_matrix[i, j] = (dx * dx + dy * dy) ** 0.5


def calculate_distances_numba(points):
    """
    Numba kullanarak GPU üzerinde tüm noktalar arasındaki mesafeleri hesaplar

    Parametreler:
    -------------
    points : list
        (x, y) koordinatları listesi

    Dönüş:
    ------
    numpy.ndarray
        Mesafe matrisi
    """
    num_points = len(points)

    # Noktaları x ve y koordinatlarına ayır
    points_x = np.array([p[0] for p in points], dtype=np.float32)
    points_y = np.array([p[1] for p in points], dtype=np.float32)

    # GPU'ya kopyala
    d_points_x = cuda.to_device(points_x)
    d_points_y = cuda.to_device(points_y)

    # Sonuç matrisini oluştur
    d_result = cuda.device_array((num_points, num_points), dtype=np.float32)

    # Grid ve blok boyutlarını hesapla
    threads_per_block = (16, 16)
    blocks_per_grid_x = (num_points + threads_per_block[0] - 1) // threads_per_block[0]
    blocks_per_grid_y = (num_points + threads_per_block[1] - 1) // threads_per_block[1]

    # Kernel'i çalıştır
    numba_distance_kernel[(blocks_per_grid_x, blocks_per_grid_y), threads_per_block](
        d_points_x, d_points_y, d_result)

    # Sonucu CPU'ya geri aktar
    result = d_result.copy_to_host()

    return result