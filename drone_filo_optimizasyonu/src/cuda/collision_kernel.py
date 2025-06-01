import numpy as np
import cupy as cp
from numba import cuda


# Ray casting algoritması için kernel
@cuda.jit
def ray_casting_kernel(points_x, points_y, polygon_x, polygon_y, polygon_sizes, results):
    """
    Nokta-içinde-poligon testi için CUDA kerneli

    Parametreler:
    -------------
    points_x : array
        Test edilecek noktaların x koordinatları
    points_y : array
        Test edilecek noktaların y koordinatları
    polygon_x : array
        Tüm poligonların x koordinatları
    polygon_y : array
        Tüm poligonların y koordinatları
    polygon_sizes : array
        Her poligondaki köşe sayısı
    results : array
        Sonuç dizisi (1: içinde, 0: dışında)
    """
    point_idx = cuda.grid(1)

    if point_idx < points_x.size:
        x, y = points_x[point_idx], points_y[point_idx]

        # Her poligon için kontrol et
        polygon_start_idx = 0
        for poly_idx in range(polygon_sizes.size):
            n_vertices = polygon_sizes[poly_idx]
            inside = False

            # Ray casting algoritması
            j = n_vertices - 1
            for i in range(n_vertices):
                i_idx = polygon_start_idx + i
                j_idx = polygon_start_idx + j

                if (((polygon_y[i_idx] > y) != (polygon_y[j_idx] > y)) and
                        (x < (polygon_x[j_idx] - polygon_x[i_idx]) * (y - polygon_y[i_idx]) /
                         (polygon_y[j_idx] - polygon_y[i_idx]) + polygon_x[i_idx])):
                    inside = not inside

                j = i

            if inside:
                results[point_idx] = 1
                return

            polygon_start_idx += n_vertices


@cuda.jit
def line_intersection_kernel(line_starts_x, line_starts_y, line_ends_x, line_ends_y,
                             polygon_x, polygon_y, polygon_sizes, results):
    """
    Çizgi-poligon kesişimi testi için CUDA kerneli

    Parametreler:
    -------------
    line_starts_x : array
        Çizgi başlangıç noktalarının x koordinatları
    line_starts_y : array
        Çizgi başlangıç noktalarının y koordinatları
    line_ends_x : array
        Çizgi bitiş noktalarının x koordinatları
    line_ends_y : array
        Çizgi bitiş noktalarının y koordinatları
    polygon_x : array
        Tüm poligonların x koordinatları
    polygon_y : array
        Tüm poligonların y koordinatları
    polygon_sizes : array
        Her poligondaki köşe sayısı
    results : array
        Sonuç dizisi (1: kesişim var, 0: kesişim yok)
    """
    line_idx = cuda.grid(1)

    if line_idx < line_starts_x.size:
        x1, y1 = line_starts_x[line_idx], line_starts_y[line_idx]
        x2, y2 = line_ends_x[line_idx], line_ends_y[line_idx]

        # Her poligon için kontrol et
        polygon_start_idx = 0
        for poly_idx in range(polygon_sizes.size):
            n_vertices = polygon_sizes[poly_idx]

            # Çizginin her kenarla kesişimini kontrol et
            for i in range(n_vertices):
                i_idx = polygon_start_idx + i
                j_idx = polygon_start_idx + ((i + 1) % n_vertices)

                x3, y3 = polygon_x[i_idx], polygon_y[i_idx]
                x4, y4 = polygon_x[j_idx], polygon_y[j_idx]

                # Çizgi kesişim algoritması
                den = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)
                if den == 0:  # Paralel çizgiler
                    continue

                ua = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / den
                ub = ((x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)) / den

                if 0 <= ua <= 1 and 0 <= ub <= 1:
                    results[line_idx] = 1
                    return

            polygon_start_idx += n_vertices


def check_points_in_polygons_gpu(points, polygons):
    """
    GPU kullanarak noktaların poligonlar içinde olup olmadığını kontrol eder

    Parametreler:
    -------------
    points : list
        (x, y) koordinatları listesi
    polygons : list
        Poligon listesi, her biri [(x1,y1), (x2,y2), ...] şeklinde

    Dönüş:
    ------
    numpy.ndarray
        Her nokta için sonuç (1: en az bir poligon içinde, 0: hiçbir poligon içinde değil)
    """
    num_points = len(points)

    # Noktaları ayır
    points_x = np.array([p[0] for p in points], dtype=np.float32)
    points_y = np.array([p[1] for p in points], dtype=np.float32)

    # Poligon verilerini düzenle
    polygon_sizes = np.array([len(poly) for poly in polygons], dtype=np.int32)
    total_vertices = sum(polygon_sizes)

    polygon_x = np.zeros(total_vertices, dtype=np.float32)
    polygon_y = np.zeros(total_vertices, dtype=np.float32)

    # Tüm poligon köşelerini düz dizilere doldur
    idx = 0
    for poly in polygons:
        for vertex in poly:
            polygon_x[idx] = vertex[0]
            polygon_y[idx] = vertex[1]
            idx += 1

    # GPU'ya kopyala
    d_points_x = cuda.to_device(points_x)
    d_points_y = cuda.to_device(points_y)
    d_polygon_x = cuda.to_device(polygon_x)
    d_polygon_y = cuda.to_device(polygon_y)
    d_polygon_sizes = cuda.to_device(polygon_sizes)

    # Sonuç dizisi
    d_results = cuda.device_array(num_points, dtype=np.int32)
    d_results.fill(0)  # İlk değer: hiçbir poligon içinde değil

    # Kernel konfigürasyonu
    threads_per_block = 256
    blocks_per_grid = (num_points + threads_per_block - 1) // threads_per_block

    # Kernel çalıştır
    ray_casting_kernel[blocks_per_grid, threads_per_block](
        d_points_x, d_points_y, d_polygon_x, d_polygon_y, d_polygon_sizes, d_results)

    # Sonucu CPU'ya kopyala
    results = d_results.copy_to_host()

    return results


def check_line_intersections_gpu(lines, polygons):
    """
    GPU kullanarak çizgilerin poligonlarla kesişip kesişmediğini kontrol eder

    Parametreler:
    -------------
    lines : list
        Çizgi listesi, her biri [(x1,y1), (x2,y2)] şeklinde
    polygons : list
        Poligon listesi, her biri [(x1,y1), (x2,y2), ...] şeklinde

    Dönüş:
    ------
    numpy.ndarray
        Her çizgi için sonuç (1: en az bir poligonla kesişiyor, 0: hiçbir poligonla kesişmiyor)
    """
    num_lines = len(lines)

    # Çizgileri ayır
    line_starts_x = np.array([line[0][0] for line in lines], dtype=np.float32)
    line_starts_y = np.array([line[0][1] for line in lines], dtype=np.float32)
    line_ends_x = np.array([line[1][0] for line in lines], dtype=np.float32)
    line_ends_y = np.array([line[1][1] for line in lines], dtype=np.float32)

    # Poligon verilerini düzenle
    polygon_sizes = np.array([len(poly) for poly in polygons], dtype=np.int32)
    total_vertices = sum(polygon_sizes)

    polygon_x = np.zeros(total_vertices, dtype=np.float32)
    polygon_y = np.zeros(total_vertices, dtype=np.float32)

    # Tüm poligon köşelerini düz dizilere doldur
    idx = 0
    for poly in polygons:
        for vertex in poly:
            polygon_x[idx] = vertex[0]
            polygon_y[idx] = vertex[1]
            idx += 1

    # GPU'ya kopyala
    d_line_starts_x = cuda.to_device(line_starts_x)
    d_line_starts_y = cuda.to_device(line_starts_y)
    d_line_ends_x = cuda.to_device(line_ends_x)
    d_line_ends_y = cuda.to_device(line_ends_y)
    d_polygon_x = cuda.to_device(polygon_x)
    d_polygon_y = cuda.to_device(polygon_y)
    d_polygon_sizes = cuda.to_device(polygon_sizes)

    # Sonuç dizisi
    d_results = cuda.device_array(num_lines, dtype=np.int32)
    d_results.fill(0)  # İlk değer: hiçbir poligonla kesişmiyor

    # Kernel konfigürasyonu
    threads_per_block = 256
    blocks_per_grid = (num_lines + threads_per_block - 1) // threads_per_block

    # Kernel çalıştır
    line_intersection_kernel[blocks_per_grid, threads_per_block](
        d_line_starts_x, d_line_starts_y, d_line_ends_x, d_line_ends_y,
        d_polygon_x, d_polygon_y, d_polygon_sizes, d_results)

    # Sonucu CPU'ya kopyala
    results = d_results.copy_to_host()

    return results


def check_paths_against_no_fly_zones(paths, no_fly_zones):
    """
    Bir dizi yolu no-fly zone'lara karşı kontrol eder

    Parametreler:
    -------------
    paths : list
        Yol listesi, her biri [(x1,y1), (x2,y2), ...] şeklinde
    no_fly_zones : list
        NoFlyZone nesnelerinin listesi

    Dönüş:
    ------
    list
        Her yol için boolean listesi (True: engelleniyor, False: engellenmiyor)
    """
    # Tüm yollardaki tüm çizgi segmentlerini oluştur
    all_segments = []
    segment_to_path_map = {}  # Her segmentin hangi yola ait olduğunu sakla

    for path_idx, path in enumerate(paths):
        for i in range(len(path) - 1):
            segment = (path[i], path[i + 1])
            segment_idx = len(all_segments)
            all_segments.append(segment)
            segment_to_path_map[segment_idx] = path_idx

    # Aktif no-fly zone'ların poligonlarını al
    active_polygons = [nfz.coordinates for nfz in no_fly_zones]

    if not active_polygons or not all_segments:
        return [False] * len(paths)

    # Tüm segmentleri kontrol et
    intersections = check_line_intersections_gpu(all_segments, active_polygons)

    # Her yol için sonuçları grupla
    path_blocked = [False] * len(paths)

    for segment_idx, has_intersection in enumerate(intersections):
        if has_intersection:
            path_idx = segment_to_path_map[segment_idx]
            path_blocked[path_idx] = True

    return path_blocked