import numpy as np
import cupy as cp
from numba import cuda


# Genetik algoritma için fitness hesaplama kerneli
@cuda.jit
def fitness_kernel(drone_assignments, drone_capacities, delivery_weights,
                   delivery_priorities, distance_matrix, results):
    """
    Genetik algoritma için fitness değerini hesaplayan CUDA kerneli

    Parametreler:
    -------------
    drone_assignments : array
        Her teslimat için drone atamaları (2D dizi: [popülasyon_boyutu, teslimat_sayısı])
    drone_capacities : array
        Her drone'un maksimum kapasitesi
    delivery_weights : array
        Her teslimatın ağırlığı
    delivery_priorities : array
        Her teslimatın önceliği
    distance_matrix : array
        Noktalar arası mesafeler (ilk nokta depo)
    results : array
        Hesaplanan fitness değerleri
    """
    individual_idx = cuda.grid(1)

    if individual_idx < drone_assignments.shape[0]:
        # Toplam istatistikleri başlat
        total_deliveries = 0
        total_distance = 0.0
        total_violations = 0

        num_drones = len(drone_capacities)
        num_deliveries = drone_assignments.shape[1]

        # Her drone için yük ve rota bilgileri
        for drone_id in range(num_drones):
            current_load = 0.0
            current_position = 0  # Depo

            # Bu drone'a atanan teslimatları bul
            for delivery_id in range(num_deliveries):
                if drone_assignments[individual_idx, delivery_id] == drone_id:
                    # Bu teslimat bu drone'a atanmış

                    # Teslimat pozisyonu (depo 0, teslimatlar 1'den başlıyor)
                    delivery_position = delivery_id + 1

                    # Mesafeyi ekle
                    total_distance += distance_matrix[current_position, delivery_position]

                    # Yükü güncelle
                    current_load += delivery_weights[delivery_id]

                    # Kapasite kontrolü
                    if current_load > drone_capacities[drone_id]:
                        total_violations += 1

                    # Teslimat tamamlandı
                    total_deliveries += 1

                    # Pozisyonu güncelle
                    current_position = delivery_position

            # Depoya dönüş mesafesi
            total_distance += distance_matrix[current_position, 0]

        # Fitness hesaplama: teslimat_sayısı * 100 - mesafe * 0.5 - ihlal * 2000
        fitness = (total_deliveries * 100) - (total_distance * 0.5) - (total_violations * 2000)

        # Sonucu kaydet
        results[individual_idx] = fitness


def calculate_fitness_gpu(population, drones, delivery_points, distance_matrix):
    """
    GPU kullanarak popülasyonun fitness değerlerini hesaplar

    Parametreler:
    -------------
    population : numpy.ndarray
        Her satırı bir birey olan popülasyon matrisi
    drones : list
        Drone nesnelerinin listesi
    delivery_points : list
        DeliveryPoint nesnelerinin listesi
    distance_matrix : numpy.ndarray
        Noktalar arası mesafe matrisi

    Dönüş:
    ------
    numpy.ndarray
        Her birey için fitness değerleri
    """
    pop_size = population.shape[0]

    # Drone kapasiteleri
    drone_capacities = np.array([drone.max_weight for drone in drones], dtype=np.float32)

    # Teslimat ağırlıkları ve öncelikleri
    delivery_weights = np.array([dp.weight for dp in delivery_points], dtype=np.float32)
    delivery_priorities = np.array([dp.priority for dp in delivery_points], dtype=np.int32)

    # GPU'ya kopyala
    d_population = cuda.to_device(population)
    d_drone_capacities = cuda.to_device(drone_capacities)
    d_delivery_weights = cuda.to_device(delivery_weights)
    d_delivery_priorities = cuda.to_device(delivery_priorities)
    d_distance_matrix = cuda.to_device(distance_matrix)

    # Sonuç dizisi
    d_results = cuda.device_array(pop_size, dtype=np.float32)

    # Kernel konfigürasyonu
    threads_per_block = 256
    blocks_per_grid = (pop_size + threads_per_block - 1) // threads_per_block

    # Kernel çalıştır
    fitness_kernel[blocks_per_grid, threads_per_block](
        d_population, d_drone_capacities, d_delivery_weights,
        d_delivery_priorities, d_distance_matrix, d_results)

    # Sonucu CPU'ya kopyala
    results = d_results.copy_to_host()

    return results