import argparse
import json
import time
import os
import sys
import warnings
import numpy as np
import matplotlib.pyplot as plt
import random

# Path düzeltmesi - src klasöründen çalıştırma için
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
sys.path.insert(0, current_dir)

# CUDA uyarılarını bastır
warnings.filterwarnings("ignore", message="CUDA path could not be detected")

# Model imports - src klasöründen çalıştırma için düzeltildi
try:
    # Proje kökünden çalıştırıldığında
    from src.models.drone import Drone
    from src.models.delivery_point import DeliveryPoint
    from src.models.no_fly_zone import NoFlyZone
    from src.algorithms.graph import DeliveryGraph
    from src.algorithms.astar import AStarPathfinder
    from src.algorithms.csp import DroneCSP
    from src.algorithms.genetic import GeneticAlgorithm
    from src.utils.visualizer import DroneVisualizer
    from src.utils.data_generator import DroneDataGenerator

    IMPORT_MODE = "root"
except ImportError:
    # src klasöründen çalıştırıldığında
    from models.drone import Drone
    from models.delivery_point import DeliveryPoint
    from models.no_fly_zone import NoFlyZone
    from algorithms.graph import DeliveryGraph
    from algorithms.astar import AStarPathfinder
    from algorithms.csp import DroneCSP
    from algorithms.genetic import GeneticAlgorithm
    from utils.visualizer import DroneVisualizer
    from utils.data_generator import DroneDataGenerator

    IMPORT_MODE = "src"

print(f"Import mode: {IMPORT_MODE}")

# CUDA imports with fallback - TEK SEFERLIK
CUDA_AVAILABLE = False
try:
    if IMPORT_MODE == "root":
        from src.cuda.distance_kernel import calculate_distances_gpu, calculate_distances_numba
        from src.cuda.collision_kernel import check_paths_against_no_fly_zones
        from src.cuda.fitness_kernel import calculate_fitness_gpu
    else:
        from cuda.distance_kernel import calculate_distances_gpu, calculate_distances_numba
        from cuda.collision_kernel import check_paths_against_no_fly_zones
        from cuda.fitness_kernel import calculate_fitness_gpu
    CUDA_AVAILABLE = True
    print("CUDA kütüphaneleri başarıyla yüklendi.")
except ImportError as e:
    print(f"CUDA mevcut değil, CPU hesaplama kullanılacak: {e}")


    # Fallback fonksiyonlar
    def calculate_distances_gpu(*args, **kwargs):
        return np.array([])


    def calculate_distances_numba(*args, **kwargs):
        return np.array([])


    def check_paths_against_no_fly_zones(*args, **kwargs):
        return np.array([])


    def calculate_fitness_gpu(*args, **kwargs):
        return 0.0

# Dinamik veri üretici için fallback
try:
    if IMPORT_MODE == "root":
        from src.utils.dynamic_data_generator import DynamicDroneDataGenerator
    else:
        from utils.dynamic_data_generator import DynamicDroneDataGenerator
    DYNAMIC_GENERATOR_AVAILABLE = True
    print("✅ Dinamik veri üretici yüklendi.")
except ImportError:
    print("⚠️ Dinamik veri üretici bulunamadı, standart versiyon kullanılacak.")
    DYNAMIC_GENERATOR_AVAILABLE = False


def get_project_root():
    """Proje kök dizinini döndürür"""
    current_file = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_file)

    if IMPORT_MODE == "root":
        # main.py proje kökünde, current_dir zaten proje kökü
        project_root = parent_dir  # parent_dir global olarak tanımlı
    else:
        # main.py src klasöründe, bir üst dizin proje kökü
        project_root = os.path.dirname(current_dir)

    return project_root


def ensure_directories():
    """Gerekli dizinleri oluşturur"""
    project_root = get_project_root()

    directories = [
        os.path.join(project_root, "data"),
        os.path.join(project_root, "visualizations"),
        os.path.join(project_root, "logs")
    ]

    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Dizin oluşturuldu/kontrol edildi: {directory}")


def load_data(data_file):
    """Veri dosyasını yükler"""
    project_root = get_project_root()
    full_path = os.path.join(project_root, data_file)

    try:
        with open(full_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Drone nesneleri oluştur
        drones = [Drone.from_dict(drone) for drone in data['drones']]

        # Teslimat noktası nesneleri oluştur
        delivery_points = [DeliveryPoint.from_dict(dp) for dp in data['delivery_points']]

        # No-fly zone nesneleri oluştur
        no_fly_zones = [NoFlyZone.from_dict(nfz) for nfz in data['no_fly_zones']]

        # Depo konumu
        depot_position = tuple(data['depot'])

        return drones, delivery_points, no_fly_zones, depot_position

    except FileNotFoundError:
        raise FileNotFoundError(f"Veri dosyası bulunamadı: {full_path}")
    except Exception as e:
        raise Exception(f"Veri yükleme hatası: {e}")


def calculate_distance(point1, point2):
    """İki nokta arasındaki mesafeyi hesaplar"""
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


def calculate_performance_metrics(routes, delivery_points, drones):
    """Performans metriklerini hesaplar - düzeltilmiş fonksiyon"""
    metrics = {
        'total_deliveries': len(delivery_points),
        'completed_deliveries': 0,
        'completion_percentage': 0,
        'total_energy_consumption': 0,
        'average_energy_per_drone': 0,
        'total_distance': 0
    }

    completed_deliveries = set()
    total_energy = 0

    for drone_id, route in routes.items():
        if len(route) > 1:  # Sadece depot'tan farklı rotalar
            # Mesafe hesabı
            route_distance = sum(
                calculate_distance(route[i], route[i + 1])
                for i in range(len(route) - 1)
            )
            metrics['total_distance'] += route_distance

            # Enerji hesabı
            drone = drones[drone_id]
            energy = route_distance * drone.speed * 0.1  # Örnek formül
            total_energy += energy

            # Tamamlanan teslimatlar
            for point in route[1:-1]:  # Başlangıç ve bitiş depot'u hariç
                for dp in delivery_points:
                    if abs(point[0] - dp.pos[0]) < 1 and abs(point[1] - dp.pos[1]) < 1:
                        completed_deliveries.add(dp.id)

    metrics['completed_deliveries'] = len(completed_deliveries)
    metrics['completion_percentage'] = (len(completed_deliveries) / len(delivery_points)) * 100 if len(
        delivery_points) > 0 else 0
    metrics['total_energy_consumption'] = total_energy
    metrics['average_energy_per_drone'] = total_energy / len(drones) if len(drones) > 0 else 0

    return metrics


def create_fallback_dynamic_data(num_drones=5, num_deliveries=20, num_nfz=3):
    """Dinamik veri üretici yoksa fallback versiyon"""
    print("📊 Fallback dinamik veri oluşturuluyor...")

    # Zaman bazlı seed
    current_time = int(time.time() * 1000) % 100000
    random.seed(current_time)
    np.random.seed(current_time)

    print(f"🎲 Dinamik seed: {current_time}")

    # Harita boyutu
    map_size = (1000, 1000)
    depot_position = (500 + random.randint(-50, 50), 500 + random.randint(-50, 50))

    # Drone verileri - çeşitli tipler
    drones = []
    drone_types = [
        {"weight_range": (5, 12), "battery_range": (4000, 8000), "speed_range": (15, 25)},
        {"weight_range": (8, 18), "battery_range": (6000, 10000), "speed_range": (12, 20)},
        {"weight_range": (15, 25), "battery_range": (8000, 15000), "speed_range": (8, 15)},
    ]

    for i in range(num_drones):
        drone_type = random.choice(drone_types)
        angle = random.uniform(0, 2 * 3.14159)
        distance = random.uniform(20, 100)

        start_x = int(depot_position[0] + distance * np.cos(angle))
        start_y = int(depot_position[1] + distance * np.sin(angle))

        drone_data = {
            'id': i,
            'max_weight': round(random.uniform(*drone_type["weight_range"]), 1),
            'battery': random.randint(*drone_type["battery_range"]),
            'speed': round(random.uniform(*drone_type["speed_range"]), 1),
            'start_pos': [start_x, start_y]
        }
        drones.append(drone_data)

    # Teslimat noktaları - çeşitli öncelikler
    delivery_points = []
    priorities = [5] * 3 + [4] * 4 + [3] * 8 + [2] * 4 + [1] * 1  # Dengeli dağılım

    for i in range(num_deliveries):
        priority = priorities[i % len(priorities)]

        # Pozisyon stratejisi
        if random.random() < 0.3:  # Merkeze yakın
            radius = random.uniform(50, 200)
        else:  # Daha uzak
            radius = random.uniform(200, 400)

        angle = random.uniform(0, 2 * 3.14159)
        x = int(depot_position[0] + radius * np.cos(angle))
        y = int(depot_position[1] + radius * np.sin(angle))

        # Sınır kontrolü
        x = max(50, min(x, map_size[0] - 50))
        y = max(50, min(y, map_size[1] - 50))

        # Ağırlık önceliğe göre
        if priority >= 4:
            weight = random.uniform(0.5, 5.0)  # Öncelikli teslimatlar daha hafif
        else:
            weight = random.uniform(2.0, 10.0)  # Normal teslimatlar daha ağır

        # Zaman penceresi
        start_hour = random.randint(9, 16)
        duration = random.randint(1, 4)
        end_hour = min(start_hour + duration, 18)

        delivery_data = {
            'id': i,
            'pos': [x, y],
            'weight': round(weight, 1),
            'priority': priority,
            'time_window': [f"{start_hour:02d}:00", f"{end_hour:02d}:00"]
        }
        delivery_points.append(delivery_data)

    # No-fly zones - çeşitli şekiller
    no_fly_zones = []
    for i in range(num_nfz):
        # Merkez nokta
        center_x = random.randint(150, map_size[0] - 150)
        center_y = random.randint(150, map_size[1] - 150)
        size = random.randint(60, 120)

        # Şekil türü
        shapes = ["rectangle", "circle", "triangle"]
        shape = random.choice(shapes)

        if shape == "rectangle":
            w, h = size, size + random.randint(-20, 20)
            coordinates = [
                [center_x - w // 2, center_y - h // 2],
                [center_x + w // 2, center_y - h // 2],
                [center_x + w // 2, center_y + h // 2],
                [center_x - w // 2, center_y + h // 2]
            ]
        elif shape == "circle":
            # Çokgen yaklaşımı
            coordinates = []
            sides = 8
            for j in range(sides):
                angle = (2 * 3.14159 * j) / sides
                x = int(center_x + size * np.cos(angle))
                y = int(center_y + size * np.sin(angle))
                coordinates.append([x, y])
        else:  # triangle
            coordinates = []
            for j in range(3):
                angle = (2 * 3.14159 * j) / 3
                x = int(center_x + size * np.cos(angle))
                y = int(center_y + size * np.sin(angle))
                coordinates.append([x, y])

        nfz_data = {
            'id': i,
            'coordinates': coordinates,
            'active_time': ["09:00", "17:00"]
        }
        no_fly_zones.append(nfz_data)

    # Metadata
    metadata = {
        'generation_time': time.strftime('%Y-%m-%d %H:%M:%S'),
        'seed': current_time,
        'scenario_name': f"fallback_scenario_{current_time}",
        'complexity_level': "Orta"
    }

    return {
        'drones': drones,
        'delivery_points': delivery_points,
        'no_fly_zones': no_fly_zones,
        'depot': list(depot_position),
        'map_size': list(map_size),
        'metadata': metadata
    }


def generate_dynamic_scenarios():
    """Her çalıştırmada farklı senaryolar üret"""
    print("🎲 Dinamik senaryolar oluşturuluyor...")

    # Zaman bazlı unique ID
    timestamp = int(time.time())

    scenarios = [
        {
            'name': f'daily_scenario_{timestamp}',
            'drones': 5,
            'deliveries': 20,
            'nfz': 3,
            'description': 'Günlük dinamik senaryo'
        },
        {
            'name': f'challenge_scenario_{timestamp}',
            'drones': 8,
            'deliveries': 35,
            'nfz': 4,
            'description': 'Zorlu dinamik senaryo'
        },
        {
            'name': f'performance_scenario_{timestamp}',
            'drones': 10,
            'deliveries': 50,
            'nfz': 5,
            'description': 'Performans testi senaryosu'
        }
    ]

    generated_files = []
    project_root = get_project_root()

    for scenario in scenarios:
        print(f"\n📊 {scenario['description']} oluşturuluyor...")

        try:
            if DYNAMIC_GENERATOR_AVAILABLE:
                # Dinamik generator kullan
                generator = DynamicDroneDataGenerator(
                    num_drones=scenario['drones'],
                    num_delivery_points=scenario['deliveries'],
                    num_no_fly_zones=scenario['nfz'],
                    scenario_name=scenario['name'],
                    use_time_seed=True
                )

                filename = os.path.join(project_root, f"data/{scenario['name']}.json")
                generator.save_to_file(filename)

            else:
                # Fallback generator kullan
                data = create_fallback_dynamic_data(
                    scenario['drones'],
                    scenario['deliveries'],
                    scenario['nfz']
                )

                filename = os.path.join(project_root, f"data/{scenario['name']}.json")
                os.makedirs(os.path.dirname(filename), exist_ok=True)

                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)

            generated_files.append(f"data/{scenario['name']}.json")
            print(f"✅ {scenario['description']} kaydedildi")

        except Exception as e:
            print(f"❌ {scenario['description']} oluşturulamadı: {e}")
            continue

    return generated_files


def run_a_star(drones, delivery_points, no_fly_zones, depot_position):
    """A* algoritmasını çalıştırır"""
    print("A* algoritması başlatılıyor...")
    start_time = time.time()

    try:
        # A* algoritması
        pathfinder = AStarPathfinder(delivery_points, no_fly_zones, depot_position)

        # Her drone için rota bul
        routes = {}
        total_cost = 0

        # Teslimatları dronelar arasında basitçe böl
        deliveries_per_drone = len(delivery_points) // len(drones)

        for i, drone in enumerate(drones):
            # Bu drone'un teslimat noktaları
            start_idx = i * deliveries_per_drone
            end_idx = (i + 1) * deliveries_per_drone if i < len(drones) - 1 else len(delivery_points)

            # Teslimat ID'leri
            delivery_ids = [dp.id + 1 for dp in delivery_points[start_idx:end_idx]]

            if not delivery_ids:
                routes[i] = [depot_position]
                continue

            # Rota bul
            route_ids, cost = pathfinder.find_optimal_route(0, delivery_ids)

            # ID'leri koordinatlara dönüştür
            route = [pathfinder.point_id_map[id] for id in route_ids]

            # Sonuçları sakla
            routes[i] = route
            total_cost += cost

        end_time = time.time()
        elapsed_time = end_time - start_time

        print(f"A* algoritması tamamlandı!")
        print(f"Çalışma süresi: {elapsed_time:.4f} saniye")
        print(f"Toplam maliyet: {total_cost:.2f}")

        return routes, total_cost, elapsed_time

    except Exception as e:
        print(f"A* algoritması hatası: {e}")
        return {}, float('inf'), 0


def run_csp(drones, delivery_points, no_fly_zones, depot_position):
    """CSP algoritmasını çalıştırır"""
    print("CSP algoritması başlatılıyor...")
    start_time = time.time()

    try:
        # CSP çözücü
        csp_solver = DroneCSP(drones, delivery_points, no_fly_zones, depot_position)

        # CSP çöz
        assignments = csp_solver.solve()

        if assignments is None or not any(assignments.values()):
            print("CSP çözümü bulunamadı!")
            return None, float('inf'), time.time() - start_time

        # Her drone için rota oluştur
        routes = {}
        total_cost = 0

        for drone_id, delivery_ids in assignments.items():
            if not delivery_ids:
                routes[drone_id] = [depot_position]
                continue

            # A* ile rota bul
            pathfinder = AStarPathfinder(delivery_points, no_fly_zones, depot_position)

            # ID'leri 1'den başladığı için +1 ekle
            route_ids, cost = pathfinder.find_optimal_route(0, [dp_id + 1 for dp_id in delivery_ids])

            # ID'leri koordinatlara dönüştür
            route = [pathfinder.point_id_map[id] for id in route_ids]

            # Sonuçları sakla
            routes[drone_id] = route
            total_cost += cost

        end_time = time.time()
        elapsed_time = end_time - start_time

        print(f"CSP algoritması tamamlandı!")
        print(f"Çalışma süresi: {elapsed_time:.4f} saniye")
        print(f"Toplam maliyet: {total_cost:.2f}")

        return routes, total_cost, elapsed_time

    except Exception as e:
        print(f"CSP algoritması hatası: {e}")
        return None, float('inf'), 0


def run_genetic_algorithm(drones, delivery_points, no_fly_zones, depot_position):
    """Genetik algoritma çalıştırır"""
    print("Genetik algoritma başlatılıyor...")
    start_time = time.time()

    try:
        # Genetik algoritma
        ga = GeneticAlgorithm(drones, delivery_points, no_fly_zones, depot_position,
                              population_size=100, generations=50)

        # GA çalıştır
        assignments, fitness = ga.run()

        # Her drone için rota oluştur
        routes = {}
        total_cost = 0

        for drone_id, delivery_ids in assignments.items():
            if not delivery_ids:
                routes[drone_id] = [depot_position]
                continue

            # A* ile rota bul
            pathfinder = AStarPathfinder(delivery_points, no_fly_zones, depot_position)

            # ID'leri 1'den başladığı için +1 ekle
            route_ids, cost = pathfinder.find_optimal_route(0, [dp_id + 1 for dp_id in delivery_ids])

            # ID'leri koordinatlara dönüştür
            route = [pathfinder.point_id_map[id] for id in route_ids]

            # Sonuçları sakla
            routes[drone_id] = route
            total_cost += cost

        end_time = time.time()
        elapsed_time = end_time - start_time

        print(f"Genetik algoritma tamamlandı!")
        print(f"Çalışma süresi: {elapsed_time:.4f} saniye")
        print(f"En iyi fitness: {fitness:.2f}")
        print(f"Toplam maliyet: {total_cost:.2f}")

        return routes, fitness, elapsed_time

    except Exception as e:
        print(f"Genetik algoritma hatası: {e}")
        return {}, 0, 0


def compare_algorithms(drones, delivery_points, no_fly_zones, depot_position):
    """Algoritmaları karşılaştırır"""
    print("\n=== Algoritma Karşılaştırması ===")
    project_root = get_project_root()

    results = {}

    # A* algoritması
    print("\n1. A* Algoritması çalıştırılıyor...")
    a_star_routes, a_star_cost, a_star_time = run_a_star(drones, delivery_points, no_fly_zones, depot_position)
    results['astar'] = {'routes': a_star_routes, 'cost': a_star_cost, 'time': a_star_time}

    # CSP algoritması
    print("\n2. CSP Algoritması çalıştırılıyor...")
    csp_routes, csp_cost, csp_time = run_csp(drones, delivery_points, no_fly_zones, depot_position)
    results['csp'] = {'routes': csp_routes, 'cost': csp_cost, 'time': csp_time}

    # Genetik algoritma
    print("\n3. Genetik Algoritma çalıştırılıyor...")
    ga_routes, ga_fitness, ga_time = run_genetic_algorithm(drones, delivery_points, no_fly_zones, depot_position)
    results['ga'] = {'routes': ga_routes, 'fitness': ga_fitness, 'time': ga_time}

    # Sonuçları karşılaştır
    print("\n--- Karşılaştırma Sonuçları ---")
    print(f"A* Algoritması: Süre = {a_star_time:.4f}s, Maliyet = {a_star_cost:.2f}")
    if csp_routes:
        print(f"CSP Algoritması: Süre = {csp_time:.4f}s, Maliyet = {csp_cost:.2f}")
    else:
        print("CSP Algoritması: Çözüm bulunamadı")
    print(f"Genetik Algoritma: Süre = {ga_time:.4f}s, Fitness = {ga_fitness:.2f}")

    # Görselleştirme
    try:
        visualizer = DroneVisualizer(drones, delivery_points, no_fly_zones, depot_position)

        # Rotaları görselleştir - timestamp ile unique isimler
        timestamp = int(time.time())

        if a_star_routes:
            print("\nA* rotaları görselleştiriliyor...")
            vis_path = os.path.join(project_root, "visualizations", f"a_star_routes_{timestamp}.png")
            visualizer.plot_scenario(a_star_routes, show=False, save_path=vis_path)

        if csp_routes:
            print("CSP rotaları görselleştiriliyor...")
            vis_path = os.path.join(project_root, "visualizations", f"csp_routes_{timestamp}.png")
            visualizer.plot_scenario(csp_routes, show=False, save_path=vis_path)

        if ga_routes:
            print("GA rotaları görselleştiriliyor...")
            vis_path = os.path.join(project_root, "visualizations", f"ga_routes_{timestamp}.png")
            visualizer.plot_scenario(ga_routes, show=False, save_path=vis_path)

        # Performans grafikleri
        print("\nPerformans grafikleri oluşturuluyor...")
        plt.figure(figsize=(10, 6))

        algorithms = ["A*", "CSP", "GA"]
        times = [a_star_time, csp_time if csp_routes else 0, ga_time]

        plt.bar(algorithms, times, color=['blue', 'green', 'red'])
        plt.title("Algoritmaların Çalışma Süreleri")
        plt.xlabel("Algoritma")
        plt.ylabel("Süre (saniye)")
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        perf_path = os.path.join(project_root, "visualizations", f"performance_comparison_{timestamp}.png")
        plt.savefig(perf_path)
        plt.close()

        # En iyi rotaları animasyona dönüştür
        best_routes = ga_routes if ga_routes else a_star_routes

        if best_routes:
            print("\nEn iyi rotaların animasyonu oluşturuluyor...")
            anim_path = os.path.join(project_root, "visualizations", f"drone_animation_{timestamp}.png")
            visualizer.animate_routes(best_routes, output_path=anim_path)

    except Exception as e:
        print(f"Görselleştirme hatası: {e}")

    print("\nKarşılaştırma tamamlandı. Sonuçlar visualizations/ dizininde kaydedildi.")
    return results


def analyze_performance(data_file, output_file):
    """Performans analizi yapar"""
    print("\n=== Performans Analizi ===")
    project_root = get_project_root()

    try:
        # Veriyi yükle
        drones, delivery_points, no_fly_zones, depot_position = load_data(data_file)

        # Farklı boyutlardaki problemler için performans testi
        delivery_counts = [10, 20, 30, 40, 50]
        a_star_times = []
        csp_times = []
        ga_times = []

        for count in delivery_counts:
            print(f"\nTest: {count} teslimat noktası")

            # İlk 'count' kadar teslimat noktasını al
            subset_deliveries = delivery_points[:min(count, len(delivery_points))]

            # A* algoritması
            start_time = time.time()
            try:
                pathfinder = AStarPathfinder(subset_deliveries, no_fly_zones, depot_position)
                delivery_ids = [dp.id + 1 for dp in subset_deliveries]
                pathfinder.find_optimal_route(0, delivery_ids)
                a_star_time = time.time() - start_time
            except:
                a_star_time = 0
            a_star_times.append(a_star_time)
            print(f"A* süresi: {a_star_time:.4f}s")

            # CSP algoritması
            start_time = time.time()
            try:
                csp_solver = DroneCSP(drones, subset_deliveries, no_fly_zones, depot_position)
                csp_solver.solve()
                csp_time = time.time() - start_time
            except:
                csp_time = 0
            csp_times.append(csp_time)
            print(f"CSP süresi: {csp_time:.4f}s")

            # Genetik algoritma
            start_time = time.time()
            try:
                ga = GeneticAlgorithm(drones, subset_deliveries, no_fly_zones, depot_position,
                                      population_size=50, generations=20)
                ga.run()
                ga_time = time.time() - start_time
            except:
                ga_time = 0
            ga_times.append(ga_time)
            print(f"GA süresi: {ga_time:.4f}s")

        # Performans grafikleri
        plt.figure(figsize=(12, 8))

        plt.plot(delivery_counts, a_star_times, 'b-', marker='o', label="A*")
        plt.plot(delivery_counts, csp_times, 'g-', marker='s', label="CSP")
        plt.plot(delivery_counts, ga_times, 'r-', marker='^', label="GA")

        plt.title("Algoritmalar için Çalışma Süresi Karşılaştırması")
        plt.xlabel("Teslimat Noktası Sayısı")
        plt.ylabel("Çalışma Süresi (saniye)")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()

        output_path = os.path.join(project_root, output_file)
        plt.savefig(output_path)
        plt.close()

        print(f"\nPerformans analizi tamamlandı. Sonuçlar {output_path} dosyasına kaydedildi.")

    except Exception as e:
        print(f"Performans analizi hatası: {e}")


def main():
    """Ana fonksiyon - dinamik senaryo desteği ile"""
    print("=== Drone Filo Optimizasyonu (Dinamik Sürüm) ===")

    # Proje kök dizinini ayarla
    project_root = get_project_root()
    print(f"Proje kökü: {project_root}")

    # Gerekli dizinleri oluştur
    ensure_directories()

    parser = argparse.ArgumentParser(description="Drone Filo Optimizasyonu")
    parser.add_argument('--data', type=str, default="data/drone_data.json",
                        help="Veri dosyası (JSON)")
    parser.add_argument('--generate', action='store_true',
                        help="Yeni veri üret")
    parser.add_argument('--generate-dynamic', action='store_true',
                        help="Dinamik farklı senaryolar üret")
    parser.add_argument('--scenario', type=str, choices=['scenario1', 'scenario2', 'random'],
                        default='scenario1', help="PDF Senaryo seçimi")
    parser.add_argument('--algorithm', type=str, choices=['astar', 'csp', 'ga', 'all'],
                        default='all', help="Çalıştırılacak algoritma")
    parser.add_argument('--analyze', action='store_true',
                        help="Performans analizi yap")
    parser.add_argument('--batch-test', action='store_true',
                        help="Birden fazla senaryo ile test")

    args = parser.parse_args()

    # Dinamik senaryo üretimi
    if args.generate_dynamic:
        print("🎲 Dinamik senaryolar üretiliyor...")
        generated_files = generate_dynamic_scenarios()

        print(f"\n✅ {len(generated_files)} farklı senaryo oluşturuldu:")
        for i, file in enumerate(generated_files, 1):
            print(f"  {i}. {file}")

        print(f"\n📋 Kullanım örnekleri:")
        for file in generated_files:
            print(f"  python main.py --data {file} --algorithm all")

        return

    # Batch test
    if args.batch_test:
        print("🧪 Batch test modu - Birden fazla senaryo test ediliyor...")

        # Dinamik senaryolar oluştur
        scenario_files = generate_dynamic_scenarios()

        batch_results = []

        for i, scenario_file in enumerate(scenario_files, 1):
            print(f"\n{'=' * 50}")
            print(f"📊 Senaryo {i}/{len(scenario_files)}: {os.path.basename(scenario_file)}")
            print(f"{'=' * 50}")

            try:
                # Veriyi yükle
                drones, delivery_points, no_fly_zones, depot_position = load_data(scenario_file)
                print(f"Yüklenen: {len(drones)} drone, {len(delivery_points)} teslimat, {len(no_fly_zones)} NFZ")

                # Algoritmaları çalıştır
                scenario_results = {}

                # A* Test
                print(f"\n🔍 A* algoritması test ediliyor...")
                a_star_routes, a_star_cost, a_star_time = run_a_star(drones, delivery_points, no_fly_zones,
                                                                     depot_position)
                scenario_results['astar'] = {'time': a_star_time, 'cost': a_star_cost}

                # CSP Test
                print(f"\n🧩 CSP algoritması test ediliyor...")
                csp_routes, csp_cost, csp_time = run_csp(drones, delivery_points, no_fly_zones, depot_position)
                scenario_results['csp'] = {'time': csp_time, 'cost': csp_cost}

                # GA Test
                print(f"\n🧬 Genetik algoritma test ediliyor...")
                ga_routes, ga_fitness, ga_time = run_genetic_algorithm(drones, delivery_points, no_fly_zones,
                                                                       depot_position)
                scenario_results['ga'] = {'time': ga_time, 'fitness': ga_fitness}

                # Sonuçları kaydet
                batch_results.append({
                    'scenario': os.path.basename(scenario_file),
                    'num_drones': len(drones),
                    'num_deliveries': len(delivery_points),
                    'results': scenario_results
                })

                print(f"\n📈 Senaryo {i} Sonuçları:")
                print(f"  A*:  {a_star_time:.3f}s, Maliyet: {a_star_cost:.1f}")
                print(f"  CSP: {csp_time:.3f}s, Maliyet: {csp_cost:.1f}")
                print(f"  GA:  {ga_time:.3f}s, Fitness: {ga_fitness:.1f}")

            except Exception as e:
                print(f"❌ Senaryo {i} hatası: {e}")
                continue

        # Batch sonuçlarını özetle
        print(f"\n{'=' * 60}")
        print(f"📊 BATCH TEST SONUÇLARI ({len(batch_results)} senaryo)")
        print(f"{'=' * 60}")

        for result in batch_results:
            print(f"\n📋 {result['scenario']}")
            print(f"   Problem: {result['num_drones']}D-{result['num_deliveries']}T")
            for alg, metrics in result['results'].items():
                if alg == 'ga':
                    print(f"   {alg.upper():4}: {metrics['time']:6.3f}s, Fitness: {metrics['fitness']:8.1f}")
                else:
                    print(f"   {alg.upper():4}: {metrics['time']:6.3f}s, Maliyet:  {metrics['cost']:8.1f}")

        # Ortalama performansları hesapla
        avg_times = {}
        for alg in ['astar', 'csp', 'ga']:
            times = [r['results'][alg]['time'] for r in batch_results if alg in r['results']]
            avg_times[alg] = sum(times) / len(times) if times else 0

        print(f"\n📈 Ortalama Performanslar:")
        for alg, avg_time in avg_times.items():
            print(f"   {alg.upper():4}: {avg_time:.3f}s ortalama")

        # En iyi performansı belirle
        if avg_times:
            fastest_alg = min(avg_times, key=avg_times.get)
            print(f"\n🏆 En Hızlı: {fastest_alg.upper()} ({avg_times[fastest_alg]:.3f}s)")

        print(f"\n✅ Batch test tamamlandı!")
        return

    # Normal senaryo üretimi
    data_path = os.path.join(project_root, args.data)

    if not os.path.exists(data_path) or args.generate:
        print("📊 Dinamik veri oluşturuluyor...")

        try:
            # Senaryo boyutuna göre parametreler
            scenario_params = {
                'scenario1': {'drones': 5, 'deliveries': 20, 'nfz': 2},
                'scenario2': {'drones': 10, 'deliveries': 50, 'nfz': 5},
                'random': {
                    'drones': random.randint(3, 12),
                    'deliveries': random.randint(15, 60),
                    'nfz': random.randint(2, 6)
                }
            }

            params = scenario_params[args.scenario]
            print(
                f"🎯 {args.scenario.title()} senaryo: {params['drones']} drone, {params['deliveries']} teslimat, {params['nfz']} NFZ")

            if DYNAMIC_GENERATOR_AVAILABLE:
                # Dinamik veri üretici
                generator = DynamicDroneDataGenerator(
                    num_drones=params['drones'],
                    num_delivery_points=params['deliveries'],
                    num_no_fly_zones=params['nfz'],
                    scenario_name=f"{args.scenario}_scenario",
                    use_time_seed=True  # Her seferinde farklı
                )
                generator.save_to_file(data_path)
            else:
                # Fallback generator
                data = create_fallback_dynamic_data(
                    params['drones'],
                    params['deliveries'],
                    params['nfz']
                )

                os.makedirs(os.path.dirname(data_path), exist_ok=True)
                with open(data_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)

        except Exception as e:
            print(f"❌ Dinamik veri oluşturma hatası: {e}")
            print("🔄 Fallback: Standart veri oluşturuluyor...")

            # Son çare: standart generator
            try:
                generator = DroneDataGenerator()
                generator.save_to_file(data_path)
            except Exception as e2:
                print(f"❌ Standart veri oluşturma da başarısız: {e2}")
                return

    # Veriyi yükle
    try:
        drones, delivery_points, no_fly_zones, depot_position = load_data(args.data)
        print(f"\n📋 Yüklenen veri: {len(drones)} drone, {len(delivery_points)} teslimat, {len(no_fly_zones)} NFZ")

        # Veri çeşitliliğini göster (varsa)
        try:
            if hasattr(delivery_points[0], 'type'):
                delivery_types = {}
                for dp in delivery_points:
                    dp_type = getattr(dp, 'type', 'Unknown')
                    delivery_types[dp_type] = delivery_types.get(dp_type, 0) + 1
                print(f"📦 Teslimat tipleri: {delivery_types}")

            if hasattr(drones[0], 'type'):
                drone_types = {}
                for drone in drones:
                    drone_type = getattr(drone, 'type', 'Unknown')
                    drone_types[drone_type] = drone_types.get(drone_type, 0) + 1
                print(f"🚁 Drone tipleri: {drone_types}")
        except:
            pass  # Type bilgileri yoksa atla

    except Exception as e:
        print(f"❌ Veri yükleme hatası: {e}")
        print("💡 Çözüm: --generate parametresi ile yeni veri oluşturun")
        return

    # Algoritmaları çalıştır
    print(f"\n🚀 Algoritmalar çalıştırılıyor...")

    if args.algorithm == 'astar' or args.algorithm == 'all':
        try:
            print(f"\n🔍 A* algoritması...")
            routes, cost, elapsed_time = run_a_star(drones, delivery_points, no_fly_zones, depot_position)
            if routes:
                visualizer = DroneVisualizer(drones, delivery_points, no_fly_zones, depot_position)
                vis_path = os.path.join(project_root, "visualizations", f"a_star_routes_{int(time.time())}.png")
                visualizer.plot_scenario(routes, save_path=vis_path, show=False)
        except Exception as e:
            print(f"❌ A* algoritması hatası: {e}")

    if args.algorithm == 'csp' or args.algorithm == 'all':
        try:
            print(f"\n🧩 CSP algoritması...")
            routes, cost, elapsed_time = run_csp(drones, delivery_points, no_fly_zones, depot_position)
            if routes:
                visualizer = DroneVisualizer(drones, delivery_points, no_fly_zones, depot_position)
                vis_path = os.path.join(project_root, "visualizations", f"csp_routes_{int(time.time())}.png")
                visualizer.plot_scenario(routes, save_path=vis_path, show=False)
        except Exception as e:
            print(f"❌ CSP algoritması hatası: {e}")

    if args.algorithm == 'ga' or args.algorithm == 'all':
        try:
            print(f"\n🧬 Genetik algoritma...")
            routes, fitness, elapsed_time = run_genetic_algorithm(drones, delivery_points, no_fly_zones, depot_position)
            if routes:
                visualizer = DroneVisualizer(drones, delivery_points, no_fly_zones, depot_position)
                vis_path = os.path.join(project_root, "visualizations", f"ga_routes_{int(time.time())}.png")
                visualizer.plot_scenario(routes, save_path=vis_path, show=False)

                # Animasyon
                anim_path = os.path.join(project_root, "visualizations", f"drone_animation_{int(time.time())}.png")
                visualizer.animate_routes(routes, output_path=anim_path)
        except Exception as e:
            print(f"❌ Genetik algoritma hatası: {e}")

    # Algoritmaların karşılaştırması
    if args.algorithm == 'all':
        try:
            print(f"\n📊 Algoritma karşılaştırması...")
            compare_algorithms(drones, delivery_points, no_fly_zones, depot_position)
        except Exception as e:
            print(f"❌ Karşılaştırma hatası: {e}")

    # Performans analizi
    if args.analyze:
        try:
            print(f"\n📈 Performans analizi...")
            analyze_performance(args.data, f"visualizations/performance_analysis_{int(time.time())}.png")
        except Exception as e:
            print(f"❌ Performans analizi hatası: {e}")

    print(f"\n🎉 İşlem tamamlandı!")
    print(f"📁 Sonuçlar:")
    print(f"   Görselleştirmeler: {os.path.join(project_root, 'visualizations')}")
    print(f"   Veri dosyaları: {os.path.join(project_root, 'data')}")


if __name__ == "__main__":
    main()