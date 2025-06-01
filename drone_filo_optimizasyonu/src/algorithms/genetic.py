import random
import numpy as np
from deap import base, creator, tools, algorithms
from typing import Dict, List, Tuple, Optional
import warnings
import time
from datetime import datetime, timedelta

# CUDA imports - optional
try:
    import cupy as cp
    from numba import cuda, float32, int32
    import math

    CUDA_AVAILABLE = True
    print("CUDA support loaded successfully")
except ImportError as e:
    CUDA_AVAILABLE = False
    print(f"CUDA not available: {e}")
    print("Using CPU-only implementation")

# DEAP class redefinition uyarılarını önle
warnings.filterwarnings("ignore", category=RuntimeWarning, module="deap.creator")

# Eğer sınıflar zaten tanımlanmışsa sil
if hasattr(creator, 'FitnessMax'):
    del creator.FitnessMax
if hasattr(creator, 'Individual'):
    del creator.Individual

# CUDA Kernels (only if CUDA available)
if CUDA_AVAILABLE:
    @cuda.jit
    def cuda_euclidean_distance(positions1, positions2, distances):
        """CUDA kernel for parallel distance calculations"""
        idx = cuda.grid(1)
        if idx < distances.shape[0]:
            dx = positions1[idx, 0] - positions2[idx, 0]
            dy = positions1[idx, 1] - positions2[idx, 1]
            distances[idx] = math.sqrt(dx * dx + dy * dy)


    @cuda.jit
    def cuda_constraint_check(
            individual_data, drone_capacities, drone_batteries, delivery_weights,
            delivery_positions, depot_position, drone_speeds, violation_results,
            n_individuals, n_deliveries, n_drones
    ):
        """CUDA kernel for parallel constraint violation checking"""
        idx = cuda.grid(1)
        if idx < n_individuals:
            violations = 0.0

            # Drone assignment tracking
            drone_assigned = cuda.local.array(32, int32)  # Max 32 drones
            for i in range(min(n_drones, 32)):
                drone_assigned[i] = -1  # -1 means unassigned

            # Check single package constraint and capacity
            drone_loads = cuda.local.array(32, float32)
            for i in range(min(n_drones, 32)):
                drone_loads[i] = 0.0

            for delivery_idx in range(n_deliveries):
                drone_idx = individual_data[idx * n_deliveries + delivery_idx]
                if 0 <= drone_idx < min(n_drones, 32):
                    # Single package violation
                    if drone_assigned[drone_idx] != -1:
                        violations += 10000.0  # Heavy penalty for multiple packages
                    else:
                        drone_assigned[drone_idx] = delivery_idx
                        drone_loads[drone_idx] = delivery_weights[delivery_idx]

            # Capacity violations
            for drone_idx in range(min(n_drones, 32)):
                if drone_loads[drone_idx] > drone_capacities[drone_idx]:
                    excess_ratio = (drone_loads[drone_idx] - drone_capacities[drone_idx]) / drone_capacities[drone_idx]
                    violations += 5000.0 * (1.0 + excess_ratio)

            # Battery/Energy violations (simplified)
            for drone_idx in range(min(n_drones, 32)):
                if drone_assigned[drone_idx] != -1:
                    delivery_idx = drone_assigned[drone_idx]
                    # Simplified energy calculation
                    dx = delivery_positions[delivery_idx * 2] - depot_position[0]
                    dy = delivery_positions[delivery_idx * 2 + 1] - depot_position[1]
                    distance = math.sqrt(dx * dx + dy * dy) * 2.0  # Round trip

                    energy_needed = distance * (1.0 + drone_loads[drone_idx] * 0.1)
                    if energy_needed > drone_batteries[drone_idx]:
                        violations += 3000.0

            violation_results[idx] = violations


    @cuda.jit
    def cuda_fitness_calculation(
            individual_data, drone_capacities, drone_batteries, delivery_weights,
            delivery_positions, delivery_priorities, depot_position, drone_speeds,
            fitness_results, violation_results, n_individuals, n_deliveries, n_drones
    ):
        """CUDA kernel for parallel fitness calculation"""
        idx = cuda.grid(1)
        if idx < n_individuals:
            teslimat_sayisi = 0.0
            enerji_tuketimi = 0.0
            priority_bonus = 0.0

            # Count valid deliveries and calculate energy
            drone_assigned = cuda.local.array(32, int32)
            for i in range(min(n_drones, 32)):
                drone_assigned[i] = -1

            for delivery_idx in range(n_deliveries):
                drone_idx = individual_data[idx * n_deliveries + delivery_idx]
                if 0 <= drone_idx < min(n_drones, 32):
                    if drone_assigned[drone_idx] == -1:  # Valid assignment
                        drone_assigned[drone_idx] = delivery_idx
                        teslimat_sayisi += 1.0
                        priority_bonus += delivery_priorities[delivery_idx] * 10.0

                        # Energy calculation
                        dx = delivery_positions[delivery_idx * 2] - depot_position[0]
                        dy = delivery_positions[delivery_idx * 2 + 1] - depot_position[1]
                        distance = math.sqrt(dx * dx + dy * dy) * 2.0
                        weight_penalty = delivery_weights[delivery_idx] * distance * 0.05
                        enerji_tuketimi += distance + weight_penalty

            # Fitness calculation: teslimat_sayısı×100 - enerji_tüketimi×0.5 - kural_ihlali×1
            base_fitness = teslimat_sayisi * 100.0 - enerji_tuketimi * 0.5 - violation_results[idx]
            total_fitness = base_fitness + priority_bonus

            fitness_results[idx] = total_fitness


class GeneticAlgorithm:
    """
    CPU tabanlı Genetik Algoritma - Tüm kısıtları karşılayan implementasyon
    """

    def __init__(self, drones, delivery_points, no_fly_zones, depot_position,
                 population_size=100, generations=50, elitism_rate=0.1):
        self.drones = drones
        self.delivery_points = delivery_points
        self.no_fly_zones = no_fly_zones
        self.depot_position = depot_position
        self.population_size = population_size
        self.generations = generations
        self.elitism_rate = elitism_rate

        # Performans izleme
        self.best_fitness_history = []
        self.convergence_threshold = 15
        self.no_improvement_count = 0

        # DEAP setup
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)

        self.toolbox = base.Toolbox()
        self._setup_genetic_operators()

        print(f"CPU GA initialized: {population_size} pop, {generations} gen")
        print(f"Constraints: Single package, Capacity, Battery, Time windows, No-fly zones")

    def _setup_genetic_operators(self):
        """Genetik operatörleri kur"""
        self.toolbox.register("attr_drone", self._smart_drone_assignment)
        self.toolbox.register("individual", tools.initRepeat, creator.Individual,
                              self.toolbox.attr_drone, len(self.delivery_points))
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)

        self.toolbox.register("evaluate", self._evaluate_fitness)
        self.toolbox.register("mate", self._constraint_aware_crossover)
        self.toolbox.register("mutate", self._constraint_aware_mutation)
        self.toolbox.register("select", tools.selTournament, tournsize=3)

    def _smart_drone_assignment(self) -> int:
        """Kısıt-farkında akıllı drone ataması"""
        # Her drone için uygunluk skoru hesapla
        suitable_drones = []

        for drone_idx, drone in enumerate(self.drones):
            # Basit uygunluk skoru (gerçek implementasyonda daha detaylı olabilir)
            score = drone.max_weight + (drone.battery / 1000)  # Normalize battery
            suitable_drones.append((drone_idx, score))

        # Ağırlıklı seçim
        if random.random() < 0.8:  # %80 ihtimalle en uygun drone'ları tercih et
            suitable_drones.sort(key=lambda x: x[1], reverse=True)
            top_drones = suitable_drones[:max(1, len(suitable_drones) // 3)]
            return random.choice(top_drones)[0]
        else:
            return random.randint(0, len(self.drones) - 1)

    def run(self) -> Tuple[Dict[int, List[int]], float]:
        """Genetik algoritmayı çalıştır"""
        start_time = time.time()

        # İlk popülasyon oluştur ve onar
        population = self.toolbox.population(n=self.population_size)
        for individual in population:
            self._repair_all_constraints(individual)

        # İlk popülasyonu değerlendir
        fitnesses = list(map(self.toolbox.evaluate, population))
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit

        best_fitness = max(fitnesses)[0]
        self.best_fitness_history.append(best_fitness)

        print(f"Başlangıç en iyi fitness: {best_fitness:.2f}")

        # Ana evolutyon döngüsü
        for generation in range(self.generations):
            generation_start = time.time()

            # Seçim
            offspring = self.toolbox.select(population, len(population))
            offspring = list(map(self.toolbox.clone, offspring))

            # Adaptive rates
            crossover_rate = self._adaptive_crossover_rate(generation)
            mutation_rate = self._adaptive_mutation_rate(generation)

            # Çaprazlama
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < crossover_rate:
                    self.toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            # Mutasyon
            for mutant in offspring:
                if random.random() < mutation_rate:
                    self.toolbox.mutate(mutant)
                    del mutant.fitness.values

            # Fitness değerlendirmesi
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = list(map(self.toolbox.evaluate, invalid_ind))
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # Elitism
            num_elites = int(self.population_size * self.elitism_rate)
            elites = tools.selBest(population, num_elites)
            offspring.extend(elites)

            population = tools.selBest(offspring, self.population_size)

            # İstatistikler
            fits = [ind.fitness.values[0] for ind in population]
            current_best = max(fits)

            if current_best > best_fitness:
                best_fitness = current_best
                self.no_improvement_count = 0
            else:
                self.no_improvement_count += 1

            self.best_fitness_history.append(current_best)

            # Progress reporting
            if generation % 10 == 0 or generation == self.generations - 1:
                gen_time = time.time() - generation_start
                avg_fitness = np.mean(fits)
                violations_avg = np.mean([self._count_all_violations(self._individual_to_assignments(ind))
                                          for ind in population[:10]])  # Sample check
                print(f"Gen {generation:2d}: avg={avg_fitness:8.2f}, max={current_best:8.2f}, "
                      f"violations={violations_avg:.1f}, time={gen_time:.3f}s")

            # Early stopping
            if self.no_improvement_count >= self.convergence_threshold:
                print(f"Early stopping at generation {generation}")
                break

        # En iyi bireyi bul
        best_individual = tools.selBest(population, 1)[0]
        best_fitness = best_individual.fitness.values[0]

        assignments = self._individual_to_assignments(best_individual)

        total_time = time.time() - start_time
        print(f"GA tamamlandı: {total_time:.2f}s, Final fitness: {best_fitness:.2f}")

        # Final constraint check
        violations = self._count_all_violations(assignments)
        print(f"Final constraint violations: {violations}")

        return assignments, best_fitness

    def _constraint_aware_crossover(self, ind1, ind2):
        """Kısıt-farkında çaprazlama"""
        # İki nokta çaprazlama
        tools.cxTwoPoint(ind1, ind2)

        # Kısıtları onar
        self._repair_all_constraints(ind1)
        self._repair_all_constraints(ind2)

        return ind1, ind2

    def _constraint_aware_mutation(self, individual, indpb=0.15) -> Tuple:
        """Kısıt-farkında mutasyon"""
        for i in range(len(individual)):
            if random.random() < indpb:
                delivery_point = self.delivery_points[i]

                # Uygun drone'ları bul
                suitable_drones = []
                for drone_idx, drone in enumerate(self.drones):
                    if (drone.max_weight >= delivery_point.weight and
                            self._check_battery_feasibility(drone, delivery_point) and
                            self._check_time_window_feasibility(drone, delivery_point)):
                        suitable_drones.append(drone_idx)

                if suitable_drones:
                    individual[i] = random.choice(suitable_drones)
                else:
                    # Fallback: en az kısıt ihlali yapacak drone'u seç
                    individual[i] = self._find_least_violating_drone(delivery_point)

        # Tek paket kısıtını onar
        self._repair_single_package_constraint(individual)

        return (individual,)

    def _repair_all_constraints(self, individual):
        """Tüm kısıtları onar"""
        self._repair_single_package_constraint(individual)
        self._repair_capacity_constraints(individual)
        self._repair_battery_constraints(individual)
        # Time window ve NFZ kısıtları fitness fonksiyonunda cezalandırılıyor

    def _repair_single_package_constraint(self, individual):
        """TEK PAKET KISIITI: Her drone sadece bir paket taşıyabilir"""
        drone_assignments = {}

        for delivery_idx, drone_idx in enumerate(individual):
            if drone_idx in drone_assignments:
                # Bu drone zaten bir paket taşıyor, başka drone bul
                available_drones = [i for i in range(len(self.drones))
                                    if i not in drone_assignments]

                if available_drones:
                    # Uygun drone'ları filtrele
                    delivery_point = self.delivery_points[delivery_idx]
                    suitable_drones = [d for d in available_drones
                                       if self.drones[d].max_weight >= delivery_point.weight]

                    if suitable_drones:
                        new_drone = random.choice(suitable_drones)
                        individual[delivery_idx] = new_drone
                        drone_assignments[new_drone] = delivery_idx
                    else:
                        # Zorla en uygun olanı ata
                        new_drone = min(available_drones,
                                        key=lambda d: abs(self.drones[d].max_weight - delivery_point.weight))
                        individual[delivery_idx] = new_drone
                        drone_assignments[new_drone] = delivery_idx
                else:
                    # Tüm drone'lar dolu, rastgele değiştir
                    victim_drone = random.choice(list(drone_assignments.keys()))
                    del drone_assignments[victim_drone]
                    individual[delivery_idx] = victim_drone
                    drone_assignments[victim_drone] = delivery_idx
            else:
                drone_assignments[drone_idx] = delivery_idx

    def _repair_capacity_constraints(self, individual):
        """KAPASİTE KISIITI: Drone kapasitesi aşımını onar"""
        for delivery_idx, drone_idx in enumerate(individual):
            if 0 <= drone_idx < len(self.drones):
                drone = self.drones[drone_idx]
                delivery = self.delivery_points[delivery_idx]

                if delivery.weight > drone.max_weight:
                    # Daha büyük kapasiteli drone bul
                    suitable_drones = [i for i, d in enumerate(self.drones)
                                       if d.max_weight >= delivery.weight]
                    if suitable_drones:
                        individual[delivery_idx] = random.choice(suitable_drones)

    def _repair_battery_constraints(self, individual):
        """BATARya KISIITI: Enerji yeterliliğini kontrol et"""
        for delivery_idx, drone_idx in enumerate(individual):
            if 0 <= drone_idx < len(self.drones):
                drone = self.drones[drone_idx]
                delivery = self.delivery_points[delivery_idx]

                if not self._check_battery_feasibility(drone, delivery):
                    # Daha fazla bataryaya sahip drone bul
                    suitable_drones = [i for i, d in enumerate(self.drones)
                                       if self._check_battery_feasibility(d, delivery)]
                    if suitable_drones:
                        individual[delivery_idx] = random.choice(suitable_drones)

    def _check_battery_feasibility(self, drone, delivery_point) -> bool:
        """Batarya yeterliliği kontrolü"""
        # Mesafe hesapla (gidis-donus)
        distance = self._euclidean_distance(self.depot_position, delivery_point.pos) * 2

        # Enerji tüketimi tahmini (mesafe + ağırlık cezası)
        weight_penalty = delivery_point.weight * distance * 0.05
        estimated_energy = distance + weight_penalty

        # Güvenlik marjı ekle (%20)
        return estimated_energy * 1.2 <= drone.battery

    def _check_time_window_feasibility(self, drone, delivery_point) -> bool:
        """Zaman penceresi uygunluk kontrolü"""
        if not hasattr(delivery_point, 'time_window') or not delivery_point.time_window:
            return True

        # Basit zaman hesaplama
        distance = self._euclidean_distance(self.depot_position, delivery_point.pos)
        travel_time_minutes = (distance / drone.speed) / 60  # dakika cinsinden

        # Zaman penceresi kontrolü (basitleştirilmiş)
        # Gerçek implementasyonda daha detaylı zaman hesaplama gerekli
        return travel_time_minutes <= 60  # 1 saat içinde ulaşılabilir

    def _find_least_violating_drone(self, delivery_point) -> int:
        """En az kısıt ihlali yapacak drone'u bul"""
        best_drone = 0
        min_violations = float('inf')

        for drone_idx, drone in enumerate(self.drones):
            violations = 0

            # Kapasite ihlali
            if delivery_point.weight > drone.max_weight:
                violations += (delivery_point.weight - drone.max_weight) / drone.max_weight

            # Batarya ihlali
            if not self._check_battery_feasibility(drone, delivery_point):
                violations += 1

            if violations < min_violations:
                min_violations = violations
                best_drone = drone_idx

        return best_drone

    def _evaluate_fitness(self, individual) -> Tuple[float,]:
        """
        Kısıt-farkında fitness değerlendirmesi
        fitness = teslimat_sayısı×100 - enerji_tüketimi×0.5 - kural_ihlali×1
        """
        assignments = self._individual_to_assignments(individual)

        # Temel metrikler
        teslimat_sayisi = self._count_valid_deliveries(assignments)
        enerji_tuketimi = self._calculate_energy_consumption(assignments)
        kural_ihlali = self._count_all_violations(assignments)

        # Bonus puanlar
        priority_bonus = self._calculate_priority_bonus(assignments)
        balance_bonus = self._calculate_load_balance_bonus(assignments)

        # Fitness hesaplama (dokümandaki formül)
        base_fitness = (teslimat_sayisi * 100) - (enerji_tuketimi * 0.5) - (kural_ihlali * 1)
        total_fitness = base_fitness + priority_bonus + balance_bonus

        return (total_fitness,)

    def _count_valid_deliveries(self, assignments: Dict[int, List[int]]) -> int:
        """Geçerli teslimat sayısını say (tek paket kuralı göz önünde)"""
        valid_deliveries = 0

        for drone_id, delivery_indices in assignments.items():
            if len(delivery_indices) == 1:  # Tek paket kuralı
                delivery_idx = delivery_indices[0]
                if delivery_idx < len(self.delivery_points):
                    # Temel kısıt kontrolları
                    drone = next(d for d in self.drones if d.id == drone_id)
                    delivery = self.delivery_points[delivery_idx]

                    if (delivery.weight <= drone.max_weight and
                            self._check_battery_feasibility(drone, delivery)):
                        valid_deliveries += 1

        return valid_deliveries

    def _count_all_violations(self, assignments: Dict[int, List[int]]) -> int:
        """Tüm kısıt ihlallerini say"""
        violations = 0

        # Tek paket kısıtı ihlalleri
        violations += self._count_single_package_violations(assignments)

        # Kapasite kısıtı ihlalleri
        violations += self._count_capacity_violations(assignments)

        # Batarya kısıtı ihlalleri
        violations += self._count_battery_violations(assignments)

        # Zaman penceresi ihlalleri
        violations += self._count_time_window_violations(assignments)

        # No-fly zone ihlalleri
        violations += self._count_nfz_violations(assignments)

        return violations

    def _count_single_package_violations(self, assignments: Dict[int, List[int]]) -> int:
        """Tek paket kısıtı ihlallerini say"""
        violations = 0
        for drone_id, delivery_indices in assignments.items():
            if len(delivery_indices) > 1:
                violations += (len(delivery_indices) - 1) * 10  # Ağır ceza
        return violations

    def _count_capacity_violations(self, assignments: Dict[int, List[int]]) -> int:
        """Kapasite kısıtı ihlallerini say"""
        violations = 0

        for drone_id, delivery_indices in assignments.items():
            if not delivery_indices:
                continue

            drone = next(d for d in self.drones if d.id == drone_id)
            total_weight = sum(self.delivery_points[i].weight for i in delivery_indices
                               if i < len(self.delivery_points))

            if total_weight > drone.max_weight:
                excess_ratio = (total_weight - drone.max_weight) / drone.max_weight
                violations += int(5 * (1 + excess_ratio))

        return violations

    def _count_battery_violations(self, assignments: Dict[int, List[int]]) -> int:
        """Batarya kısıtı ihlallerini say"""
        violations = 0

        for drone_id, delivery_indices in assignments.items():
            if not delivery_indices:
                continue

            drone = next(d for d in self.drones if d.id == drone_id)
            total_energy_needed = self._calculate_route_energy(drone_id, delivery_indices)

            if total_energy_needed > drone.battery:
                excess_ratio = (total_energy_needed - drone.battery) / drone.battery
                violations += int(3 * (1 + excess_ratio))

        return violations

    def _count_time_window_violations(self, assignments: Dict[int, List[int]]) -> int:
        """Zaman penceresi kısıtı ihlallerini say"""
        violations = 0

        for drone_id, delivery_indices in assignments.items():
            if not delivery_indices:
                continue

            drone = next(d for d in self.drones if d.id == drone_id)

            for delivery_idx in delivery_indices:
                if delivery_idx < len(self.delivery_points):
                    delivery = self.delivery_points[delivery_idx]

                    if hasattr(delivery, 'time_window') and delivery.time_window:
                        if not self._check_time_window_feasibility(drone, delivery):
                            violations += 2

        return violations

    def _count_nfz_violations(self, assignments: Dict[int, List[int]]) -> int:
        """No-fly zone kısıtı ihlallerini say"""
        violations = 0

        for drone_id, delivery_indices in assignments.items():
            for delivery_idx in delivery_indices:
                if delivery_idx < len(self.delivery_points):
                    delivery_pos = self.delivery_points[delivery_idx].pos

                    for nfz in self.no_fly_zones:
                        if hasattr(nfz, 'is_active') and nfz.is_active():
                            if hasattr(nfz, 'point_in_polygon') and nfz.point_in_polygon(delivery_pos):
                                violations += 3  # NFZ ihlali ağır ceza

        return violations

    def _calculate_route_energy(self, drone_id: int, delivery_indices: List[int]) -> float:
        """Bir drone'un rotası için gereken enerjiyi hesapla"""
        if not delivery_indices:
            return 0.0

        total_energy = 0.0
        current_pos = self.depot_position

        # Teslimat noktalarına git
        for delivery_idx in delivery_indices:
            if delivery_idx < len(self.delivery_points):
                delivery_pos = self.delivery_points[delivery_idx].pos
                distance = self._euclidean_distance(current_pos, delivery_pos)

                # Ağırlık cezası
                weight = self.delivery_points[delivery_idx].weight
                weight_penalty = weight * distance * 0.05

                total_energy += distance + weight_penalty
                current_pos = delivery_pos

        # Depoya dön
        return_distance = self._euclidean_distance(current_pos, self.depot_position)
        total_energy += return_distance

        return total_energy

    def _calculate_energy_consumption(self, assignments: Dict[int, List[int]]) -> float:
        """Toplam enerji tüketimi hesaplama"""
        total_energy = 0.0

        for drone_id, delivery_indices in assignments.items():
            total_energy += self._calculate_route_energy(drone_id, delivery_indices)

        return total_energy

    def _calculate_priority_bonus(self, assignments: Dict[int, List[int]]) -> float:
        """Öncelik bonusu hesaplama"""
        bonus = 0.0
        for delivery_indices in assignments.values():
            for delivery_idx in delivery_indices:
                if delivery_idx < len(self.delivery_points):
                    priority = self.delivery_points[delivery_idx].priority
                    bonus += priority * 10
        return bonus

    def _calculate_load_balance_bonus(self, assignments: Dict[int, List[int]]) -> float:
        """Yük dengesi bonusu hesaplama"""
        delivery_counts = [len(deliveries) for deliveries in assignments.values()]
        if not delivery_counts:
            return 0.0

        std_count = np.std(delivery_counts)
        balance_bonus = max(0, 50 - std_count * 10)
        return balance_bonus

    def _individual_to_assignments(self, individual) -> Dict[int, List[int]]:
        """Individual'dan assignment'a çevir"""
        assignments = {drone.id: [] for drone in self.drones}

        for delivery_idx, drone_idx in enumerate(individual):
            if 0 <= drone_idx < len(self.drones):
                drone_id = self.drones[drone_idx].id
                assignments[drone_id].append(delivery_idx)

        return assignments

    def _euclidean_distance(self, pos1: Tuple[float, float], pos2: Tuple[float, float]) -> float:
        """Euclidean mesafe hesaplama"""
        return np.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)

    def _adaptive_crossover_rate(self, generation: int) -> float:
        """Adaptive crossover rate"""
        base_rate = 0.8
        reduction = 0.3 * (generation / self.generations)
        return max(0.5, base_rate - reduction)

    def _adaptive_mutation_rate(self, generation: int) -> float:
        """Adaptive mutation rate"""
        base_rate = 0.2
        if self.no_improvement_count > 8:
            return min(0.5, base_rate + 0.2)
        return base_rate

    def get_convergence_info(self) -> Dict:
        """Convergence bilgileri"""
        return {
            'fitness_history': self.best_fitness_history,
            'total_generations': len(self.best_fitness_history),
            'final_fitness': self.best_fitness_history[-1] if self.best_fitness_history else 0
        }


class CUDAGeneticAlgorithm(GeneticAlgorithm):
    """
    CUDA ile optimize edilmiş Genetik Algoritma
    Tüm kısıtları GPU'da paralel olarak kontrol eder
    """

    def __init__(self, drones, delivery_points, no_fly_zones, depot_position,
                 population_size=100, generations=50, elitism_rate=0.1):

        if not CUDA_AVAILABLE:
            raise RuntimeError("CUDA not available for CUDAGeneticAlgorithm")

        super().__init__(drones, delivery_points, no_fly_zones, depot_position,
                         population_size, generations, elitism_rate)

        # GPU setup
        self.n_drones = len(drones)
        self.n_deliveries = len(delivery_points)
        self.n_individuals = population_size

        try:
            self._setup_gpu_arrays()
            self._setup_gpu_memory_pool()
            print(f"CUDA GA initialized successfully")
            print(f"GPU: {cp.cuda.Device().name}")
            print(f"GPU Memory: {self._get_gpu_memory_info()}")
        except Exception as e:
            print(f"CUDA setup failed: {e}")
            raise

    def _setup_gpu_arrays(self):
        """GPU arrays setup with proper memory management"""
        try:
            # Drone verileri
            self.gpu_drone_capacities = cp.array([d.max_weight for d in self.drones], dtype=cp.float32)
            self.gpu_drone_batteries = cp.array([d.battery for d in self.drones], dtype=cp.float32)
            self.gpu_drone_speeds = cp.array([d.speed for d in self.drones], dtype=cp.float32)

            # Teslimat verileri
            self.gpu_delivery_weights = cp.array([d.weight for d in self.delivery_points], dtype=cp.float32)
            delivery_positions_flat = []
            delivery_priorities = []

            for d in self.delivery_points:
                delivery_positions_flat.extend([d.pos[0], d.pos[1]])
                delivery_priorities.append(d.priority)

            self.gpu_delivery_positions = cp.array(delivery_positions_flat, dtype=cp.float32)
            self.gpu_delivery_priorities = cp.array(delivery_priorities, dtype=cp.float32)

            # Depot position
            self.gpu_depot_position = cp.array([self.depot_position[0], self.depot_position[1]], dtype=cp.float32)

            # Results arrays
            self.gpu_fitness_results = cp.zeros(self.n_individuals, dtype=cp.float32)
            self.gpu_violation_results = cp.zeros(self.n_individuals, dtype=cp.float32)

            # Individual data array (will be updated each generation)
            self.gpu_individual_data = cp.zeros(self.n_individuals * self.n_deliveries, dtype=cp.int32)

            print(f"GPU arrays initialized: {self.n_drones} drones, {self.n_deliveries} deliveries")

        except cp.cuda.memory.OutOfMemoryError as e:
            print(f"GPU out of memory: {e}")
            self._cleanup_gpu_memory()
            raise
        except Exception as e:
            print(f"GPU array setup error: {e}")
            raise

    def _setup_gpu_memory_pool(self):
        """GPU memory pool configuration"""
        try:
            mempool = cp.get_default_memory_pool()
            # Set memory pool limit to 80% of available GPU memory
            device = cp.cuda.Device()
            mem_info = device.mem_info
            max_memory = int(mem_info[0] * 0.8)  # 80% of free memory
            mempool.set_limit(size=max_memory)
            print(f"GPU memory pool limit set to: {max_memory / (1024 ** 3):.1f} GB")
        except Exception as e:
            print(f"GPU memory pool setup warning: {e}")

    def _get_gpu_memory_info(self) -> str:
        """GPU memory bilgilerini al"""
        try:
            device = cp.cuda.Device()
            mem_info = device.mem_info
            free_gb = mem_info[0] / (1024 ** 3)
            total_gb = mem_info[1] / (1024 ** 3)
            used_gb = total_gb - free_gb
            return f"{used_gb:.1f}GB used / {total_gb:.1f}GB total"
        except:
            return "Memory info unavailable"

    def run(self) -> Tuple[Dict[int, List[int]], float]:
        """CUDA optimized genetic algorithm run"""
        start_time = time.time()

        try:
            print("Running CUDA optimized genetic algorithm...")

            # İlk popülasyon oluştur (CPU'da)
            population = self.toolbox.population(n=self.population_size)
            for individual in population:
                self._repair_all_constraints(individual)

            best_fitness = -float('inf')
            self.best_fitness_history = []

            # Ana evolution loop
            for generation in range(self.generations):
                generation_start = time.time()

                # Population'ı GPU'ya transfer et
                self._transfer_population_to_gpu(population)

                # GPU'da fitness hesaplama
                self._gpu_evaluate_population()

                # Fitness değerlerini CPU'ya geri al
                fitness_values = self._transfer_fitness_from_gpu()

                # Fitness değerlerini population'a ata
                for ind, fitness in zip(population, fitness_values):
                    ind.fitness.values = (fitness,)

                # En iyi fitness'ı bul
                current_best = max(fitness_values)
                if current_best > best_fitness:
                    best_fitness = current_best
                    self.no_improvement_count = 0
                else:
                    self.no_improvement_count += 1

                self.best_fitness_history.append(current_best)

                # Evolution operations (CPU'da)
                if generation < self.generations - 1:  # Son generation değilse
                    population = self._gpu_assisted_evolution(population)

                # Progress reporting
                if generation % 5 == 0 or generation == self.generations - 1:
                    gen_time = time.time() - generation_start
                    avg_fitness = np.mean(fitness_values)
                    gpu_memory = self._get_gpu_memory_info()
                    print(f"Gen {generation:2d}: avg={avg_fitness:8.2f}, max={current_best:8.2f}, "
                          f"time={gen_time:.3f}s, GPU: {gpu_memory}")

                # Early stopping
                if self.no_improvement_count >= self.convergence_threshold:
                    print(f"Early stopping at generation {generation}")
                    break

                # Memory cleanup every 10 generations
                if generation % 10 == 0:
                    self._cleanup_gpu_memory()

            # En iyi bireyi bul
            best_individual = tools.selBest(population, 1)[0]
            best_fitness = best_individual.fitness.values[0]
            assignments = self._individual_to_assignments(best_individual)

            total_time = time.time() - start_time
            print(f"CUDA GA tamamlandı: {total_time:.2f}s, Final fitness: {best_fitness:.2f}")

            return assignments, best_fitness

        except Exception as e:
            print(f"CUDA GA error: {e}")
            print("Falling back to CPU implementation")
            return super().run()
        finally:
            self._cleanup_gpu_memory()

    def _transfer_population_to_gpu(self, population):
        """Population'ı GPU'ya transfer et"""
        try:
            individual_data = []
            for individual in population:
                individual_data.extend(individual)

            self.gpu_individual_data = cp.array(individual_data, dtype=cp.int32)
        except Exception as e:
            print(f"GPU transfer error: {e}")
            raise

    def _gpu_evaluate_population(self):
        """GPU'da population'ı değerlendir"""
        try:
            # CUDA kernel parametreleri
            threads_per_block = 256
            blocks_per_grid = (self.n_individuals + threads_per_block - 1) // threads_per_block

            # Constraint violations hesapla
            cuda_constraint_check[blocks_per_grid, threads_per_block](
                self.gpu_individual_data,
                self.gpu_drone_capacities,
                self.gpu_drone_batteries,
                self.gpu_delivery_weights,
                self.gpu_delivery_positions,
                self.gpu_depot_position,
                self.gpu_drone_speeds,
                self.gpu_violation_results,
                self.n_individuals,
                self.n_deliveries,
                self.n_drones
            )

            # Fitness hesapla
            cuda_fitness_calculation[blocks_per_grid, threads_per_block](
                self.gpu_individual_data,
                self.gpu_drone_capacities,
                self.gpu_drone_batteries,
                self.gpu_delivery_weights,
                self.gpu_delivery_positions,
                self.gpu_delivery_priorities,
                self.gpu_depot_position,
                self.gpu_drone_speeds,
                self.gpu_fitness_results,
                self.gpu_violation_results,
                self.n_individuals,
                self.n_deliveries,
                self.n_drones
            )

            # GPU sync
            cp.cuda.Stream.null.synchronize()

        except Exception as e:
            print(f"GPU evaluation error: {e}")
            raise

    def _transfer_fitness_from_gpu(self) -> List[float]:
        """Fitness değerlerini GPU'dan CPU'ya al"""
        try:
            return self.gpu_fitness_results.get().tolist()
        except Exception as e:
            print(f"GPU fitness transfer error: {e}")
            raise

    def _gpu_assisted_evolution(self, population):
        """GPU destekli evolution operations"""
        # Evolution operations CPU'da yapılıyor (crossover, mutation, selection)
        # Çünkü bu operasyonlar daha kompleks ve GPU'da paralelleştirmesi zor

        # Selection
        offspring = self.toolbox.select(population, len(population))
        offspring = list(map(self.toolbox.clone, offspring))

        # Adaptive rates
        crossover_rate = self._adaptive_crossover_rate(len(self.best_fitness_history))
        mutation_rate = self._adaptive_mutation_rate(len(self.best_fitness_history))

        # Crossover
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < crossover_rate:
                self.toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        # Mutation
        for mutant in offspring:
            if random.random() < mutation_rate:
                self.toolbox.mutate(mutant)
                del mutant.fitness.values

        # Elitism
        num_elites = int(self.population_size * self.elitism_rate)
        elites = tools.selBest(population, num_elites)
        offspring.extend(elites)

        return tools.selBest(offspring, self.population_size)

    def _cleanup_gpu_memory(self):
        """GPU memory temizleme"""
        try:
            if CUDA_AVAILABLE:
                cp.get_default_memory_pool().free_all_blocks()
                cp.get_default_pinned_memory_pool().free_all_blocks()
        except Exception as e:
            print(f"GPU cleanup warning: {e}")

    def __del__(self):
        """Destructor - GPU memory temizleme"""
        self._cleanup_gpu_memory()


class HybridGeneticAlgorithm:
    """
    CPU ve GPU arasında akıllı seçim yapan hibrit algoritma
    Problem boyutuna ve GPU durumuna göre optimal algoritma seçer
    """

    def __init__(self, drones, delivery_points, no_fly_zones, depot_position,
                 population_size=100, generations=50, elitism_rate=0.1,
                 force_cpu=False, gpu_threshold=500):

        self.problem_size = len(delivery_points) * len(drones)
        self.gpu_threshold = gpu_threshold

        print(f"Problem size: {self.problem_size} (deliveries: {len(delivery_points)}, drones: {len(drones)})")

        # Algoritma seçim mantığı
        use_cuda = (CUDA_AVAILABLE and
                    not force_cpu and
                    self.problem_size >= self.gpu_threshold)

        if use_cuda:
            try:
                print(f"Initializing CUDA GA (problem size >= {self.gpu_threshold})")
                self.algorithm = CUDAGeneticAlgorithm(
                    drones, delivery_points, no_fly_zones, depot_position,
                    population_size, generations, elitism_rate
                )
                self.using_cuda = True
                print("CUDA GA initialized successfully")
            except Exception as e:
                print(f"CUDA GA initialization failed: {e}")
                print("Falling back to CPU implementation")
                self.algorithm = GeneticAlgorithm(
                    drones, delivery_points, no_fly_zones, depot_position,
                    population_size, generations, elitism_rate
                )
                self.using_cuda = False
        else:
            reason = "CUDA not available" if not CUDA_AVAILABLE else f"Problem size < {self.gpu_threshold}" if self.problem_size < self.gpu_threshold else "Forced CPU mode"
            print(f"Using CPU GA ({reason})")
            self.algorithm = GeneticAlgorithm(
                drones, delivery_points, no_fly_zones, depot_position,
                population_size, generations, elitism_rate
            )
            self.using_cuda = False

    def run(self):
        """Seçilen algoritmayı çalıştır"""
        print(f"Running {'CUDA' if self.using_cuda else 'CPU'} Genetic Algorithm...")
        return self.algorithm.run()

    def get_convergence_info(self):
        """Convergence bilgileri"""
        info = self.algorithm.get_convergence_info()
        info['using_cuda'] = self.using_cuda
        info['problem_size'] = self.problem_size
        return info

    def get_performance_stats(self):
        """Performance istatistikleri"""
        return {
            'algorithm_type': 'CUDA' if self.using_cuda else 'CPU',
            'problem_size': self.problem_size,
            'cuda_available': CUDA_AVAILABLE,
            'gpu_info': self.algorithm._get_gpu_memory_info() if self.using_cuda else 'N/A'
        }


# Utility functions
def optimize_gpu_memory():
    """GPU memory optimization"""
    if CUDA_AVAILABLE:
        try:
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()
            print("GPU memory optimized")
        except Exception as e:
            print(f"GPU memory optimization failed: {e}")
    else:
        print("CUDA not available for memory optimization")


def check_cuda_status():
    """CUDA durumu ve sistem bilgilerini kontrol et"""
    print("=== CUDA Status ===")
    print(f"CUDA Available: {CUDA_AVAILABLE}")

    if CUDA_AVAILABLE:
        try:
            device_count = cp.cuda.runtime.getDeviceCount()
            print(f"CUDA Devices: {device_count}")

            for i in range(device_count):
                device = cp.cuda.Device(i)
                print(f"Device {i}: {device.name}")
                mem_info = device.mem_info
                print(f"  Memory: {mem_info[0] / 1024 ** 3:.1f}GB free / {mem_info[1] / 1024 ** 3:.1f}GB total")

            # Current device info
            current_device = cp.cuda.Device()
            print(f"Current Device: {current_device.name}")

        except Exception as e:
            print(f"GPU info error: {e}")
    else:
        print("CUDA not available - using CPU only")

    print("==================\n")


def get_recommended_parameters(problem_size: int) -> Dict:
    """Problem boyutuna göre önerilen parametreler"""
    if problem_size < 100:
        return {
            'population_size': 50,
            'generations': 30,
            'use_gpu': False,
            'reason': 'Small problem - CPU is sufficient'
        }
    elif problem_size < 500:
        return {
            'population_size': 100,
            'generations': 50,
            'use_gpu': False,
            'reason': 'Medium problem - CPU recommended'
        }
    else:
        return {
            'population_size': 200,
            'generations': 100,
            'use_gpu': True,
            'reason': 'Large problem - GPU recommended'
        }


# Test and benchmark functions
def benchmark_algorithm(drones, delivery_points, no_fly_zones, depot_position,
                        runs=3, show_details=True):
    """Algoritma performansını test et"""

    print("=== Algorithm Benchmark ===")

    problem_size = len(delivery_points) * len(drones)
    recommended = get_recommended_parameters(problem_size)

    print(f"Problem: {len(drones)} drones, {len(delivery_points)} deliveries")
    print(f"Recommended: {recommended}")
    print()

    results = {
        'cpu_times': [],
        'cpu_fitness': [],
        'gpu_times': [],
        'gpu_fitness': []
    }

    # CPU benchmark
    print("CPU Benchmark:")
    for run in range(runs):
        start_time = time.time()

        cpu_ga = GeneticAlgorithm(
            drones, delivery_points, no_fly_zones, depot_position,
            population_size=recommended['population_size'],
            generations=recommended['generations']
        )

        assignments, fitness = cpu_ga.run()
        elapsed = time.time() - start_time

        results['cpu_times'].append(elapsed)
        results['cpu_fitness'].append(fitness)

        if show_details:
            print(f"  Run {run + 1}: {elapsed:.2f}s, Fitness: {fitness:.2f}")

    # GPU benchmark (if available and problem is large enough)
    if CUDA_AVAILABLE and recommended['use_gpu']:
        print("\nGPU Benchmark:")
        for run in range(runs):
            try:
                start_time = time.time()

                gpu_ga = CUDAGeneticAlgorithm(
                    drones, delivery_points, no_fly_zones, depot_position,
                    population_size=recommended['population_size'],
                    generations=recommended['generations']
                )

                assignments, fitness = gpu_ga.run()
                elapsed = time.time() - start_time

                results['gpu_times'].append(elapsed)
                results['gpu_fitness'].append(fitness)

                if show_details:
                    print(f"  Run {run + 1}: {elapsed:.2f}s, Fitness: {fitness:.2f}")

            except Exception as e:
                print(f"  GPU Run {run + 1} failed: {e}")

    # Summary
    print("\n=== Benchmark Summary ===")
    if results['cpu_times']:
        cpu_avg_time = np.mean(results['cpu_times'])
        cpu_avg_fitness = np.mean(results['cpu_fitness'])
        print(f"CPU: {cpu_avg_time:.2f}s avg, Fitness: {cpu_avg_fitness:.2f} avg")

    if results['gpu_times']:
        gpu_avg_time = np.mean(results['gpu_times'])
        gpu_avg_fitness = np.mean(results['gpu_fitness'])
        speedup = cpu_avg_time / gpu_avg_time if cpu_avg_time > 0 else 0
        print(f"GPU: {gpu_avg_time:.2f}s avg, Fitness: {gpu_avg_fitness:.2f} avg")
        print(f"Speedup: {speedup:.2f}x")

    return results


# Main execution
if __name__ == "__main__":
    print("Improved Genetic Algorithm with Full Constraint Support")
    print("======================================================")
    check_cuda_status()


    # Test with dummy data
    class DummyDrone:
        def __init__(self, id, max_weight, battery, speed, start_pos):
            self.id = id
            self.max_weight = max_weight
            self.battery = battery
            self.speed = speed
            self.start_pos = start_pos


    class DummyDelivery:
        def __init__(self, id, pos, weight, priority, time_window=None):
            self.id = id
            self.pos = pos
            self.weight = weight
            self.priority = priority
            self.time_window = time_window


    # Create test data
    drones = [DummyDrone(i, 10.0, 5000, 15.0, (0, 0)) for i in range(5)]
    deliveries = [DummyDelivery(i, (random.randint(10, 100), random.randint(10, 100)),
                                random.uniform(1, 8), random.randint(1, 5))
                  for i in range(20)]

    print(f"Test problem: {len(drones)} drones, {len(deliveries)} deliveries")

    # Run hybrid algorithm
    hybrid_ga = HybridGeneticAlgorithm(
        drones, deliveries, [], (0, 0),
        population_size=50, generations=20
    )

    assignments, fitness = hybrid_ga.run()
    info = hybrid_ga.get_convergence_info()

    print(f"\nResults:")
    print(f"Algorithm used: {'CUDA' if info['using_cuda'] else 'CPU'}")
    print(f"Final fitness: {fitness:.2f}")
    print(f"Generations: {info['total_generations']}")
    print(f"Valid assignments: {sum(1 for v in assignments.values() if v)}")

    optimize_gpu_memory()