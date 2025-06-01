import numpy as np
import time
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import heapq
import math
from datetime import datetime, timedelta
from functools import lru_cache

try:
    from constraint import Problem, AllDifferentConstraint

    CONSTRAINT_LIB_AVAILABLE = True
except ImportError:
    CONSTRAINT_LIB_AVAILABLE = False
    print("Warning: python-constraint library not available, using fallback implementation")


class DroneCSP:
    """
    İyileştirilmiş CSP implementasyonu - Daha esnek kısıtlar ve daha iyi performans

    Ana iyileştirmeler:
    1. Esnek kısıt değerlendirmesi (soft constraints)
    2. Daha iyi heuristic fonksiyonlar
    3. Akıllı backtracking
    4. Performans optimizasyonları
    """

    def __init__(self, drones, delivery_points, no_fly_zones, depot_position, current_time="09:00"):
        self.drones = drones
        self.delivery_points = delivery_points
        self.no_fly_zones = no_fly_zones
        self.depot_position = depot_position
        self.current_time = self._parse_time(current_time)

        # Kısıt ağırlıkları - ayarlanabilir
        self.constraint_weights = {
            'capacity': 1.0,  # Hard constraint
            'battery': 0.9,  # Biraz esnek
            'time_window': 0.7,  # Daha esnek
            'no_fly_zone': 1.0,  # Hard constraint
            'priority': 0.8  # Soft constraint
        }

        # Performans parametreleri
        self.max_iterations = 1000
        self.improvement_threshold = 0.01

        # Ön hesaplamalar
        self._precompute_data()
        self._initialize_csp_variables()

        print(f"İyileştirilmiş CSP initialized:")
        print(f"  - {len(self.drones)} drones")
        print(f"  - {len(self.delivery_points)} deliveries")
        print(f"  - {len(self.no_fly_zones)} no-fly zones")

    def _precompute_data(self):
        """Performans için ön hesaplamalar"""
        # Distance matrix
        self._build_distance_matrix()

        # Drone özellikleri
        self.drone_lookup = {drone.id: drone for drone in self.drones}
        self.drone_capacities = {drone.id: drone.max_weight for drone in self.drones}
        self.drone_batteries = {drone.id: drone.battery for drone in self.drones}
        self.drone_speeds = {drone.id: drone.speed for drone in self.drones}

        # Delivery özellikleri
        self.delivery_lookup = {dp.id: dp for dp in self.delivery_points}
        self.dp_weights = {dp.id: dp.weight for dp in self.delivery_points}
        self.dp_priorities = {dp.id: dp.priority for dp in self.delivery_points}
        self.dp_positions = {dp.id: dp.pos for dp in self.delivery_points}

        # Zaman pencereleri - daha esnek parse
        self.dp_time_windows = {}
        for dp in self.delivery_points:
            if hasattr(dp, 'time_window') and dp.time_window:
                try:
                    start_time = self._parse_time(dp.time_window[0])
                    end_time = self._parse_time(dp.time_window[1])
                    self.dp_time_windows[dp.id] = (start_time, end_time)
                except:
                    # Hata durumunda geniş pencere
                    self.dp_time_windows[dp.id] = (
                        self._parse_time("08:00"),
                        self._parse_time("20:00")
                    )
            else:
                # Default geniş pencere
                self.dp_time_windows[dp.id] = (
                    self._parse_time("08:00"),
                    self._parse_time("20:00")
                )

        # Uyumluluk matrisi
        self._build_compatibility_matrix()

    def _build_distance_matrix(self):
        """Mesafe matrisini oluştur"""
        all_positions = [self.depot_position] + [dp.pos for dp in self.delivery_points]
        n = len(all_positions)
        self.distance_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(i + 1, n):
                dist = self._calculate_distance(all_positions[i], all_positions[j])
                self.distance_matrix[i, j] = dist
                self.distance_matrix[j, i] = dist

    def _build_compatibility_matrix(self):
        """Drone-teslimat uyumluluk matrisi - daha esnek"""
        self.compatibility = {}
        self.compatibility_scores = {}  # Uyumluluk skorları

        for drone in self.drones:
            self.compatibility[drone.id] = set()
            self.compatibility_scores[drone.id] = {}

            for dp in self.delivery_points:
                score = self._calculate_compatibility_score(drone, dp)
                if score > 0.3:  # Minimum uyumluluk eşiği
                    self.compatibility[drone.id].add(dp.id)
                    self.compatibility_scores[drone.id][dp.id] = score

    def _calculate_compatibility_score(self, drone, delivery_point):
        """Drone-teslimat uyumluluk skoru (0-1 arası)"""
        score = 1.0

        # Ağırlık uyumluluğu
        if delivery_point.weight > drone.max_weight:
            return 0.0  # Kesinlikle uyumsuz
        else:
            weight_ratio = delivery_point.weight / drone.max_weight
            score *= (1.0 - weight_ratio * 0.5)  # Ağırlık arttıkça skor düşer

        # Mesafe/batarya uyumluluğu
        distance = self._calculate_distance(drone.start_pos, delivery_point.pos)
        round_trip = distance * 2
        energy_needed = round_trip * 0.1  # Basitleştirilmiş

        if energy_needed > drone.battery:
            return 0.0  # Kesinlikle uyumsuz
        else:
            battery_ratio = energy_needed / drone.battery
            score *= (1.0 - battery_ratio * 0.7)

        # Hız uyumluluğu (hızlı drone'lar acil teslimatlar için)
        if hasattr(delivery_point, 'priority') and delivery_point.priority >= 4:
            speed_factor = drone.speed / 20.0  # 20 m/s referans hız
            score *= min(1.5, speed_factor)

        return score

    def _initialize_csp_variables(self):
        """CSP değişkenlerini initialize et"""
        self.variables = []
        self.domains = {}

        for dp in self.delivery_points:
            var_name = f"delivery_{dp.id}"
            self.variables.append(var_name)

            # Domain: Uyumlu drone'lar (skorla sıralı)
            compatible_drones = []
            for drone in self.drones:
                if dp.id in self.compatibility.get(drone.id, set()):
                    score = self.compatibility_scores[drone.id].get(dp.id, 0)
                    compatible_drones.append((score, drone.id))

            # Skora göre sırala
            compatible_drones.sort(reverse=True)
            self.domains[var_name] = [drone_id for _, drone_id in compatible_drones]

            # Eğer hiç uyumlu drone yoksa, en az yüklü drone'ları ekle
            if not self.domains[var_name]:
                all_drones = list(range(len(self.drones)))
                self.domains[var_name] = all_drones

    def solve(self) -> Optional[Dict[int, List[int]]]:
        """Ana CSP çözücü - geliştirilmiş versiyon"""
        print("\nİyileştirilmiş CSP çözümü başlatılıyor...")
        start_time = time.time()

        num_deliveries = len(self.delivery_points)
        num_drones = len(self.drones)

        print(f"Problem: {num_drones} drone, {num_deliveries} teslimat")

        # Çözüm stratejisi seçimi
        if num_deliveries <= 15:
            solution = self._solve_exact_backtracking()
        elif num_deliveries <= 30:
            solution = self._solve_hybrid_approach()
        else:
            solution = self._solve_large_scale_heuristic()

        solve_time = time.time() - start_time

        if solution:
            total_assigned = sum(len(assignments) for assignments in solution.values())
            success_rate = (total_assigned / num_deliveries) * 100
            print(f"\nCSP çözüm bulundu: {solve_time:.3f}s")
            print(f"Atanan: {total_assigned}/{num_deliveries} (%{success_rate:.1f})")

            # Çözüm kalitesini değerlendir
            quality_score = self._evaluate_solution_quality(solution)
            print(f"Çözüm kalitesi: {quality_score:.2f}/100")

            # Post-processing optimizasyonu
            if success_rate < 100:
                print("\nPost-processing optimizasyonu...")
                solution = self._post_process_solution(solution)

                total_assigned = sum(len(assignments) for assignments in solution.values())
                success_rate = (total_assigned / num_deliveries) * 100
                print(f"Optimizasyon sonrası: {total_assigned}/{num_deliveries} (%{success_rate:.1f})")

        return solution

    def _solve_exact_backtracking(self):
        """Küçük problemler için exact backtracking"""
        print("Exact backtracking çözümü...")

        assignment = {}
        unassigned_vars = list(self.variables)

        # Değişken sıralama - öncelik ve kısıtlılık bazlı
        unassigned_vars.sort(key=lambda var: self._get_variable_ordering_score(var), reverse=True)

        solution = self._backtrack_search_improved(assignment, unassigned_vars, depth=0)

        if solution:
            return self._format_solution(solution)
        else:
            print("Exact çözüm bulunamadı, hybrid yaklaşıma geçiliyor...")
            return self._solve_hybrid_approach()

    def _solve_hybrid_approach(self):
        """Orta boyutlu problemler için hybrid yaklaşım"""
        print("Hybrid çözüm yaklaşımı...")

        # İlk olarak greedy ile başlangıç çözümü
        initial_solution = self._greedy_initial_solution()

        # Local search ile iyileştirme
        improved_solution = self._local_search_improvement(initial_solution)

        return improved_solution

    def _solve_large_scale_heuristic(self):
        """Büyük problemler için heuristic yaklaşım"""
        print("Large-scale heuristic çözümü...")

        assignments = {drone.id: [] for drone in self.drones}

        # Teslimatları akıllıca sırala
        sorted_deliveries = self._smart_delivery_ordering()

        # İki aşamalı atama
        # Aşama 1: Öncelikli ve kısıtlı teslimatlar
        priority_deliveries = [dp for dp in sorted_deliveries if dp.priority >= 4]
        regular_deliveries = [dp for dp in sorted_deliveries if dp.priority < 4]

        # Öncelikli teslimatları ata
        for dp in priority_deliveries:
            best_drone = self._find_best_drone_advanced(dp, assignments)
            if best_drone is not None:
                assignments[best_drone].append(dp.id)

        # Normal teslimatları ata
        for dp in regular_deliveries:
            best_drone = self._find_best_drone_advanced(dp, assignments)
            if best_drone is not None:
                assignments[best_drone].append(dp.id)

        return assignments

    def _backtrack_search_improved(self, assignment, unassigned_vars, depth):
        """Geliştirilmiş backtracking - pruning ve heuristics ile"""

        # Derinlik limiti
        if depth > 100:
            return None

        # Tüm değişkenler atandı
        if not unassigned_vars:
            return assignment

        # En kısıtlı değişkeni seç (MRV + degree heuristic)
        var = self._select_most_constrained_variable(unassigned_vars, assignment)

        # Domain değerlerini sırala (LCV)
        ordered_values = self._order_domain_values_smart(var, assignment)

        for value in ordered_values:
            # Forward checking
            if not self._is_value_consistent(var, value, assignment):
                continue

            # Atama yap
            new_assignment = assignment.copy()
            new_assignment[var] = value

            # Constraint propagation
            remaining_vars = [v for v in unassigned_vars if v != var]
            pruned_domains = self._forward_check(var, value, remaining_vars)

            if pruned_domains is not None:
                # Recursive çağrı
                result = self._backtrack_search_improved(
                    new_assignment,
                    remaining_vars,
                    depth + 1
                )

                if result is not None:
                    return result

            # Backtrack - domain'leri geri yükle
            self._restore_domains(pruned_domains)

        return None

    def _greedy_initial_solution(self):
        """Akıllı greedy başlangıç çözümü"""
        assignments = {drone.id: [] for drone in self.drones}

        # Drone yük ve rota bilgileri
        drone_loads = {d.id: 0.0 for d in self.drones}
        drone_routes = {d.id: [] for d in self.drones}
        drone_positions = {d.id: d.start_pos for d in self.drones}

        # Teslimatları sırala
        sorted_deliveries = sorted(
            self.delivery_points,
            key=lambda dp: (5 - dp.priority, dp.weight),  # Öncelik önce, hafif paketler önce
        )

        for dp in sorted_deliveries:
            best_drone = None
            best_score = float('-inf')

            for drone in self.drones:
                # Temel uyumluluk
                if drone_loads[drone.id] + dp.weight > drone.max_weight:
                    continue

                # Skor hesapla
                score = self._calculate_assignment_score(drone, dp, drone_positions[drone.id], drone_loads[drone.id])

                if score > best_score:
                    best_score = score
                    best_drone = drone.id

            if best_drone is not None:
                assignments[best_drone].append(dp.id)
                drone_loads[best_drone] += dp.weight
                drone_positions[best_drone] = dp.pos
                drone_routes[best_drone].append(dp.id)

        return assignments

    def _local_search_improvement(self, initial_solution):
        """Local search ile çözüm iyileştirme"""
        current_solution = initial_solution.copy()
        current_score = self._evaluate_solution_quality(current_solution)

        improved = True
        iterations = 0

        while improved and iterations < self.max_iterations:
            improved = False
            iterations += 1

            # Swap operasyonu
            for drone1 in range(len(self.drones)):
                for drone2 in range(drone1 + 1, len(self.drones)):
                    if not current_solution[drone1] or not current_solution[drone2]:
                        continue

                    # Her teslimat çifti için swap dene
                    for i, dp1 in enumerate(current_solution[drone1]):
                        for j, dp2 in enumerate(current_solution[drone2]):
                            # Swap yap
                            new_solution = self._swap_deliveries(current_solution, drone1, i, drone2, j)

                            # Feasibility kontrolü
                            if self._is_solution_feasible(new_solution):
                                new_score = self._evaluate_solution_quality(new_solution)

                                if new_score > current_score + self.improvement_threshold:
                                    current_solution = new_solution
                                    current_score = new_score
                                    improved = True
                                    break
                        if improved:
                            break
                if improved:
                    break

        print(f"Local search: {iterations} iterasyon, skor: {current_score:.2f}")
        return current_solution

    def _post_process_solution(self, solution):
        """Atanmamış teslimatları zorla atamaya çalış"""
        # Atanmamış teslimatları bul
        assigned_deliveries = set()
        for deliveries in solution.values():
            assigned_deliveries.update(deliveries)

        unassigned = [dp for dp in self.delivery_points if dp.id not in assigned_deliveries]

        if not unassigned:
            return solution

        print(f"\n{len(unassigned)} atanmamış teslimat için çözüm aranıyor...")

        # Kısıtları gevşeterek atama yap
        for dp in unassigned:
            # En az yüklü drone'u bul
            min_load = float('inf')
            best_drone = None

            for drone_id in range(len(self.drones)):
                current_load = len(solution[drone_id])

                # Temel kontroller (gevşetilmiş)
                total_weight = sum(
                    self.dp_weights.get(dp_id, 0)
                    for dp_id in solution[drone_id]
                ) + dp.weight

                # %20 kapasite aşımına izin ver
                if total_weight <= self.drone_capacities[drone_id] * 1.2:
                    if current_load < min_load:
                        min_load = current_load
                        best_drone = drone_id

            if best_drone is not None:
                solution[best_drone].append(dp.id)
                print(f"  Teslimat {dp.id} -> Drone {best_drone} (gevşetilmiş kısıtlar)")

        return solution

    # Yardımcı metodlar
    def _get_variable_ordering_score(self, var):
        """Değişken sıralama skoru"""
        delivery_id = int(var.split('_')[1])
        priority = self.dp_priorities.get(delivery_id, 3)
        domain_size = len(self.domains.get(var, []))

        # Yüksek öncelik, küçük domain = yüksek skor
        return priority * 10 - domain_size

    def _select_most_constrained_variable(self, unassigned_vars, assignment):
        """En kısıtlı değişkeni seç (MRV + degree)"""
        best_var = unassigned_vars[0]
        best_score = float('inf')

        for var in unassigned_vars:
            # Domain boyutu
            domain_size = len(self.domains.get(var, []))

            # Kısıt derecesi (bu değişkenle ilişkili kısıt sayısı)
            constraint_degree = self._count_constraints(var, unassigned_vars)

            # Skor: küçük domain, yüksek derece
            score = domain_size - constraint_degree * 0.1

            if score < best_score:
                best_score = score
                best_var = var

        return best_var

    def _order_domain_values_smart(self, var, assignment):
        """Domain değerlerini akıllıca sırala (LCV + load balancing)"""
        delivery_id = int(var.split('_')[1])
        values = self.domains.get(var, [])

        # Her değer için skor hesapla
        value_scores = []
        for drone_id in values:
            # Mevcut yük
            current_load = sum(1 for v, d in assignment.items() if d == drone_id)

            # Uyumluluk skoru
            compatibility = self.compatibility_scores.get(drone_id, {}).get(delivery_id, 0.5)

            # Load balancing faktörü
            load_factor = 1.0 / (current_load + 1)

            # Toplam skor
            score = compatibility * 0.7 + load_factor * 0.3
            value_scores.append((score, drone_id))

        # Skora göre sırala (yüksekten düşüğe)
        value_scores.sort(reverse=True)
        return [drone_id for _, drone_id in value_scores]

    def _is_value_consistent(self, var, value, assignment):
        """Değer ataması tutarlı mı - esnek kontrol"""
        delivery_id = int(var.split('_')[1])
        drone_id = value

        # Test assignment
        test_assignment = assignment.copy()
        test_assignment[var] = value

        # Temel fizibilite kontrolü
        drone_deliveries = [
            int(v.split('_')[1]) for v, d in test_assignment.items() if d == drone_id
        ]

        # Ağırlık kontrolü (biraz esnek)
        total_weight = sum(self.dp_weights.get(dp_id, 0) for dp_id in drone_deliveries)
        if total_weight > self.drone_capacities[drone_id] * 1.1:  # %10 tolerans
            return False

        # Basit batarya kontrolü
        if not self._simple_battery_check(drone_id, drone_deliveries):
            return False

        return True

    def _simple_battery_check(self, drone_id, delivery_ids):
        """Basitleştirilmiş batarya kontrolü"""
        if not delivery_ids:
            return True

        # Toplam mesafe tahmini
        total_distance = 0
        drone_pos = self.drones[drone_id].start_pos

        for dp_id in delivery_ids:
            dp_pos = self.dp_positions[dp_id]
            total_distance += self._calculate_distance(drone_pos, dp_pos)
            drone_pos = dp_pos

        # Depot'a dönüş
        total_distance += self._calculate_distance(drone_pos, self.depot_position)

        # Basit enerji hesabı
        energy_needed = total_distance * 0.1

        return energy_needed <= self.drone_batteries[drone_id] * 0.85  # %15 güvenlik marjı

    def _forward_check(self, var, value, remaining_vars):
        """Forward checking - constraint propagation"""
        # Basit implementasyon - geliştirilebilir
        return {}

    def _restore_domains(self, pruned_domains):
        """Domain'leri geri yükle"""
        pass

    def _smart_delivery_ordering(self):
        """Teslimatları akıllıca sırala"""
        # Çok kriterli sıralama
        return sorted(
            self.delivery_points,
            key=lambda dp: (
                5 - dp.priority,  # Öncelik (ters)
                self._get_delivery_difficulty(dp),  # Zorluk
                dp.weight  # Ağırlık
            )
        )

    def _get_delivery_difficulty(self, delivery_point):
        """Teslimat zorluğunu hesapla"""
        difficulty = 0

        # Uzaklık faktörü
        distance_from_depot = self._calculate_distance(self.depot_position, delivery_point.pos)
        difficulty += distance_from_depot / 1000  # Normalize

        # Ağırlık faktörü
        difficulty += delivery_point.weight / 10

        # Zaman penceresi darlığı
        if delivery_point.id in self.dp_time_windows:
            start, end = self.dp_time_windows[delivery_point.id]
            window_minutes = (end.hour - start.hour) * 60 + (end.minute - start.minute)
            difficulty += 100 / max(window_minutes, 1)

        return difficulty

    def _find_best_drone_advanced(self, delivery_point, current_assignments):
        """Gelişmiş drone seçimi"""
        best_drone = None
        best_score = float('-inf')

        for drone in self.drones:
            # Fizibilite kontrolü
            if not self._can_drone_handle_delivery(drone, delivery_point, current_assignments[drone.id]):
                continue

            # Çok kriterli skorlama
            score = 0

            # Uyumluluk skoru
            compatibility = self.compatibility_scores.get(drone.id, {}).get(delivery_point.id, 0.5)
            score += compatibility * 40

            # Yük dengesi skoru
            current_load = len(current_assignments[drone.id])
            avg_load = sum(len(assignments) for assignments in current_assignments.values()) / len(self.drones)
            load_balance = 1.0 / (abs(current_load - avg_load) + 1)
            score += load_balance * 30

            # Verimlilik skoru (mevcut rotaya yakınlık)
            if current_assignments[drone.id]:
                last_delivery_id = current_assignments[drone.id][-1]
                last_pos = self.dp_positions[last_delivery_id]
                distance = self._calculate_distance(last_pos, delivery_point.pos)
            else:
                distance = self._calculate_distance(drone.start_pos, delivery_point.pos)

            efficiency = 1000 / (distance + 1)
            score += efficiency * 30

            if score > best_score:
                best_score = score
                best_drone = drone.id

        return best_drone

    def _can_drone_handle_delivery(self, drone, delivery_point, current_deliveries):
        """Drone bu teslimatı yapabilir mi - esnek kontrol"""
        # Ağırlık kontrolü
        current_weight = sum(self.dp_weights.get(dp_id, 0) for dp_id in current_deliveries)
        if current_weight + delivery_point.weight > drone.max_weight * 1.05:  # %5 tolerans
            return False

        # Basit batarya tahmini
        delivery_count = len(current_deliveries) + 1
        if delivery_count * 100 > drone.battery:  # Çok basit tahmin
            return False

        return True

    def _calculate_assignment_score(self, drone, delivery_point, current_pos, current_load):
        """Atama skoru hesapla"""
        score = 0

        # Mesafe skoru
        distance = self._calculate_distance(current_pos, delivery_point.pos)
        score -= distance * 0.01

        # Öncelik skoru
        score += delivery_point.priority * 10

        # Yük dengesi
        score -= current_load * 5

        # Hız uyumu (acil teslimatlar için hızlı drone)
        if delivery_point.priority >= 4:
            score += drone.speed * 0.5

        return score

    def _swap_deliveries(self, solution, drone1, idx1, drone2, idx2):
        """İki teslimatı swap et"""
        new_solution = {}
        for drone_id, deliveries in solution.items():
            new_solution[drone_id] = deliveries.copy()

        # Swap
        dp1 = new_solution[drone1][idx1]
        dp2 = new_solution[drone2][idx2]

        new_solution[drone1][idx1] = dp2
        new_solution[drone2][idx2] = dp1

        return new_solution

    def _is_solution_feasible(self, solution):
        """Çözüm fizibil mi - hızlı kontrol"""
        for drone_id, deliveries in solution.items():
            # Ağırlık kontrolü
            total_weight = sum(self.dp_weights.get(dp_id, 0) for dp_id in deliveries)
            if total_weight > self.drone_capacities[drone_id] * 1.1:  # %10 tolerans
                return False

            # Basit batarya kontrolü
            if not self._simple_battery_check(drone_id, deliveries):
                return False

        return True

    def _evaluate_solution_quality(self, solution):
        """Çözüm kalitesini değerlendir (0-100)"""
        score = 0
        max_score = 0

        # Atanmış teslimat sayısı
        total_assigned = sum(len(assignments) for assignments in solution.values())
        total_deliveries = len(self.delivery_points)

        assignment_score = (total_assigned / total_deliveries) * 40
        score += assignment_score
        max_score += 40

        # Öncelik uyumu
        priority_score = 0
        for drone_id, deliveries in solution.items():
            for dp_id in deliveries:
                priority = self.dp_priorities.get(dp_id, 3)
                priority_score += priority

        max_priority = sum(dp.priority for dp in self.delivery_points)
        if max_priority > 0:
            score += (priority_score / max_priority) * 20
        max_score += 20

        # Yük dengesi
        if len(solution) > 0:
            loads = [len(deliveries) for deliveries in solution.values()]
            avg_load = sum(loads) / len(loads)
            variance = sum((l - avg_load) ** 2 for l in loads) / len(loads)
            std_dev = math.sqrt(variance)

            # Düşük standart sapma = iyi denge
            balance_score = max(0, 20 - std_dev * 5)
            score += balance_score
        max_score += 20

        # Verimlilik (toplam mesafe)
        total_distance = 0
        for drone_id, deliveries in solution.items():
            if deliveries:
                route_distance = self._calculate_route_distance(drone_id, deliveries)
                total_distance += route_distance

        # Referans mesafe (her teslimat direkt gidiş-dönüş)
        reference_distance = sum(
            2 * self._calculate_distance(self.depot_position, dp.pos)
            for dp in self.delivery_points
        )

        if reference_distance > 0:
            efficiency = (reference_distance - total_distance) / reference_distance
            efficiency_score = max(0, efficiency * 20)
            score += efficiency_score
        max_score += 20

        return (score / max_score) * 100

    def _calculate_route_distance(self, drone_id, delivery_ids):
        """Rota toplam mesafesini hesapla"""
        if not delivery_ids:
            return 0

        total_distance = 0
        current_pos = self.drones[drone_id].start_pos

        for dp_id in delivery_ids:
            next_pos = self.dp_positions[dp_id]
            total_distance += self._calculate_distance(current_pos, next_pos)
            current_pos = next_pos

        # Depot'a dönüş
        total_distance += self._calculate_distance(current_pos, self.depot_position)

        return total_distance

    def _count_constraints(self, var, unassigned_vars):
        """Değişkenle ilişkili kısıt sayısı"""
        # Basit implementasyon - her değişken diğerleriyle ilişkili
        return len(unassigned_vars) - 1

    @lru_cache(maxsize=10000)
    def _calculate_distance(self, pos1, pos2):
        """İki nokta arası mesafe - cached"""
        if isinstance(pos1, list):
            pos1 = tuple(pos1)
        if isinstance(pos2, list):
            pos2 = tuple(pos2)

        return math.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)

    def _parse_time(self, time_str):
        """Zaman parse - daha robust"""
        if isinstance(time_str, datetime):
            return time_str.time()

        if hasattr(time_str, 'hour'):  # Zaten time object
            return time_str

        try:
            # Standart format
            return datetime.strptime(time_str, "%H:%M").time()
        except:
            try:
                # Alternatif format
                return datetime.strptime(time_str, "%H.%M").time()
            except:
                # Default
                return datetime.strptime("09:00", "%H:%M").time()

    def _format_solution(self, solution):
        """Çözümü istenen formata dönüştür"""
        assignments = {drone.id: [] for drone in self.drones}

        if isinstance(solution, dict):
            for var_name, drone_id in solution.items():
                if var_name.startswith("delivery_"):
                    delivery_id = int(var_name.split('_')[1])
                    if drone_id is not None:
                        assignments[drone_id].append(delivery_id)

        return assignments

    def print_solution_summary(self, solution):
        """Çözüm özetini yazdır"""
        if not solution:
            print("Çözüm bulunamadı!")
            return

        print("\n=== ÇÖZÜM ÖZETİ ===")

        total_assigned = 0
        total_weight = 0
        total_distance = 0

        for drone_id, deliveries in solution.items():
            if not deliveries:
                continue

            drone = self.drone_lookup[drone_id]
            print(f"\nDrone {drone_id}:")
            print(f"  Teslimat sayısı: {len(deliveries)}")

            # Ağırlık hesabı
            route_weight = sum(self.dp_weights.get(dp_id, 0) for dp_id in deliveries)
            print(f"  Toplam ağırlık: {route_weight:.1f}/{drone.max_weight:.1f} kg")

            # Mesafe hesabı
            route_distance = self._calculate_route_distance(drone_id, deliveries)
            print(f"  Rota mesafesi: {route_distance:.1f} m")

            # Öncelikler
            priorities = [self.dp_priorities.get(dp_id, 3) for dp_id in deliveries]
            print(f"  Öncelikler: {priorities}")

            total_assigned += len(deliveries)
            total_weight += route_weight
            total_distance += route_distance

        print(f"\n=== GENEL ÖZET ===")
        print(f"Toplam atanan teslimat: {total_assigned}/{len(self.delivery_points)}")
        print(f"Başarı oranı: {(total_assigned / len(self.delivery_points) * 100):.1f}%")
        print(f"Toplam ağırlık: {total_weight:.1f} kg")
        print(f"Toplam mesafe: {total_distance:.1f} m")

        # Atanmamış teslimatlar
        assigned_ids = set()
        for deliveries in solution.values():
            assigned_ids.update(deliveries)

        unassigned = [dp for dp in self.delivery_points if dp.id not in assigned_ids]
        if unassigned:
            print(f"\n⚠️ Atanmamış teslimatlar: {len(unassigned)}")
            for dp in unassigned[:5]:  # İlk 5 tanesini göster
                print(f"  - ID: {dp.id}, Ağırlık: {dp.weight}kg, Öncelik: {dp.priority}")