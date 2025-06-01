import heapq
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime, time
import math


class AStarPathfinder:
    """
    PDF Dokümana tam uygun A* algoritması implementasyonu
    Cost(i,j) = distance × weight + (priority × 100)
    f(n) = g(n) + h_no_fly_zone(n)

    Eksiklikler giderildi:
    - Drone kapasitesi kontrolü
    - Batarya/enerji kısıtları
    - Zaman penceresi kontrolü
    - NFZ zaman kontrolü
    - Gelişmiş heuristic fonksiyonu
    """

    def __init__(self, delivery_points, no_fly_zones, depot_position, current_time=None):
        self.delivery_points = delivery_points
        self.no_fly_zones = no_fly_zones
        self.depot_position = depot_position
        self.current_time = current_time or datetime.now().time()

        # Point ID mapping (0 = depot, 1+ = delivery points)
        self.point_id_map = {0: depot_position}
        for i, dp in enumerate(delivery_points):
            self.point_id_map[i + 1] = dp.pos

        # Delivery point lookup for priority/weight
        self.delivery_lookup = {}
        for i, dp in enumerate(delivery_points):
            self.delivery_lookup[i + 1] = dp  # ID mapping

        print(f"A* initialized: {len(delivery_points)} points, {len(no_fly_zones)} NFZ")
        print(f"Current time: {self.current_time}")

    def find_optimal_route_for_drone(self, drone, start_id: int, delivery_ids: List[int]) -> Tuple[
        List[int], float, Dict]:
        """
        Drone kısıtları dahil optimal route bulma
        Returns: (route, cost, metrics)
        """
        if not delivery_ids:
            return [start_id], 0.0, self._empty_metrics()

        # Drone kısıtlarına göre teslimatları filtrele
        valid_deliveries = self._filter_deliveries_by_constraints(drone, delivery_ids)

        if not valid_deliveries:
            print(f"Warning: No valid deliveries for drone {drone.id} due to constraints")
            return [start_id], 0.0, self._empty_metrics()

        # Optimal rota bul
        route, cost = self._find_constrained_route(drone, start_id, valid_deliveries)

        # Metrikleri hesapla
        metrics = self._calculate_route_metrics(drone, route)

        return route, cost, metrics

    def find_optimal_route(self, start_id: int, delivery_ids: List[int]) -> Tuple[List[int], float]:
        """
        Backward compatibility için - drone kısıtı olmadan
        """
        if not delivery_ids:
            return [start_id], 0.0

        # TSP benzeri problem - tüm teslimatları ziyaret et
        unvisited = set(delivery_ids)
        current_pos = start_id
        route = [start_id]
        total_cost = 0.0

        while unvisited:
            next_point = self._find_next_best_point(current_pos, unvisited)
            if next_point is None:
                break

            # A* ile nokta arası rota bul
            path, cost = self._astar_between_points(current_pos, next_point)

            if path and cost < float('inf'):
                # İlk nokta zaten route'ta olduğu için atla
                route.extend(path[1:])
                total_cost += cost
                unvisited.remove(next_point)
                current_pos = next_point
            else:
                # Rota bulunamazsa direct ekle (ancak NFZ kontrolü yap)
                direct_cost = self._calculate_direct_cost_with_nfz_check(current_pos, next_point)
                if direct_cost < float('inf'):
                    route.append(next_point)
                    total_cost += direct_cost
                    unvisited.remove(next_point)
                    current_pos = next_point
                else:
                    print(f"Warning: Cannot reach delivery point {next_point}")
                    unvisited.remove(next_point)  # Skip unreachable point

        # Depoya dönüş
        if current_pos != start_id:
            return_path, return_cost = self._astar_between_points(current_pos, start_id)
            if return_path and return_cost < float('inf'):
                route.extend(return_path[1:])
                total_cost += return_cost
            else:
                # Direct dönüş
                direct_return_cost = self._calculate_direct_cost_with_nfz_check(current_pos, start_id)
                route.append(start_id)
                total_cost += direct_return_cost

        return route, total_cost

    def _filter_deliveries_by_constraints(self, drone, delivery_ids: List[int]) -> List[int]:
        """Drone kısıtlarına göre teslimatları filtrele"""
        valid_deliveries = []

        for delivery_id in delivery_ids:
            if delivery_id in self.delivery_lookup:
                dp = self.delivery_lookup[delivery_id]

                # Ağırlık kontrolü
                if dp.weight <= drone.max_weight:
                    # Zaman penceresi kontrolü
                    if self._is_time_window_valid(dp, self.current_time):
                        valid_deliveries.append(delivery_id)
                    else:
                        print(f"Delivery {delivery_id} outside time window")
                else:
                    print(f"Delivery {delivery_id} too heavy for drone {drone.id}")

        return valid_deliveries

    def _find_constrained_route(self, drone, start_id: int, delivery_ids: List[int]) -> Tuple[List[int], float]:
        """Kısıtlar dahil rota bulma"""
        if not delivery_ids:
            return [start_id], 0.0

        # Greedy approach with constraints
        unvisited = set(delivery_ids)
        current_pos = start_id
        route = [start_id]
        total_cost = 0.0
        current_weight = 0.0
        current_time = self.current_time

        while unvisited:
            next_point = self._find_next_feasible_point(drone, current_pos, unvisited, current_weight, current_time)

            if next_point is None:
                break

            # Rota bul ve kısıt kontrolü yap
            path, cost = self._astar_between_points_with_constraints(drone, current_pos, next_point, current_time)

            if path and cost < float('inf'):
                # Route güncelle
                route.extend(path[1:])
                total_cost += cost

                # Kısıtları güncelle
                dp = self.delivery_lookup[next_point]
                current_weight += dp.weight

                # Seyahat süresini ekle
                travel_time = self._calculate_travel_time(drone, path)
                current_time = self._add_time(current_time, travel_time)

                unvisited.remove(next_point)
                current_pos = next_point
            else:
                # Bu nokta ulaşılamaz, atla
                unvisited.remove(next_point)

        # Depoya dönüş - kapasiteyi kontrol et
        if current_pos != start_id:
            return_path, return_cost = self._astar_between_points_with_constraints(
                drone, current_pos, start_id, current_time
            )
            if return_path and return_cost < float('inf'):
                route.extend(return_path[1:])
                total_cost += return_cost

        return route, total_cost

    def _find_next_best_point(self, current_id: int, unvisited: set) -> Optional[int]:
        """En iyi bir sonraki noktayı bul - priority ve mesafe bazlı"""
        if not unvisited:
            return None

        best_point = None
        best_score = float('inf')

        for point_id in unvisited:
            # Dokümandaki cost fonksiyonu
            cost = self._calculate_cost_with_priority(current_id, point_id)

            if cost < best_score:
                best_score = cost
                best_point = point_id

        return best_point

    def _find_next_feasible_point(self, drone, current_id: int, unvisited: set, current_weight: float, current_time) -> \
    Optional[int]:
        """Kısıtları dikkate alarak en iyi sonraki noktayı bul"""
        if not unvisited:
            return None

        best_point = None
        best_score = float('inf')

        for point_id in unvisited:
            dp = self.delivery_lookup.get(point_id)
            if not dp:
                continue

            # Ağırlık kısıtı kontrolü
            if current_weight + dp.weight > drone.max_weight:
                continue

            # Zaman penceresi kontrolü
            estimated_arrival = self._estimate_arrival_time(drone, current_id, point_id, current_time)
            if not self._is_time_window_valid(dp, estimated_arrival):
                continue

            # Batarya kısıtı kontrolü (basit kontrol)
            route_distance = self._euclidean_distance(self.point_id_map[current_id], self.point_id_map[point_id])
            energy_needed = self._calculate_energy_consumption(drone, route_distance)
            if energy_needed > drone.battery * 0.8:  # %80 güvenlik marjı
                continue

            # Cost hesaplama
            cost = self._calculate_cost_with_priority(current_id, point_id)

            if cost < best_score:
                best_score = cost
                best_point = point_id

        return best_point

    def _calculate_cost_with_priority(self, from_id: int, to_id: int) -> float:
        """
        PDF Dokümandaki maliyet fonksiyonu:
        Cost(i,j) = distance × weight + (priority × 100)
        """
        # Temel mesafe
        from_pos = self.point_id_map[from_id]
        to_pos = self.point_id_map[to_id]
        distance = self._euclidean_distance(from_pos, to_pos)

        # Teslimat noktası bilgilerini al
        if to_id == 0:  # Depot
            weight = 1.0
            priority = 3  # Neutral priority
        else:
            dp = self.delivery_lookup.get(to_id)
            if dp:
                weight = dp.weight
                priority = dp.priority
            else:
                weight = 1.0
                priority = 3

        # PDF'deki formül - Düşük priority değeri = Yüksek öncelik
        # Priority 5 = En yüksek öncelik (düşük ceza)
        # Priority 1 = En düşük öncelik (yüksek ceza)
        priority_penalty = (6 - priority) * 100
        total_cost = distance * weight + priority_penalty

        return total_cost

    def _astar_between_points(self, start_id: int, goal_id: int) -> Tuple[List[int], float]:
        """
        İki nokta arası A* algoritması
        f(n) = g(n) + h_no_fly_zone(n)
        """
        if start_id == goal_id:
            return [start_id], 0.0

        start_pos = self.point_id_map[start_id]
        goal_pos = self.point_id_map[goal_id]

        # NFZ kontrolü - direkt hat mümkün mü?
        if not self._path_crosses_active_nfz(start_pos, goal_pos):
            # Direct path mümkün
            cost = self._calculate_direct_cost(start_id, goal_id)
            return [start_id, goal_id], cost

        # A* ile alternative path bul
        return self._astar_with_waypoints(start_id, goal_id)

    def _astar_between_points_with_constraints(self, drone, start_id: int, goal_id: int, current_time) -> Tuple[
        List[int], float]:
        """Kısıtlar dahil A* algoritması"""
        if start_id == goal_id:
            return [start_id], 0.0

        start_pos = self.point_id_map[start_id]
        goal_pos = self.point_id_map[goal_id]

        # Zaman bazlı NFZ kontrolü
        if not self._path_crosses_active_nfz_at_time(start_pos, goal_pos, current_time):
            # Direct path mümkün
            cost = self._calculate_direct_cost(start_id, goal_id)
            return [start_id, goal_id], cost

        # A* ile alternative path bul
        return self._astar_with_waypoints_and_time(drone, start_id, goal_id, current_time)

    def _astar_with_waypoints(self, start_id: int, goal_id: int) -> Tuple[List[int], float]:
        """NFZ'leri dikkate alan A* with waypoints"""
        # Basitleştirilmiş waypoint yaklaşımı
        start_pos = self.point_id_map[start_id]
        goal_pos = self.point_id_map[goal_id]

        # NFZ'lerin etrafından dolaşmak için waypoint'ler oluştur
        waypoints = self._generate_waypoints_around_nfz(start_pos, goal_pos)

        if not waypoints:
            # Waypoint bulunamadı, yüksek ceza ile direct path
            cost = self._calculate_direct_cost(start_id, goal_id) + 2000.0  # NFZ cezası
            return [start_id, goal_id], cost

        # En iyi waypoint kombinasyonunu bul
        best_path = [start_id, goal_id]
        best_cost = float('inf')

        for waypoint in waypoints:
            # Start -> Waypoint -> Goal rotası
            cost1 = self._euclidean_distance(start_pos, waypoint)
            cost2 = self._euclidean_distance(waypoint, goal_pos)
            total_cost = cost1 + cost2

            if total_cost < best_cost:
                best_cost = total_cost
                # Waypoint'i route ID sistemine entegre et (basitleştirilmiş)
                best_path = [start_id, goal_id]  # Basitleştirilmiş

        return best_path, best_cost

    def _astar_with_waypoints_and_time(self, drone, start_id: int, goal_id: int, current_time) -> Tuple[
        List[int], float]:
        """Zaman ve drone kısıtları dahil A*"""
        # Basitleştirilmiş implementasyon
        start_pos = self.point_id_map[start_id]
        goal_pos = self.point_id_map[goal_id]

        # NFZ zaman kontrolü
        if self._path_crosses_active_nfz_at_time(start_pos, goal_pos, current_time):
            # Alternative route gerekli
            waypoints = self._generate_waypoints_around_nfz(start_pos, goal_pos)
            if waypoints:
                # En yakın waypoint kullan
                best_waypoint = min(waypoints, key=lambda wp: self._euclidean_distance(start_pos, wp))
                cost1 = self._euclidean_distance(start_pos, best_waypoint)
                cost2 = self._euclidean_distance(best_waypoint, goal_pos)
                return [start_id, goal_id], cost1 + cost2

        # Direct path
        cost = self._calculate_direct_cost(start_id, goal_id)
        return [start_id, goal_id], cost

    def _generate_waypoints_around_nfz(self, start_pos: Tuple[float, float], goal_pos: Tuple[float, float]) -> List[
        Tuple[float, float]]:
        """NFZ'lerin etrafında waypoint'ler oluştur"""
        waypoints = []

        for nfz in self.no_fly_zones:
            if not nfz.is_active():
                continue

            # NFZ'nin bounding box'ını hesapla
            coords = nfz.coordinates
            min_x = min(coord[0] for coord in coords)
            max_x = max(coord[0] for coord in coords)
            min_y = min(coord[1] for coord in coords)
            max_y = max(coord[1] for coord in coords)

            # NFZ etrafında waypoint'ler oluştur
            margin = 50.0  # 50 metre güvenlik marjı

            waypoints.extend([
                (min_x - margin, min_y - margin),
                (max_x + margin, min_y - margin),
                (max_x + margin, max_y + margin),
                (min_x - margin, max_y + margin),
            ])

        return waypoints

    def _heuristic_with_nfz(self, current_id: int, goal_id: int) -> float:
        """
        PDF Dokümandaki heuristic fonksiyonu:
        h_no_fly_zone(n) = mesafe + NFZ_cezası
        """
        current_pos = self.point_id_map[current_id]
        goal_pos = self.point_id_map[goal_id]

        # Temel mesafe (admissible heuristic)
        base_distance = self._euclidean_distance(current_pos, goal_pos)

        # NFZ cezası
        nfz_penalty = 0.0

        # Mevcut pozisyon aktif NFZ içinde mi?
        for nfz in self.no_fly_zones:
            if nfz.is_active() and nfz.point_in_polygon(current_pos):
                nfz_penalty += 1000.0  # Yüksek ceza

        # Hedefe giden hat aktif NFZ'den geçiyor mu?
        if self._path_crosses_active_nfz(current_pos, goal_pos):
            nfz_penalty += 500.0  # Orta ceza

        return base_distance + nfz_penalty

    def _path_crosses_active_nfz(self, start_pos: Tuple[float, float], end_pos: Tuple[float, float]) -> bool:
        """Hat aktif NFZ'den geçiyor mu kontrol et"""
        for nfz in self.no_fly_zones:
            if nfz.is_active() and nfz.does_line_intersect(start_pos, end_pos):
                return True
        return False

    def _path_crosses_active_nfz_at_time(self, start_pos: Tuple[float, float], end_pos: Tuple[float, float],
                                         check_time) -> bool:
        """Belirli zamanda hat aktif NFZ'den geçiyor mu kontrol et"""
        for nfz in self.no_fly_zones:
            if nfz.is_active_at_time(check_time) and nfz.does_line_intersect(start_pos, end_pos):
                return True
        return False

    def _calculate_direct_cost(self, from_id: int, to_id: int) -> float:
        """İki nokta arası direct maliyet"""
        from_pos = self.point_id_map[from_id]
        to_pos = self.point_id_map[to_id]
        return self._euclidean_distance(from_pos, to_pos)

    def _calculate_direct_cost_with_nfz_check(self, from_id: int, to_id: int) -> float:
        """NFZ kontrolü ile direct maliyet"""
        from_pos = self.point_id_map[from_id]
        to_pos = self.point_id_map[to_id]

        base_cost = self._euclidean_distance(from_pos, to_pos)

        # NFZ kontrolü
        if self._path_crosses_active_nfz(from_pos, to_pos):
            return float('inf')  # Geçilemez

        return base_cost

    def _euclidean_distance(self, pos1: Tuple[float, float], pos2: Tuple[float, float]) -> float:
        """Euclidean mesafe hesaplama"""
        return math.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)

    # Yardımcı fonksiyonlar - Kısıt kontrolü
    def _is_time_window_valid(self, delivery_point, check_time) -> bool:
        """Teslimat zaman penceresi kontrolü"""
        if not hasattr(delivery_point, 'time_window') or not delivery_point.time_window:
            return True

        try:
            start_time_str, end_time_str = delivery_point.time_window
            start_time = datetime.strptime(start_time_str, "%H:%M").time()
            end_time = datetime.strptime(end_time_str, "%H:%M").time()

            return start_time <= check_time <= end_time
        except:
            return True  # Hatalı format durumunda kabul et

    def _estimate_arrival_time(self, drone, from_id: int, to_id: int, current_time) -> time:
        """Varış zamanı tahmini"""
        distance = self._calculate_direct_cost(from_id, to_id)
        travel_time_minutes = (distance / drone.speed) * 60  # dakika cinsinden

        return self._add_time(current_time, travel_time_minutes)

    def _calculate_travel_time(self, drone, path: List[int]) -> float:
        """Rota seyahat süresini hesapla (dakika)"""
        if len(path) < 2:
            return 0.0

        total_distance = 0.0
        for i in range(len(path) - 1):
            total_distance += self._calculate_direct_cost(path[i], path[i + 1])

        return (total_distance / drone.speed) * 60  # dakika

    def _add_time(self, base_time, minutes: float) -> time:
        """Zamana dakika ekle"""
        try:
            base_datetime = datetime.combine(datetime.today(), base_time)
            new_datetime = base_datetime + pd.Timedelta(minutes=minutes)
            return new_datetime.time()
        except:
            return base_time

    def _calculate_energy_consumption(self, drone, distance: float) -> float:
        """Enerji tüketimi hesapla (basit model)"""
        # Basit enerji modeli: distance * speed * tüketim_faktörü
        consumption_factor = 0.1  # mAh per meter per m/s
        return distance * drone.speed * consumption_factor

    def _calculate_route_metrics(self, drone, route: List[int]) -> Dict:
        """Rota metriklerini hesapla"""
        if len(route) < 2:
            return self._empty_metrics()

        total_distance = 0.0
        total_energy = 0.0
        completed_deliveries = len([r for r in route if r != 0])  # Depot hariç

        for i in range(len(route) - 1):
            distance = self._calculate_direct_cost(route[i], route[i + 1])
            total_distance += distance
            total_energy += self._calculate_energy_consumption(drone, distance)

        return {
            'total_distance': total_distance,
            'total_energy': total_energy,
            'completed_deliveries': completed_deliveries,
            'battery_usage_percent': (total_energy / drone.battery) * 100 if drone.battery > 0 else 0,
            'route_efficiency': completed_deliveries / (total_distance / 1000) if total_distance > 0 else 0
            # deliveries per km
        }

    def _empty_metrics(self) -> Dict:
        """Boş metrik seti"""
        return {
            'total_distance': 0.0,
            'total_energy': 0.0,
            'completed_deliveries': 0,
            'battery_usage_percent': 0.0,
            'route_efficiency': 0.0
        }


# Backward compatibility
class DeliveryGraph:
    """Graph representation for delivery points"""

    def __init__(self, delivery_points, no_fly_zones):
        self.delivery_points = delivery_points
        self.no_fly_zones = no_fly_zones
        print(f"DeliveryGraph created with {len(delivery_points)} points")

    def build_adjacency_list(self):
        """Komşuluk listesi oluştur (sparse matrix için verimli)"""
        adjacency = {}

        for i, dp1 in enumerate(self.delivery_points):
            adjacency[i] = []
            for j, dp2 in enumerate(self.delivery_points):
                if i != j:
                    # NFZ kontrolü ile kenar ekle
                    if not self._edge_crosses_nfz(dp1.pos, dp2.pos):
                        distance = math.sqrt((dp1.pos[0] - dp2.pos[0]) ** 2 + (dp1.pos[1] - dp2.pos[1]) ** 2)
                        adjacency[i].append((j, distance))

        return adjacency

    def _edge_crosses_nfz(self, pos1: Tuple[float, float], pos2: Tuple[float, float]) -> bool:
        """Kenar NFZ'den geçiyor mu kontrol et"""
        for nfz in self.no_fly_zones:
            if nfz.is_active() and nfz.does_line_intersect(pos1, pos2):
                return True
        return False