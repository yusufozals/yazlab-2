import heapq
import numpy as np
from ..models.no_fly_zone import NoFlyZone


class AStarPathfinder:
    def __init__(self, delivery_points, no_fly_zones, depot_position, current_time="09:00"):
        """
        A* algoritması ile rota bulucu

        Parametreler:
        -------------
        delivery_points : list
            DeliveryPoint nesnelerinin listesi
        no_fly_zones : list
            NoFlyZone nesnelerinin listesi
        depot_position : tuple
            Depo konumu (x, y)
        current_time : str
            Mevcut zaman ("HH:MM" formatında)
        """
        self.delivery_points = delivery_points
        self.no_fly_zones = no_fly_zones
        self.depot_position = depot_position
        self.current_time = current_time

        # Tüm noktaları ekle (depo + teslimat noktaları)
        self.all_points = [depot_position]
        self.point_id_map = {0: depot_position}

        for dp in delivery_points:
            self.all_points.append(dp.pos)
            self.point_id_map[dp.id + 1] = dp.pos  # Depo 0, teslimatlar 1'den başlar

    def heuristic(self, a, b):
        """İki nokta arasındaki kuş uçuşu mesafeyi hesaplar"""
        return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

    def get_active_no_fly_zones(self):
        """Mevcut zamanda aktif olan no-fly zone'ları döndürür"""
        return [nfz for nfz in self.no_fly_zones if nfz.is_active(self.current_time)]

    def is_path_blocked(self, start, end):
        """İki nokta arasındaki yolun no-fly zone ile engellenip engellenmediğini kontrol eder"""
        active_nfzs = self.get_active_no_fly_zones()

        for nfz in active_nfzs:
            if nfz.does_line_intersect(start, end):
                return True

        return False

    def get_neighbors(self, node_id, max_distance=float('inf')):
        """Bir düğümün komşularını döndürür"""
        neighbors = []
        node_pos = self.point_id_map[node_id]

        for other_id, other_pos in self.point_id_map.items():
            if other_id == node_id:
                continue  # Aynı düğüm

            # Mesafeyi hesapla
            distance = self.heuristic(node_pos, other_pos)

            # Çok uzaktaki düğümleri filtrele
            if distance > max_distance:
                continue

            # Yol engellenmiş mi kontrol et
            if self.is_path_blocked(node_pos, other_pos):
                continue

            neighbors.append((other_id, distance))

        return neighbors

    def a_star_search(self, start_id, goal_id, weight_factor=1.0, priority_factor=100.0):
        """
        A* algoritması ile en iyi yolu bulur

        Parametreler:
        -------------
        start_id : int
            Başlangıç düğümünün ID'si
        goal_id : int
            Hedef düğümünün ID'si
        weight_factor : float
            Ağırlık faktörü (mesafe × ağırlık)
        priority_factor : float
            Öncelik faktörü (öncelik × sabit)

        Dönüş:
        ------
        list
            Düğüm ID'lerinin listesi (yol)
        float
            Yolun toplam maliyeti
        """
        if start_id == goal_id:
            return [start_id], 0

        # Önceki düğümleri saklama
        came_from = {}

        # g skorları (başlangıçtan her düğüme olan maliyet)
        g_score = {node_id: float('inf') for node_id in self.point_id_map}
        g_score[start_id] = 0

        # f skorları (g + tahmin h)
        f_score = {node_id: float('inf') for node_id in self.point_id_map}
        start_pos = self.point_id_map[start_id]
        goal_pos = self.point_id_map[goal_id]
        f_score[start_id] = self.heuristic(start_pos, goal_pos)

        # Öncelik kuyruğu (açık liste)
        open_set = []
        heapq.heappush(open_set, (f_score[start_id], start_id))

        # Ziyaret edilmiş düğümler
        closed_set = set()

        while open_set:
            # En düşük f skorlu düğümü al
            _, current = heapq.heappop(open_set)

            if current in closed_set:
                continue

            if current == goal_id:
                # Yolu yeniden oluştur
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()

                return path, g_score[goal_id]

            closed_set.add(current)

            # Tüm komşuları kontrol et
            for neighbor, distance in self.get_neighbors(current):
                if neighbor in closed_set:
                    continue

                # Teslimat noktası ile ilgili özel maliyetleri hesapla
                weight_cost = 0
                priority_cost = 0

                # Eğer komşu bir teslimat noktasıysa
                if neighbor != 0 and neighbor < len(self.delivery_points) + 1:  # 0 depo
                    delivery_point = self.delivery_points[neighbor - 1]  # -1 çünkü ID 1'den başlıyor
                    weight_cost = delivery_point.weight * weight_factor
                    priority_cost = (6 - delivery_point.priority) * priority_factor  # Öncelik 5 en yüksek

                # Toplam maliyet
                cost = distance + weight_cost + priority_cost

                # Yeni g skoru
                tentative_g_score = g_score[current] + cost

                if tentative_g_score < g_score[neighbor]:
                    # Bu daha iyi bir yol
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = g_score[neighbor] + self.heuristic(self.point_id_map[neighbor], goal_pos)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

        # Yol bulunamadı
        return None, float('inf')

    def find_optimal_route(self, start_id, delivery_ids, return_to_depot=True):
        """
        Birden fazla teslimat noktası için en iyi rotayı bul

        Parametreler:
        -------------
        start_id : int
            Başlangıç düğümünün ID'si (genellikle depo)
        delivery_ids : list
            Teslimat noktalarının ID'lerinin listesi
        return_to_depot : bool
            True ise, rotanın depoya dönmesi sağlanır

        Dönüş:
        ------
        list
            Düğüm ID'lerinin listesi (optimal rota)
        float
            Rotanın toplam maliyeti
        """
        # Tüm teslimat noktaları tamamlanmalı
        remaining = set(delivery_ids)

        # Rota ve maliyet
        route = [start_id]
        total_cost = 0
        current = start_id

        while remaining:
            best_next = None
            best_cost = float('inf')
            best_path = None

            # Her bir kalan teslimat noktasını değerlendirme
            for next_id in remaining:
                path, cost = self.a_star_search(current, next_id)

                if path and cost < best_cost:
                    best_next = next_id
                    best_cost = cost
                    best_path = path

            if best_next is None:
                # Hiçbir teslimat noktasına gidilemez
                break

            # Bu yolu rotaya ekle (ilk eleman zaten rotada olduğu için atla)
            route.extend(best_path[1:])
            total_cost += best_cost

            # Tamamlanan teslimat noktasını kaldır
            remaining.remove(best_next)
            current = best_next

        # Depoya dönüş (eğer isteniyorsa)
        if return_to_depot and current != start_id:
            path, cost = self.a_star_search(current, start_id)
            if path:
                route.extend(path[1:])  # İlk eleman zaten rotada
                total_cost += cost

        return route, total_cost