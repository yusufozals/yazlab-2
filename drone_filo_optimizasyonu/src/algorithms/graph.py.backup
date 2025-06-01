import numpy as np
import networkx as nx
from src.models.delivery_point import DeliveryPoint
from src.models.no_fly_zone import NoFlyZone


class DeliveryGraph:
    def __init__(self, delivery_points, no_fly_zones, depot_position):
        """
        Teslimat grafiği sınıfı

        Parametreler:
        -------------
        delivery_points : list
            DeliveryPoint nesnelerinin listesi
        no_fly_zones : list
            NoFlyZone nesnelerinin listesi
        depot_position : tuple
            Depo konumu (x, y)
        """
        self.delivery_points = delivery_points
        self.no_fly_zones = no_fly_zones
        self.depot_position = depot_position
        self.graph = nx.DiGraph()  # Yönlü graf

        # Düğümleri oluştur
        self._build_nodes()

        # Kenarları oluştur
        self._build_edges()

    def _build_nodes(self):
        """Graftaki düğümleri oluşturur"""
        # Depoyu ekle (ID: 0)
        self.graph.add_node(0, pos=self.depot_position, is_depot=True, is_delivery=False)

        # Teslimat noktalarını ekle
        for dp in self.delivery_points:
            self.graph.add_node(
                dp.id + 1,  # Depo 0, teslimatlar 1'den başlar
                pos=dp.pos,
                weight=dp.weight,
                priority=dp.priority,
                time_window=dp.time_window,
                is_depot=False,
                is_delivery=True
            )

    def _build_edges(self):
        """Graftaki kenarları oluşturur"""
        nodes = list(self.graph.nodes())

        for i, node1 in enumerate(nodes):
            for j, node2 in enumerate(nodes):
                if i == j:
                    continue  # Aynı düğüm

                pos1 = self.graph.nodes[node1]['pos']
                pos2 = self.graph.nodes[node2]['pos']

                # İki düğüm arasındaki mesafeyi hesapla
                distance = np.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)

                # Yol bir no-fly zone içinden geçiyor mu kontrol et
                path_blocked = False
                for nfz in self.no_fly_zones:
                    if nfz.does_line_intersect(pos1, pos2):
                        path_blocked = True
                        break

                # Eğer yol engellemiyorsa kenarı ekle
                if not path_blocked:
                    # Eğer node2 bir teslimat noktasıysa, önceliğe dayalı bir maliyet hesapla
                    if self.graph.nodes[node2]['is_delivery']:
                        priority = self.graph.nodes[node2]['priority']
                        weight = self.graph.nodes[node2]['weight']

                        # Maliyet fonksiyonu: mesafe × ağırlık + (önceliğe göre ceza)
                        cost = distance * weight + (6 - priority) * 100
                    else:
                        # Depoya dönüş için sadece mesafe
                        cost = distance

                    # Kenarı ekle
                    self.graph.add_edge(node1, node2, distance=distance, cost=cost)

    def get_graph(self):
        """NetworkX grafını döndürür"""
        return self.graph

    def get_shortest_path(self, start_node, end_node, algorithm='dijkstra'):
        """İki düğüm arasındaki en kısa yolu bulur"""
        if algorithm == 'dijkstra':
            return nx.dijkstra_path(self.graph, start_node, end_node, weight='cost')
        elif algorithm == 'astar':
            return nx.astar_path(self.graph, start_node, end_node, weight='cost')
        else:
            raise ValueError(f"Bilinmeyen algoritma: {algorithm}")

    def get_all_deliveries_from_depot(self, algorithm='dijkstra'):
        """Depodan tüm teslimat noktalarına en kısa yolları bulur"""
        paths = {}
        for node in self.graph.nodes():
            if node != 0 and self.graph.nodes[node]['is_delivery']:
                try:
                    paths[node] = self.get_shortest_path(0, node, algorithm)
                except nx.NetworkXNoPath:
                    # Bu düğüme yol yok
                    paths[node] = None
        return paths

    def calculate_route_cost(self, route):
        """Bir rotanın toplam maliyetini hesaplar"""
        total_cost = 0
        for i in range(len(route) - 1):
            if self.graph.has_edge(route[i], route[i + 1]):
                total_cost += self.graph[route[i]][route[i + 1]]['cost']
            else:
                return float('inf')  # Geçersiz yol
        return total_cost

    def get_nearest_neighbors(self, node, k=5):
        """Bir düğümün en yakın k komşusunu bulur"""
        distances = []
        for neighbor in self.graph.nodes():
            if neighbor != node and self.graph.has_edge(node, neighbor):
                distances.append((neighbor, self.graph[node][neighbor]['distance']))

        # Mesafeye göre sırala ve en yakın k tanesini döndür
        distances.sort(key=lambda x: x[1])
        return distances[:k]