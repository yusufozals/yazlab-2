"""
Graf yapısı modülü - teslimat noktaları ve bağlantılar
"""
import numpy as np
from typing import List, Dict, Tuple


class DeliveryGraph:
    """
    Teslimat noktalarını düğüm (node), drone hareketlerini kenar (edge) olarak modeller
    Dokümandaki graf yapısına uygun implementasyon
    """

    def __init__(self, delivery_points, no_fly_zones):
        self.delivery_points = delivery_points
        self.no_fly_zones = no_fly_zones

        # Node'ları oluştur (depot + delivery points)
        self.nodes = {}
        self.edges = {}

        self._build_graph()

        print(f"Graf oluşturuldu: {len(self.nodes)} node, {len(self.edges)} edge")

    def _build_graph(self):
        """Graf yapısını oluştur"""
        # Depot node (ID: 0)
        self.nodes[0] = {
            'position': (0, 0),  # Depot pozisyonu main'de ayarlanacak
            'type': 'depot'
        }

        # Delivery point node'ları
        for i, dp in enumerate(self.delivery_points):
            node_id = i + 1
            self.nodes[node_id] = {
                'position': dp.pos,
                'delivery_point': dp,
                'type': 'delivery'
            }

        # Edge'leri oluştur (her node çiftı arası)
        self._build_edges()

    def _build_edges(self):
        """Kenarları oluştur"""
        for from_id in self.nodes:
            self.edges[from_id] = {}
            for to_id in self.nodes:
                if from_id != to_id:
                    cost = self._calculate_edge_cost(from_id, to_id)
                    self.edges[from_id][to_id] = {
                        'cost': cost,
                        'valid': self._is_edge_valid(from_id, to_id)
                    }

    def _calculate_edge_cost(self, from_id: int, to_id: int) -> float:
        """
        Dokümandaki maliyet fonksiyonu:
        Cost(i,j) = distance × weight + (priority × 100)
        """
        from_pos = self.nodes[from_id]['position']
        to_pos = self.nodes[to_id]['position']

        # Temel mesafe
        distance = np.sqrt((from_pos[0] - to_pos[0])**2 + (from_pos[1] - to_pos[1])**2)

        # Hedef delivery point ise weight ve priority ekle
        if to_id != 0 and 'delivery_point' in self.nodes[to_id]:
            dp = self.nodes[to_id]['delivery_point']
            weight = dp.weight
            priority_penalty = (6 - dp.priority) * 100
            return distance * weight + priority_penalty
        else:
            # Depot'a dönüş - sadece mesafe
            return distance

    def _is_edge_valid(self, from_id: int, to_id: int) -> bool:
        """Edge'in geçerli olup olmadığını kontrol et (NFZ vs)"""
        from_pos = self.nodes[from_id]['position']
        to_pos = self.nodes[to_id]['position']

        # NFZ'lerden geçiyor mu?
        for nfz in self.no_fly_zones:
            if nfz.is_active() and nfz.does_line_intersect(from_pos, to_pos):
                return False

        return True

    def get_neighbors(self, node_id: int) -> List[int]:
        """Bir node'un komşularını getir"""
        if node_id in self.edges:
            return [neighbor for neighbor, edge_data in self.edges[node_id].items() 
                   if edge_data['valid']]
        return []

    def get_edge_cost(self, from_id: int, to_id: int) -> float:
        """İki node arası maliyet"""
        if from_id in self.edges and to_id in self.edges[from_id]:
            return self.edges[from_id][to_id]['cost']
        return float('inf')

    def is_edge_valid(self, from_id: int, to_id: int) -> bool:
        """Edge geçerli mi?"""
        if from_id in self.edges and to_id in self.edges[from_id]:
            return self.edges[from_id][to_id]['valid']
        return False

    def update_depot_position(self, depot_position: Tuple[float, float]):
        """Depot pozisyonunu güncelle"""
        self.nodes[0]['position'] = depot_position
        # Edge'leri yeniden hesapla
        self._build_edges()

    def get_node_info(self, node_id: int) -> Dict:
        """Node bilgilerini getir"""
        return self.nodes.get(node_id, {})

    def get_total_nodes(self) -> int:
        """Toplam node sayısı"""
        return len(self.nodes)

    def get_delivery_nodes(self) -> List[int]:
        """Sadece delivery node'larını getir"""
        return [node_id for node_id, node_data in self.nodes.items() 
                if node_data.get('type') == 'delivery']
