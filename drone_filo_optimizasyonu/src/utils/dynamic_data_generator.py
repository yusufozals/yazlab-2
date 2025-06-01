import json
import random
import os
import time
import hashlib
from typing import List, Tuple


class DynamicDroneDataGenerator:
    """
    Her çalıştırmada farklı senaryolar üreten dinamik veri üretici
    """

    def __init__(self,
                 num_drones: int = 5,
                 num_delivery_points: int = 20,
                 num_no_fly_zones: int = 3,
                 map_size: Tuple[int, int] = (1000, 1000),
                 depot_position: Tuple[int, int] = None,
                 scenario_name: str = None,
                 use_time_seed: bool = True):
        """
        Args:
            use_time_seed: True ise her seferinde farklı sonuç, False ise aynı sonuç
            scenario_name: Senaryo adı (None ise otomatik)
        """
        self.num_drones = num_drones
        self.num_delivery_points = num_delivery_points
        self.num_no_fly_zones = num_no_fly_zones
        self.map_size = map_size
        self.use_time_seed = use_time_seed

        # Dinamik seed oluştur
        self.seed = self._generate_seed(scenario_name)
        print(f"Veri üretimi seed: {self.seed}")

        # Seed'i ayarla
        random.seed(self.seed)

        # Depo konumunu belirle
        if depot_position is None:
            self.depot_position = (map_size[0] // 2 + random.randint(-50, 50),
                                   map_size[1] // 2 + random.randint(-50, 50))
        else:
            self.depot_position = depot_position

        # Veri listelerini başlat
        self.drones = []
        self.delivery_points = []
        self.no_fly_zones = []

        # Senaryo metadatası
        self.metadata = {
            'generation_time': time.strftime('%Y-%m-%d %H:%M:%S'),
            'seed': self.seed,
            'scenario_name': scenario_name or f"scenario_{self.seed}",
            'complexity_level': self._calculate_complexity()
        }

        # Veri üret
        self._generate_data()

    def _generate_seed(self, scenario_name=None):
        """Dinamik seed üret"""
        if not self.use_time_seed and scenario_name:
            # Sabit seed (aynı senaryo adı = aynı sonuç)
            return abs(hash(scenario_name)) % 10000
        elif self.use_time_seed:
            # Zaman bazlı seed (her seferinde farklı)
            current_time = int(time.time() * 1000000) % 1000000
            return current_time
        else:
            # Rastgele seed
            return random.randint(1000, 99999)

    def _calculate_complexity(self):
        """Senaryo karmaşıklık seviyesini hesapla"""
        complexity_score = (
                self.num_drones * 0.3 +
                self.num_delivery_points * 0.6 +
                self.num_no_fly_zones * 0.1
        )

        if complexity_score < 15:
            return "Kolay"
        elif complexity_score < 30:
            return "Orta"
        elif complexity_score < 50:
            return "Zor"
        else:
            return "Çok Zor"

    def _generate_data(self):
        """Tüm verileri üret"""
        print(f"📊 {self.metadata['scenario_name']} oluşturuluyor...")
        print(f"   Karmaşıklık: {self.metadata['complexity_level']}")
        print(f"   {self.num_drones} drone, {self.num_delivery_points} teslimat, {self.num_no_fly_zones} NFZ")

        self._generate_drones()
        self._generate_delivery_points()
        self._generate_no_fly_zones()

        print(f"✅ Veri üretimi tamamlandı (Seed: {self.seed})")

    def _generate_drones(self):
        """Çeşitli drone tipleri üret"""
        drone_types = [
            {"name": "Hafif", "weight_range": (3, 8), "battery_range": (3000, 6000), "speed_range": (15, 25)},
            {"name": "Orta", "weight_range": (8, 15), "battery_range": (6000, 10000), "speed_range": (12, 20)},
            {"name": "Ağır", "weight_range": (15, 30), "battery_range": (8000, 15000), "speed_range": (8, 15)},
            {"name": "Hızlı", "weight_range": (5, 12), "battery_range": (4000, 8000), "speed_range": (20, 35)},
        ]

        for i in range(self.num_drones):
            # Rastgele drone tipi seç
            drone_type = random.choice(drone_types)

            drone_data = {
                'id': i,
                'max_weight': round(random.uniform(*drone_type["weight_range"]), 1),
                'battery': random.randint(*drone_type["battery_range"]),
                'speed': round(random.uniform(*drone_type["speed_range"]), 1),
                'start_pos': self._get_random_position_near_depot(),
                'type': drone_type["name"]
            }
            self.drones.append(drone_data)

    def _generate_delivery_points(self):
        """Çeşitli teslimat tipleri üret"""
        delivery_types = [
            {"name": "Express", "weight_range": (0.5, 3), "priority": 5, "time_window_hours": 1},
            {"name": "Normal", "weight_range": (2, 8), "priority": 3, "time_window_hours": 3},
            {"name": "Bulk", "weight_range": (5, 15), "priority": 2, "time_window_hours": 6},
            {"name": "Urgent", "weight_range": (1, 5), "priority": 4, "time_window_hours": 2},
        ]

        # Öncelik dağılımını dengele
        priority_distribution = [5] * 3 + [4] * 4 + [3] * 7 + [2] * 4 + [1] * 2  # Toplam 20 için
        if self.num_delivery_points != 20:
            # Farklı sayılar için ölçekle
            priority_distribution = random.choices([1, 2, 3, 4, 5],
                                                   weights=[10, 20, 40, 20, 10],
                                                   k=self.num_delivery_points)

        for i in range(self.num_delivery_points):
            # Delivery type'ı önceliğe göre seç
            priority = priority_distribution[i % len(priority_distribution)]
            matching_types = [dt for dt in delivery_types if dt["priority"] == priority]
            if not matching_types:
                delivery_type = random.choice(delivery_types)
            else:
                delivery_type = random.choice(matching_types)

            # Pozisyon stratejisi - bazıları merkeze yakın, bazıları uzak
            if random.random() < 0.3:  # %30 merkeze yakın
                pos = self._get_random_position_near_depot(radius=200)
            elif random.random() < 0.6:  # %30 orta mesafe
                pos = self._get_random_position_near_depot(radius=400)
            else:  # %40 uzak
                pos = self._get_random_position()

            delivery_data = {
                'id': i,
                'pos': pos,
                'weight': round(random.uniform(*delivery_type["weight_range"]), 1),
                'priority': priority,
                'time_window': self._generate_time_window(delivery_type["time_window_hours"]),
                'type': delivery_type["name"]
            }
            self.delivery_points.append(delivery_data)

    def _generate_no_fly_zones(self):
        """Çeşitli NFZ tipleri üret"""
        nfz_types = [
            {"name": "Havaalanı", "size_range": (100, 200), "complexity": "high"},
            {"name": "Askeri", "size_range": (150, 300), "complexity": "high"},
            {"name": "Hastane", "size_range": (50, 100), "complexity": "low"},
            {"name": "Okul", "size_range": (40, 80), "complexity": "low"},
            {"name": "Sanayi", "size_range": (80, 150), "complexity": "medium"},
        ]

        for i in range(self.num_no_fly_zones):
            nfz_type = random.choice(nfz_types)

            nfz_data = {
                'id': i,
                'coordinates': self._generate_polygon(nfz_type["size_range"]),
                'active_time': self._generate_active_time(),
                'type': nfz_type["name"],
                'risk_level': nfz_type["complexity"]
            }
            self.no_fly_zones.append(nfz_data)

    def _get_random_position(self) -> Tuple[int, int]:
        """Harita üzerinde rastgele pozisyon"""
        x = random.randint(50, self.map_size[0] - 50)
        y = random.randint(50, self.map_size[1] - 50)
        return (x, y)

    def _get_random_position_near_depot(self, radius: int = 150) -> Tuple[int, int]:
        """Depo yakınında rastgele pozisyon"""
        angle = random.uniform(0, 2 * 3.14159)
        distance = random.uniform(20, radius)

        x = int(self.depot_position[0] + distance * random.uniform(-1, 1))
        y = int(self.depot_position[1] + distance * random.uniform(-1, 1))

        # Sınır kontrolü
        x = max(50, min(x, self.map_size[0] - 50))
        y = max(50, min(y, self.map_size[1] - 50))

        return (x, y)

    def _generate_time_window(self, duration_hours: int) -> Tuple[str, str]:
        """Gerçekçi zaman aralığı üret"""
        # İş saatleri ağırlıklı
        working_hours = list(range(8, 18))  # 08:00 - 18:00
        evening_hours = list(range(18, 21))  # 18:00 - 21:00

        # %80 iş saatleri, %20 akşam
        if random.random() < 0.8:
            start_hour = random.choice(working_hours)
        else:
            start_hour = random.choice(evening_hours)

        end_hour = min(start_hour + duration_hours, 21)  # En geç 21:00

        start_minute = random.choice([0, 15, 30, 45])
        end_minute = random.choice([0, 15, 30, 45])

        start_time = f"{start_hour:02d}:{start_minute:02d}"
        end_time = f"{end_hour:02d}:{end_minute:02d}"

        return (start_time, end_time)

    def _generate_active_time(self) -> Tuple[str, str]:
        """NFZ aktif zaman aralığı"""
        scenarios = [
            ("08:00", "17:00"),  # İş saatleri
            ("09:30", "16:30"),  # Okul saatleri
            ("00:00", "23:59"),  # 7/24 aktif
            ("06:00", "22:00"),  # Genişletilmiş
            ("12:00", "14:00"),  # Öğle molası
        ]

        return random.choice(scenarios)

    def _generate_polygon(self, size_range: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Çeşitli şekiller üret"""
        shapes = ["rectangle", "circle", "triangle", "pentagon", "hexagon", "irregular"]
        shape = random.choice(shapes)

        # Merkez nokta
        center_x = random.randint(150, self.map_size[0] - 150)
        center_y = random.randint(150, self.map_size[1] - 150)

        size = random.randint(*size_range)

        if shape == "rectangle":
            return self._generate_rectangle(center_x, center_y, size)
        elif shape == "circle":
            return self._generate_circle(center_x, center_y, size)
        elif shape == "triangle":
            return self._generate_triangle(center_x, center_y, size)
        elif shape == "pentagon":
            return self._generate_regular_polygon(center_x, center_y, size, 5)
        elif shape == "hexagon":
            return self._generate_regular_polygon(center_x, center_y, size, 6)
        else:  # irregular
            return self._generate_irregular_polygon(center_x, center_y, size)

    def _generate_rectangle(self, cx, cy, size):
        """Dikdörtgen üret"""
        w = size + random.randint(-size // 4, size // 4)
        h = size + random.randint(-size // 4, size // 4)

        return [
            (cx - w // 2, cy - h // 2),
            (cx + w // 2, cy - h // 2),
            (cx + w // 2, cy + h // 2),
            (cx - w // 2, cy + h // 2)
        ]

    def _generate_circle(self, cx, cy, size):
        """Çember (çokgen yaklaşımı) üret"""
        return self._generate_regular_polygon(cx, cy, size, 12)

    def _generate_triangle(self, cx, cy, size):
        """Üçgen üret"""
        return self._generate_regular_polygon(cx, cy, size, 3)

    def _generate_regular_polygon(self, cx, cy, size, sides):
        """Düzenli çokgen üret"""
        vertices = []
        for i in range(sides):
            angle = (2 * 3.14159 * i) / sides
            x = int(cx + size * (angle / 3.14159))
            y = int(cy + size * ((angle + 1.5) / 3.14159))
            vertices.append((x, y))
        return vertices

    def _generate_irregular_polygon(self, cx, cy, size):
        """Düzensiz çokgen üret"""
        vertices = []
        sides = random.randint(4, 8)

        for i in range(sides):
            angle = (2 * 3.14159 * i) / sides
            # Rastgele varyasyon ekle
            radius = size * (0.6 + 0.8 * random.random())
            angle_var = angle + random.uniform(-0.3, 0.3)

            x = int(cx + radius * (angle_var / 3.14159))
            y = int(cy + radius * ((angle_var + 1.5) / 3.14159))

            # Sınır kontrolü
            x = max(50, min(x, self.map_size[0] - 50))
            y = max(50, min(y, self.map_size[1] - 50))

            vertices.append((x, y))

        return vertices

    def save_to_file(self, filepath: str):
        """Dosyaya kaydet"""
        print(f"📁 {self.metadata['scenario_name']} kaydediliyor: {filepath}")

        # Klasör oluştur
        directory = os.path.dirname(filepath)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)

        # Veri yapısı
        data = {
            'drones': self.drones,
            'delivery_points': self.delivery_points,
            'no_fly_zones': self.no_fly_zones,
            'depot': self.depot_position,
            'map_size': self.map_size,
            'metadata': self.metadata
        }

        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            print(f"✅ Kaydedildi: {filepath}")
            print(f"   Seed: {self.seed}")
            print(f"   Karmaşıklık: {self.metadata['complexity_level']}")

        except Exception as e:
            print(f"❌ Kaydetme hatası: {e}")


def create_multiple_scenarios():
    """Birden fazla farklı senaryo oluştur"""
    print("🎲 Çoklu Senaryo Üretici")
    print("=" * 30)

    scenarios = [
        {"name": "hizli_test", "drones": 3, "deliveries": 10, "nfz": 1},
        {"name": "standart", "drones": 5, "deliveries": 20, "nfz": 3},
        {"name": "orta_olcek", "drones": 8, "deliveries": 30, "nfz": 4},
        {"name": "buyuk_test", "drones": 10, "deliveries": 50, "nfz": 5},
        {"name": "zorlu", "drones": 12, "deliveries": 75, "nfz": 8},
    ]

    for scenario in scenarios:
        generator = DynamicDroneDataGenerator(
            num_drones=scenario["drones"],
            num_delivery_points=scenario["deliveries"],
            num_no_fly_zones=scenario["nfz"],
            scenario_name=scenario["name"],
            use_time_seed=True  # Her seferinde farklı
        )

        filename = f"data/{scenario['name']}_{int(time.time())}.json"
        generator.save_to_file(filename)
        print()


def create_reproducible_scenarios():
    """Tekrarlanabilir test senaryoları oluştur"""
    print("🔄 Tekrarlanabilir Senaryo Üretici")
    print("=" * 35)

    test_scenarios = [
        {"name": "demo_scenario_1", "drones": 5, "deliveries": 20, "nfz": 2},
        {"name": "demo_scenario_2", "drones": 8, "deliveries": 30, "nfz": 3},
        {"name": "performance_test", "drones": 10, "deliveries": 50, "nfz": 4},
    ]

    for scenario in test_scenarios:
        generator = DynamicDroneDataGenerator(
            num_drones=scenario["drones"],
            num_delivery_points=scenario["deliveries"],
            num_no_fly_zones=scenario["nfz"],
            scenario_name=scenario["name"],
            use_time_seed=False  # Aynı isim = aynı sonuç
        )

        filename = f"data/{scenario['name']}.json"
        generator.save_to_file(filename)
        print()


# Wrapper - eski sınıfla uyumluluk için
class DroneDataGenerator(DynamicDroneDataGenerator):
    """Eski kod uyumluluğu için wrapper"""

    def __init__(self, *args, **kwargs):
        # use_time_seed=True yaparak her seferinde farklı sonuç
        kwargs['use_time_seed'] = kwargs.get('use_time_seed', True)
        super().__init__(*args, **kwargs)


if __name__ == "__main__":
    print("🎯 Dinamik Veri Üretici Test")
    print("=" * 25)

    # Farklı senaryolar oluştur
    create_multiple_scenarios()

    print("\n" + "=" * 50)

    # Tekrarlanabilir senaryolar
    create_reproducible_scenarios()

    print("\n🎉 Tüm senaryolar oluşturuldu!")
    print("📁 data/ klasörünü kontrol edin")