import json
import random
import os
from typing import List, Tuple


class DroneDataGenerator:
    """Drone, teslimat noktası ve no-fly zone verilerini rastgele üretir"""

    def __init__(self,
                 num_drones: int = 5,
                 num_delivery_points: int = 20,
                 num_no_fly_zones: int = 3,
                 map_size: Tuple[int, int] = (1000, 1000),
                 depot_position: Tuple[int, int] = None):
        """
        Args:
            num_drones: Üretilecek drone sayısı
            num_delivery_points: Üretilecek teslimat noktası sayısı
            num_no_fly_zones: Üretilecek no-fly zone sayısı
            map_size: Harita boyutu (genişlik, yükseklik)
            depot_position: Depo konumu, None ise merkeze yerleştirilir
        """
        self.num_drones = num_drones
        self.num_delivery_points = num_delivery_points
        self.num_no_fly_zones = num_no_fly_zones
        self.map_size = map_size

        # Depo konumunu belirle
        if depot_position is None:
            self.depot_position = (map_size[0] // 2, map_size[1] // 2)
        else:
            self.depot_position = depot_position

        # Veri listelerini başlat
        self.drones = []
        self.delivery_points = []
        self.no_fly_zones = []

        # Veri üret
        self._generate_data()

    def _generate_data(self):
        """Tüm verileri üretir"""
        print(f"Veri üretimi başlıyor...")
        print(f"- {self.num_drones} drone")
        print(f"- {self.num_delivery_points} teslimat noktası")
        print(f"- {self.num_no_fly_zones} no-fly zone")
        print(f"- Harita boyutu: {self.map_size}")

        self._generate_drones()
        self._generate_delivery_points()
        self._generate_no_fly_zones()

        print("Veri üretimi tamamlandı!")

    def _generate_drones(self):
        """Drone verilerini üretir"""
        for i in range(self.num_drones):
            # Drone özellikleri
            drone_data = {
                'id': i,
                'max_weight': round(random.uniform(5.0, 25.0), 1),  # 5-25 kg
                'battery': random.randint(5000, 15000),  # 5000-15000 mAh
                'speed': round(random.uniform(10.0, 30.0), 1),  # 10-30 m/s
                'start_pos': self._get_random_position_near_depot()
            }
            self.drones.append(drone_data)

    def _generate_delivery_points(self):
        """Teslimat noktası verilerini üretir"""
        for i in range(self.num_delivery_points):
            # Teslimat noktası özellikleri
            delivery_data = {
                'id': i,
                'pos': self._get_random_position(),
                'weight': round(random.uniform(0.5, 10.0), 1),  # 0.5-10 kg
                'priority': random.randint(1, 5),  # 1-5 arası öncelik
                'time_window': self._generate_time_window()
            }
            self.delivery_points.append(delivery_data)

    def _generate_no_fly_zones(self):
        """No-fly zone verilerini üretir"""
        for i in range(self.num_no_fly_zones):
            # No-fly zone özellikleri
            nfz_data = {
                'id': i,
                'coordinates': self._generate_polygon(),
                'active_time': self._generate_active_time()
            }
            self.no_fly_zones.append(nfz_data)

    def _get_random_position(self) -> Tuple[int, int]:
        """Harita üzerinde rastgele bir pozisyon döndürür"""
        x = random.randint(0, self.map_size[0])
        y = random.randint(0, self.map_size[1])
        return (x, y)

    def _get_random_position_near_depot(self, radius: int = 100) -> Tuple[int, int]:
        """Depo yakınında rastgele bir pozisyon döndürür"""
        angle = random.uniform(0, 2 * 3.14159)  # 0-2π radyan
        distance = random.uniform(0, radius)

        x = int(self.depot_position[0] + distance * random.uniform(-1, 1))
        y = int(self.depot_position[1] + distance * random.uniform(-1, 1))

        # Harita sınırlarını kontrol et
        x = max(0, min(x, self.map_size[0]))
        y = max(0, min(y, self.map_size[1]))

        return (x, y)

    def _generate_time_window(self) -> Tuple[str, str]:
        """Rastgele zaman aralığı üretir"""
        start_hour = random.randint(8, 16)  # 08:00 - 16:00 arası başlangıç
        duration = random.randint(1, 4)  # 1-4 saat arası süre
        end_hour = min(start_hour + duration, 18)  # En geç 18:00

        start_minute = random.choice([0, 15, 30, 45])
        end_minute = random.choice([0, 15, 30, 45])

        start_time = f"{start_hour:02d}:{start_minute:02d}"
        end_time = f"{end_hour:02d}:{end_minute:02d}"

        return (start_time, end_time)

    def _generate_active_time(self) -> Tuple[str, str]:
        """No-fly zone için aktif zaman aralığı üretir"""
        start_hour = random.randint(9, 15)
        duration = random.randint(2, 6)  # 2-6 saat arası
        end_hour = min(start_hour + duration, 17)

        start_time = f"{start_hour:02d}:{random.choice([0, 30]):02d}"
        end_time = f"{end_hour:02d}:{random.choice([0, 30]):02d}"

        return (start_time, end_time)

    def _generate_polygon(self) -> List[Tuple[int, int]]:
        """Rastgele çokgen (no-fly zone) üretir"""
        # Merkez nokta seç
        center_x = random.randint(100, self.map_size[0] - 100)
        center_y = random.randint(100, self.map_size[1] - 100)

        # Çokgen boyutu
        size = random.randint(50, 150)

        # 4-8 köşeli çokgen
        num_vertices = random.randint(4, 8)
        vertices = []

        for i in range(num_vertices):
            angle = (2 * 3.14159 * i) / num_vertices
            # Biraz rastgelelik ekle
            radius = size * (0.7 + 0.6 * random.random())

            x = int(center_x + radius * (angle / 3.14159))
            y = int(center_y + radius * ((angle + 1.5) / 3.14159))

            # Sınır kontrolü
            x = max(0, min(x, self.map_size[0]))
            y = max(0, min(y, self.map_size[1]))

            vertices.append((x, y))

        return vertices

    def save_to_file(self, filepath: str):
        """Verileri JSON dosyasına kaydet"""
        print(f"Veriler kaydediliyor: {filepath}")

        # Klasör yoksa oluştur
        directory = os.path.dirname(filepath)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            print(f"Klasör oluşturuldu: {directory}")

        # Veri yapısını oluştur
        data = {
            'drones': self.drones,
            'delivery_points': self.delivery_points,
            'no_fly_zones': self.no_fly_zones,
            'depot': self.depot_position,
            'map_size': self.map_size,
            'metadata': {
                'generated_by': 'DroneDataGenerator',
                'num_drones': self.num_drones,
                'num_delivery_points': self.num_delivery_points,
                'num_no_fly_zones': self.num_no_fly_zones
            }
        }

        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"Veri dosyası başarıyla kaydedildi: {filepath}")
        except Exception as e:
            print(f"Dosya kaydetme hatası: {e}")
            raise

    def load_from_file(self, filepath: str):
        """JSON dosyasından verileri yükle"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)

            self.drones = data.get('drones', [])
            self.delivery_points = data.get('delivery_points', [])
            self.no_fly_zones = data.get('no_fly_zones', [])
            self.depot_position = tuple(data.get('depot', (500, 500)))
            self.map_size = tuple(data.get('map_size', (1000, 1000)))

            print(f"Veri dosyası başarıyla yüklendi: {filepath}")
            return True

        except FileNotFoundError:
            print(f"Dosya bulunamadı: {filepath}")
            return False
        except Exception as e:
            print(f"Dosya yükleme hatası: {e}")
            return False

    def get_summary(self) -> str:
        """Veri özetini döndürür"""
        summary = f"""
=== Veri Özeti ===
Drone sayısı: {len(self.drones)}
Teslimat noktası sayısı: {len(self.delivery_points)}
No-fly zone sayısı: {len(self.no_fly_zones)}
Depo konumu: {self.depot_position}
Harita boyutu: {self.map_size}

Drone kapasiteleri: {[d['max_weight'] for d in self.drones]} kg
Teslimat ağırlıkları: {[dp['weight'] for dp in self.delivery_points]} kg
Teslimat öncelikleri: {[dp['priority'] for dp in self.delivery_points]}
"""
        return summary

    def validate_data(self) -> bool:
        """Veri geçerliliğini kontrol eder"""
        errors = []

        # Drone kontrolü
        if not self.drones:
            errors.append("Drone verisi bulunamadı")

        for drone in self.drones:
            if drone['max_weight'] <= 0:
                errors.append(f"Drone {drone['id']} geçersiz ağırlık kapasitesi")
            if drone['battery'] <= 0:
                errors.append(f"Drone {drone['id']} geçersiz batarya kapasitesi")

        # Teslimat noktası kontrolü
        if not self.delivery_points:
            errors.append("Teslimat noktası verisi bulunamadı")

        for dp in self.delivery_points:
            if dp['weight'] <= 0:
                errors.append(f"Teslimat {dp['id']} geçersiz ağırlık")
            if not (1 <= dp['priority'] <= 5):
                errors.append(f"Teslimat {dp['id']} geçersiz öncelik")

        # No-fly zone kontrolü
        for nfz in self.no_fly_zones:
            if len(nfz['coordinates']) < 3:
                errors.append(f"No-fly zone {nfz['id']} yeterli köşe sayısı yok")

        if errors:
            print("Veri geçerlilik hataları:")
            for error in errors:
                print(f"- {error}")
            return False

        print("Veri geçerlilik kontrolü başarılı!")
        return True


def create_test_scenarios():
    """Test senaryolarını oluşturur"""
    scenarios = [
        {
            'name': 'Küçük Test',
            'params': {
                'num_drones': 3,
                'num_delivery_points': 10,
                'num_no_fly_zones': 1,
                'map_size': (500, 500)
            }
        },
        {
            'name': 'Senaryo 1',
            'params': {
                'num_drones': 5,
                'num_delivery_points': 20,
                'num_no_fly_zones': 2,
                'map_size': (1000, 1000)
            }
        },
        {
            'name': 'Senaryo 2',
            'params': {
                'num_drones': 10,
                'num_delivery_points': 50,
                'num_no_fly_zones': 5,
                'map_size': (1500, 1500)
            }
        },
        {
            'name': 'Büyük Test',
            'params': {
                'num_drones': 15,
                'num_delivery_points': 100,
                'num_no_fly_zones': 8,
                'map_size': (2000, 2000)
            }
        }
    ]

    return scenarios


def main():
    """Ana fonksiyon - test amaçlı"""
    print("=== Drone Veri Üretici ===")

    # Test senaryolarını oluştur
    scenarios = create_test_scenarios()

    for scenario in scenarios:
        print(f"\n{scenario['name']} oluşturuluyor...")

        # Veri üretici oluştur
        generator = DroneDataGenerator(**scenario['params'])

        # Veri geçerliliğini kontrol et
        if generator.validate_data():
            # Dosya adını oluştur
            filename = f"data/{scenario['name'].lower().replace(' ', '_')}.json"

            # Dosyayı kaydet
            generator.save_to_file(filename)

            # Özet yazdır
            print(generator.get_summary())
        else:
            print(f"{scenario['name']} geçerlilik kontrolünde başarısız!")


if __name__ == "__main__":
    main()