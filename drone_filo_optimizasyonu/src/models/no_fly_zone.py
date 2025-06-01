import math
import datetime
class NoFlyZone:
    """Uçuş yasak bölgesi bilgilerini tutan sınıf"""

    def __init__(self, id, coordinates, active_time):
        self.id = id
        self.coordinates = [tuple(coord) if isinstance(coord, list) else coord for coord in coordinates]
        self.active_time = tuple(active_time) if isinstance(active_time, list) else active_time

    @classmethod
    def from_dict(cls, data):
        """Dict'ten NoFlyZone oluştur"""
        return cls(
            data['id'], 
            [tuple(coord) for coord in data['coordinates']], 
            tuple(data['active_time'])
        )

    def to_dict(self):
        """NoFlyZone'u dict'e çevir"""
        return {
            'id': self.id,
            'coordinates': [list(coord) for coord in self.coordinates],
            'active_time': list(self.active_time)
        }

    def does_line_intersect(self, start_pos, end_pos):
        """Bir çizgi segmentinin bu no-fly zone ile kesişip kesişmediğini kontrol eder"""
        # Basitleştirilmiş nokta-in-polygon kontrolü
        # Çizginin orta noktasını kontrol et
        mid_x = (start_pos[0] + end_pos[0]) / 2
        mid_y = (start_pos[1] + end_pos[1]) / 2

        return self.point_in_polygon((mid_x, mid_y))

    def is_active(self, current_time=None):
        """No-fly zone'un aktif olup olmadığını kontrol eder"""
        if current_time is None:
            return True

        try:
            if isinstance(current_time, str):
                hours, minutes = map(int, current_time.split(':'))
                current_seconds = hours * 3600 + minutes * 60
            else:
                current_seconds = current_time

            start_time, end_time = self.active_time
            start_hours, start_minutes = map(int, start_time.split(':'))
            end_hours, end_minutes = map(int, end_time.split(':'))

            start_seconds = start_hours * 3600 + start_minutes * 60
            end_seconds = end_hours * 3600 + end_minutes * 60

            return start_seconds <= current_seconds <= end_seconds

        except Exception:
            return True

    def point_in_polygon(self, point):
        """Bir noktanın polygon içinde olup olmadığını kontrol eder (Ray casting algoritması)"""
        x, y = point
        n = len(self.coordinates)
        inside = False

        p1x, p1y = self.coordinates[0]
        for i in range(1, n + 1):
            p2x, p2y = self.coordinates[i % n]

            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y

        return inside

    def __hash__(self):
        """Hash desteği için"""
        return hash((self.id, tuple(self.coordinates), self.active_time))

    def __eq__(self, other):
        """Eşitlik kontrolü"""
        if not isinstance(other, NoFlyZone):
            return False
        return self.id == other.id

    def __repr__(self):
        return f"NoFlyZone(id={self.id}, vertices={len(self.coordinates)}, active={self.active_time})"

    def is_active_at_time(self, check_time):
        start_time = datetime.strptime(self.active_time[0], "%H:%M").time()
        end_time = datetime.strptime(self.active_time[1], "%H:%M").time()
        return start_time <= check_time <= end_time

    # Algoritmalarda kontrol:
    def check_nfz_time_constraint(self, route, departure_time):
        current_time = departure_time
        for i in range(len(route) - 1):
            for nfz in self.no_fly_zones:
                if nfz.is_active_at_time(current_time):
                    if self.line_intersects_polygon(route[i], route[i + 1], nfz.coordinates):
                        return False
            current_time += self.calculate_travel_time(route[i], route[i + 1])
        return True