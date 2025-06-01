class Drone:
    """Drone bilgilerini tutan sınıf"""

    def __init__(self, id, max_weight, battery, speed, start_pos):
        self.id = id
        self.max_weight = max_weight
        self.battery = battery
        self.speed = speed
        self.start_pos = tuple(start_pos) if isinstance(start_pos, list) else start_pos

    @classmethod
    def from_dict(cls, data):
        """Dict'ten Drone oluştur"""
        return cls(
            data['id'],
            data['max_weight'], 
            data['battery'], 
            data['speed'], 
            tuple(data['start_pos'])
        )

    def to_dict(self):
        """Drone'u dict'e çevir"""
        return {
            'id': self.id,
            'max_weight': self.max_weight,
            'battery': self.battery,
            'speed': self.speed,
            'start_pos': list(self.start_pos)
        }

    def __hash__(self):
        """Hash desteği için"""
        return hash((self.id, self.max_weight, self.battery, self.speed, self.start_pos))

    def __eq__(self, other):
        """Eşitlik kontrolü"""
        if not isinstance(other, Drone):
            return False
        return self.id == other.id

    def __repr__(self):
        return f"Drone(id={self.id}, capacity={self.max_weight}kg, battery={self.battery}mAh)"
