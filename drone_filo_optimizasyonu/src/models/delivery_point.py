class DeliveryPoint:
    """Teslimat noktası bilgilerini tutan sınıf"""

    def __init__(self, id, pos, weight, priority, time_window):
        self.id = id
        self.pos = tuple(pos) if isinstance(pos, list) else pos  # Tuple'a çevir
        self.weight = weight
        self.priority = priority
        self.time_window = tuple(time_window) if isinstance(time_window, list) else time_window  # Tuple'a çevir

    @classmethod
    def from_dict(cls, data):
        """Dict'ten DeliveryPoint oluştur"""
        return cls(
            data['id'], 
            tuple(data['pos']), 
            data['weight'], 
            data['priority'], 
            tuple(data['time_window'])
        )

    def to_dict(self):
        """DeliveryPoint'i dict'e çevir"""
        return {
            'id': self.id,
            'pos': list(self.pos),
            'weight': self.weight,
            'priority': self.priority,
            'time_window': list(self.time_window)
        }

    def time_window_seconds(self):
        """Zaman penceresini saniyeye çevir"""
        def time_to_seconds(time_str):
            hours, minutes = map(int, time_str.split(':'))
            return hours * 3600 + minutes * 60

        start_seconds = time_to_seconds(self.time_window[0])
        end_seconds = time_to_seconds(self.time_window[1])
        return start_seconds, end_seconds

    def __hash__(self):
        """Hash desteği için"""
        return hash((self.id, self.pos, self.weight, self.priority, self.time_window))

    def __eq__(self, other):
        """Eşitlik kontrolü"""
        if not isinstance(other, DeliveryPoint):
            return False
        return self.id == other.id

    def __repr__(self):
        return f"DeliveryPoint(id={self.id}, pos={self.pos}, weight={self.weight}kg, priority={self.priority})"

    # models/delivery_point.py'da eklenmesi gereken:
    def is_delivery_time_valid(self, current_time):
        start_time = datetime.strptime(self.time_window[0], "%H:%M").time()
        end_time = datetime.strptime(self.time_window[1], "%H:%M").time()
        return start_time <= current_time <= end_time

    # Algoritmalarda kullanılacak:
    def check_time_constraints(self, route, start_time):
        current_time = start_time
        for point in route:
            if hasattr(point, 'time_window'):
                if not point.is_delivery_time_valid(current_time):
                    return False
            # Seyahat süresini ekle
            current_time += self.calculate_travel_time(point)
        return True