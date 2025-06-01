import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
from matplotlib.patches import Polygon
import warnings

# FFmpeg uyarılarını bastır
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

class DroneVisualizer:
    """Drone rotalarını görselleştiren sınıf"""

    def __init__(self, drones, delivery_points, no_fly_zones, depot_position):
        self.drones = drones
        self.delivery_points = delivery_points
        self.no_fly_zones = no_fly_zones
        self.depot_position = depot_position
        self.colors = ['blue', 'green', 'red', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']

    def plot_scenario(self, routes, show=True, save_path=None):
        """Senaryo rotalarını çizer"""
        try:
            plt.figure(figsize=(12, 10))

            # Depo
            plt.plot(self.depot_position[0], self.depot_position[1], 
                    'ks', markersize=15, label='Depo', markeredgecolor='white', markeredgewidth=2)

            # Teslimat noktaları
            for dp in self.delivery_points:
                plt.plot(dp.pos[0], dp.pos[1], 'bo', markersize=8, alpha=0.7)
                plt.text(dp.pos[0]+15, dp.pos[1]+15, f'T{dp.id}', 
                        fontsize=8, fontweight='bold')

            # No-fly zones
            for nfz in self.no_fly_zones:
                if nfz.coordinates and len(nfz.coordinates) >= 3:
                    coords = np.array(nfz.coordinates)
                    polygon = Polygon(coords, alpha=0.3, facecolor='red', 
                                    edgecolor='darkred', linewidth=2)
                    plt.gca().add_patch(polygon)

                    # NFZ merkezi için label
                    center_x = np.mean(coords[:, 0])
                    center_y = np.mean(coords[:, 1])
                    plt.text(center_x, center_y, f'NFZ-{nfz.id}', 
                            fontsize=8, ha='center', va='center', 
                            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

            # Rotalar - güvenli çizim
            plotted_routes = 0
            for drone_id, route in routes.items():
                if not route or len(route) < 2:
                    continue

                try:
                    # Route'u güvenli şekilde çiz
                    route_points = []
                    for point in route:
                        if isinstance(point, (list, tuple)) and len(point) >= 2:
                            route_points.append((float(point[0]), float(point[1])))

                    if len(route_points) >= 2:
                        route_x = [p[0] for p in route_points]
                        route_y = [p[1] for p in route_points]

                        color = self.colors[drone_id % len(self.colors)]
                        plt.plot(route_x, route_y, color=color, linewidth=2, 
                                alpha=0.8, marker='o', markersize=4, 
                                label=f'Drone {drone_id}')

                        # Yön okları ekle
                        for i in range(len(route_points) - 1):
                            dx = route_points[i+1][0] - route_points[i][0]
                            dy = route_points[i+1][1] - route_points[i][1]
                            plt.arrow(route_points[i][0], route_points[i][1], 
                                    dx*0.1, dy*0.1, head_width=20, head_length=30, 
                                    fc=color, ec=color, alpha=0.6)

                        plotted_routes += 1

                except Exception as e:
                    print(f"Drone {drone_id} rotası çizilemedi: {e}")
                    continue

            plt.title(f'Drone Teslimat Rotaları ({plotted_routes} rota)')
            plt.xlabel('X Koordinatı (m)')
            plt.ylabel('Y Koordinatı (m)')
            plt.legend(loc='upper right', fontsize=8)
            plt.grid(True, alpha=0.3)
            plt.axis('equal')

            # Grafik sınırlarını ayarla
            all_x = [self.depot_position[0]]
            all_y = [self.depot_position[1]]

            for dp in self.delivery_points:
                all_x.append(dp.pos[0])
                all_y.append(dp.pos[1])

            if all_x and all_y:
                margin = 50
                plt.xlim(min(all_x) - margin, max(all_x) + margin)
                plt.ylim(min(all_y) - margin, max(all_y) + margin)

            if save_path:
                try:
                    plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                               facecolor='white', edgecolor='none')
                    print(f"✅ Görsel kaydedildi: {save_path}")
                except Exception as e:
                    print(f"❌ Görsel kaydedilemedi: {e}")

            if show:
                plt.show()
            else:
                plt.close()

        except Exception as e:
            print(f"Görselleştirme hatası: {e}")
            plt.close()

    def animate_routes(self, routes, output_path=None, fps=2):
        """Rotaları animasyonlu olarak gösterir"""
        print("Animasyon oluşturuluyor...")

        try:
            # Animasyon için basitleştirilmiş implementasyon
            fig, ax = plt.subplots(figsize=(10, 8))

            # Statik elemanları çiz
            ax.plot(self.depot_position[0], self.depot_position[1], 
                   'ks', markersize=15, label='Depo')

            for dp in self.delivery_points:
                ax.plot(dp.pos[0], dp.pos[1], 'bo', markersize=8)
                ax.text(dp.pos[0]+10, dp.pos[1]+10, f'T{dp.id}', fontsize=8)

            # NFZ'ları çiz
            for nfz in self.no_fly_zones:
                if nfz.coordinates and len(nfz.coordinates) >= 3:
                    coords = np.array(nfz.coordinates)
                    polygon = Polygon(coords, alpha=0.3, facecolor='red')
                    ax.add_patch(polygon)

            ax.set_title('Drone Rotaları Animasyonu')
            ax.set_xlabel('X Koordinatı')
            ax.set_ylabel('Y Koordinatı')
            ax.grid(True, alpha=0.3)
            ax.legend()

            # Animasyon yerine statik görsel kaydet
            if output_path:
                try:
                    # MP4 yerine PNG kaydet (FFmpeg problemi için)
                    png_path = output_path.replace('.mp4', '_animation.png')
                    plt.savefig(png_path, dpi=300, bbox_inches='tight')
                    print(f"✅ Animasyon görseli kaydedildi: {png_path}")
                except Exception as e:
                    print(f"❌ Animasyon kaydedilemedi: {e}")

            plt.close()

        except Exception as e:
            print(f"Animasyon hatası: {e}")
            if output_path:
                print(f"Not: {output_path} yerine statik görsel kaydedildi")
