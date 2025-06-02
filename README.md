# Drone Filo Optimizasyonu: Çok Kısıtlı Ortamlarda Dinamik Teslimat Planlaması

## 📋 Proje Özeti

Bu proje, enerji limitleri ve uçuş yasağı bölgeleri (no-fly zone) gibi dinamik kısıtlar altında çalışan drone'lar için en uygun teslimat rotalarının belirlenmesini sağlayan bir algoritma geliştirme çalışmasıdır. Kocaeli Üniversitesi Teknoloji Fakültesi Bilişim Sistemleri Mühendisliği Bölümü TBL331: Yazılım Geliştirme Laboratuvarı II dersi kapsamında geliştirilmiştir.

### 🎯 Temel Özellikler

- **Çok Algoritmik Yaklaşım**: A*, CSP (Constraint Satisfaction Problem) ve Genetik Algoritma implementasyonu
- **Dinamik Kısıtlar**: Gerçek zamanlı uçuş yasağı bölgeleri ve enerji kısıtları
- **Görselleştirme**: Matplotlib tabanlı rota haritaları ve performans grafikleri
- **CUDA Desteği**: GPU hızlandırmalı hesaplamalar (opsiyonel)
- **Dinamik Veri Üretimi**: Her çalıştırmada farklı senaryolar
- **Kapsamlı Test**: Batch test modu ile çoklu senaryo analizi

## 🛠️ Geliştirme Ortamı

### Sistem Gereksinimleri
- **Python**: 3.8 veya üzeri
- **İşletim Sistemi**: Windows, macOS, Linux
- **RAM**: Minimum 4GB (8GB önerilir)
- **GPU**: CUDA destekli GPU (opsiyonel, performans için)

### Gerekli Kütüphaneler
```bash
numpy>=1.21.0
matplotlib>=3.5.0
scipy>=1.7.0
numba>=0.56.0          # CUDA desteği için
cupy>=10.0.0           # GPU hesaplama için (opsiyonel)
```

## 🚀 Kurulum ve Çalıştırma

### 1. Projeyi İndirin
```bash
git clone https://github.com/[kullanici-adi]/drone-filo-optimizasyonu.git
cd drone-filo-optimizasyonu
```

### 2. Gerekli Paketleri Yükleyin
```bash
pip install -r requirements.txt
```

### 3. Proje Yapısını Kontrol Edin
```
drone-filo-optimizasyonu/
├── src/
│   ├── models/              # Veri modelleri
│   ├── algorithms/          # Optimizasyon algoritmaları
│   ├── utils/              # Yardımcı araçlar
│   └── cuda/               # GPU hesaplama (opsiyonel)
├── data/                   # Veri dosyaları
├── visualizations/         # Çıktı görselleri
├── logs/                   # Log dosyaları
└── main.py                 # Ana çalıştırma dosyası
```

### 4. Projeyi Çalıştırın

#### Basit Kullanım
```bash
# Tüm algoritmaları çalıştır
python main.py --algorithm all

# Sadece A* algoritması
python main.py --algorithm astar

# Yeni veri oluştur ve test et
python main.py --generate --algorithm all
```

#### Gelişmiş Kullanım
```bash
# Dinamik senaryolar oluştur
python main.py --generate-dynamic

# Performans analizi
python main.py --analyze --data data/drone_data.json

# Batch test (çoklu senaryo)
python main.py --batch-test

# Özel senaryo
python main.py --scenario scenario2 --algorithm ga
```

## 📊 Algoritmalar

### 1. A* Algoritması
- **Amaç**: En kısa rota bulma
- **Özellik**: Heuristik tabanlı optimal çözüm
- **Kullanım**: Deterministik problemler için ideal

### 2. CSP (Constraint Satisfaction Problem)
- **Amaç**: Kısıt tabanlı atama problemi
- **Özellik**: Drone kapasitesi ve zaman penceresi kısıtları
- **Kullanım**: Karmaşık kısıtlı problemler için

### 3. Genetik Algoritma
- **Amaç**: Evrimsel optimizasyon
- **Özellik**: Çaprazlama ve mutasyon operatörleri
- **Kullanım**: Büyük çözüm uzayında global optimum arama

## 🎮 Geliştirilen Arayüz Örneği

### Ana Çalıştırma Ekranı
```
=== Drone Filo Optimizasyonu (Dinamik Sürüm) ===
📊 Yüklenen veri: 5 drone, 20 teslimat, 3 NFZ

🔍 A* algoritması...
A* algoritması tamamlandı!
Çalışma süresi: 0.0234 saniye
Toplam maliyet: 847.23

🧩 CSP algoritması...
CSP algoritması tamamlandı!
Çalışma süresi: 0.1456 saniye
Toplam maliyet: 923.67

🧬 Genetik algoritma...
Genetik algoritma tamamlandı!
Çalışma süresi: 2.3421 saniye
En iyi fitness: 1247.89
```

### Görselleştirme Örnekleri

#### Rota Haritası
![Drone Rotaları](visualizations/example_routes.png)
*Renkli drone rotaları, teslimat noktaları ve no-fly zone'lar*

#### Performans Grafikleri
![Performans Analizi](visualizations/performance_comparison.png)
*Algoritmaların çalışma süresi karşılaştırması*

## 📈 Test Senaryoları

### Scenario 1: Günlük Operasyon
- **Drone Sayısı**: 5
- **Teslimat Noktası**: 20
- **No-Fly Zone**: 2
- **Hedef**: < 1 dakika çalışma süresi

### Scenario 2: Yoğun Trafik
- **Drone Sayısı**: 10
- **Teslimat Noktası**: 50
- **No-Fly Zone**: 5 (dinamik)
- **Hedef**: %90+ teslimat başarısı

### Dinamik Test
```bash
# Her seferinde farklı rastgele senaryo
python main.py --scenario random --generate
```

## 🔧 Yapılandırma

### Drone Özellikleri
```python
drone = {
    'id': 0,                    # Benzersiz kimlik
    'max_weight': 10.5,         # Maksimum taşıma (kg)
    'battery': 6000,            # Batarya kapasitesi (mAh)
    'speed': 20.0,              # Hız (m/s)
    'start_pos': [100, 200]     # Başlangıç koordinatları
}
```

### Teslimat Noktası
```python
delivery_point = {
    'id': 0,                         # Benzersiz kimlik
    'pos': [350, 450],               # Koordinatlar (x, y)
    'weight': 2.5,                   # Paket ağırlığı (kg)
    'priority': 4,                   # Öncelik (1-5)
    'time_window': ["09:00", "11:00"] # Zaman penceresi
}
```

## 🚁 CUDA Optimizasyonu

### GPU Desteği Aktivasyonu
```bash
# CUDA toolkit kurulu ise
pip install cupy-cuda11x  # CUDA 11.x için
pip install numba

# Otomatik GPU tespiti
python main.py --algorithm ga  # GPU varsa otomatik kullanılır
```

### GPU vs CPU Performansı
- **CPU**: Küçük problemler için yeterli
- **GPU**: 50+ teslimat noktası için 5-10x hızlanma

## 📁 Çıktı Dosyaları

### Görselleştirmeler
- `a_star_routes_[timestamp].png` - A* algoritması rotaları
- `csp_routes_[timestamp].png` - CSP algoritması rotaları  
- `ga_routes_[timestamp].png` - Genetik algoritma rotaları
- `performance_comparison_[timestamp].png` - Performans karşılaştırması
- `drone_animation_[timestamp].png` - Animasyonlu rota gösterimi

### Veri Dosyaları
- `drone_data.json` - Ana veri dosyası
- `daily_scenario_[timestamp].json` - Günlük dinamik senaryo
- `challenge_scenario_[timestamp].json` - Zorlu senaryo
- `performance_scenario_[timestamp].json` - Performans test senaryosu

## 🧪 Test ve Hata Ayıklama

### Unit Test Çalıştırma
```bash
python -m pytest tests/ -v
```

### Hata Ayıklama Modu
```bash
# Detaylı loglar ile
python main.py --algorithm all --verbose

# Debug modu
python -c "import main; main.DEBUG = True; main.main()"
```

### Yaygın Sorunlar ve Çözümler

**Problem**: CUDA kütüphaneleri yüklenmiyor
```bash
# Çözüm: CPU modunda çalıştır
export CUDA_VISIBLE_DEVICES=""
python main.py --algorithm all
```

**Problem**: Matplotlib grafikleri gösterilmiyor
```bash
# Çözüm: Backend ayarla
export MPLBACKEND=Agg
python main.py --algorithm all
```

## 📊 Performans Metrikleri

### Değerlendirme Kriterleri
- **Tamamlanan Teslimat Yüzdesi**: Başarılı teslimat oranı
- **Ortalama Enerji Tüketimi**: Drone başına enerji kullanımı
- **Algoritma Çalışma Süresi**: Milisaniye cinsinden süre
- **Toplam Mesafe**: Tüm rotaların toplam uzunluğu
- **Kısıt İhlali**: No-fly zone ve kapasite ihlalleri

### Benchmark Sonuçları
| Algoritma | Süre (ms) | Başarı (%) | Enerji | Mesafe |
|-----------|-----------|------------|---------|---------|
| A*        | 23.4      | 94.2       | 847.2   | 2,340   |
| CSP       | 145.6     | 96.8       | 923.7   | 2,180   |
| GA        | 2,342.1   | 91.5       | 756.4   | 2,890   |

## 🤝 Katkıda Bulunma

### Geliştirme Süreci
1. Fork yapın
2. Feature branch oluşturun (`git checkout -b feature/yeni-ozellik`)
3. Değişikliklerinizi commit edin (`git commit -am 'Yeni özellik eklendi'`)
4. Branch'i push edin (`git push origin feature/yeni-ozellik`)
5. Pull Request oluşturun


## 📄 Lisans

Bu proje MIT lisansı altında dağıtılmaktadır. Detaylar için [LICENSE](LICENSE) dosyasına bakın.

## 👥 Proje Ekibi

**Kocaeli Üniversitesi**  
Teknoloji Fakültesi  
Bilişim Sistemleri Mühendisliği Bölümü  
Yusuf Samet Özal -  Doğukan Kıralı
221307017  -  221307039

