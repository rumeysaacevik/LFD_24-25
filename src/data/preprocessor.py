import pandas as pd
import os


def load_data():
    """Veriyi CSV'den yükler."""
    file_path = os.path.join("data", "raw", "istanbul_hava_durumu.csv")
    data = pd.read_csv(file_path)
    print("Veri başarıyla yüklendi.")
    return data


def clean_missing_values(data):
    """Eksik değerleri temizler."""
    # Eksik değerleri ortalama ile doldur
    data.fillna(data.mean(), inplace=True)
    print("Eksik değerler temizlendi.")
    return data


def remove_outliers(data):
    """Mantıksız verileri temizler."""
    # Mantıklı aralıkları belirle
    conditions = (
            (data['temperature'] >= -50) & (data['temperature'] <= 60) &
            (data['wind_speed'] >= 0) & (data['wind_speed'] <= 50) &
            (data['radiation'] >= 0) & (data['radiation'] <= 1500)
    )
    # Mantıksız verileri kaldır
    clean_data = data[conditions]
    print(f"Mantıksız değerler kaldırıldı: {len(data) - len(clean_data)} satır.")
    return clean_data


def preprocess_data():
    """Veriyi temizler ve işlenmiş olarak kaydeder."""
    # Veriyi yükle
    data = load_data()

    # Eksik değerleri temizle
    data = clean_missing_values(data)

    # Mantıksız değerleri kaldır
    data = remove_outliers(data)

    # Tarih sütununu datetime formatına çevir
    data['time'] = pd.to_datetime(data['time'])
    data.sort_values(by='time', inplace=True)

    # İşlenmiş veriyi kaydet
    output_path = os.path.join("data", "processed", "cleaned_weather_data.csv")
    data.to_csv(output_path, index=False)
    print(f"Temizlenmiş veri başarıyla kaydedildi: {output_path}")
