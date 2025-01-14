import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression

# Veri setlerini yükleme
weather_data = pd.read_csv('C:/Users/PC/Documents/GitHub/LFD_24-25/data/raw/weatherdb.csv')
epias_data = pd.read_csv('C:/Users/PC/Documents/GitHub/LFD_24-25/data/raw/smfdb.csv')

# Zaman formatlarını uyumlu hale getirme
weather_data['datetime'] = pd.to_datetime(weather_data['datetime'], errors='coerce')
epias_data['Tarih'] = pd.to_datetime(epias_data['Tarih'], errors='coerce')

# İki veri setini zaman üzerinden birleştirme
combined_data = pd.merge(weather_data, epias_data, on='datetime')

# Hedef değişken (elektrik fiyatları) ve bağımsız değişkenler
X = combined_data.drop(columns=['price', 'datetime'])  # 'price' hedef değişken
y = combined_data['price']

# 1. Random Forest ile Özellik Önem Skoru
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X, y)
feature_importances = pd.Series(rf_model.feature_importances_, index=X.columns)

# 2. Mutual Information (Doğrusal olmayan ilişkileri analiz etmek için)
mi_scores = mutual_info_regression(X, y)
mi_scores = pd.Series(mi_scores, index=X.columns)

# Önemli özellikleri seçme
important_features_rf = feature_importances[feature_importances > 0.01].index.tolist()
important_features_mi = mi_scores[mi_scores > 0.01].index.tolist()

# Önemli sütunları filtreleme
selected_features = list(set(important_features_rf + important_features_mi))
filtered_data = combined_data[selected_features + ['price']]  # Hedef değişkeni ekle

# Filtrelenmiş veriyi kaydetme
filtered_data.to_csv('data/raw/Filtered_Epias_Weather_Data.csv', index=False)
print(f"Filtered data saved to: data/raw/Filtered_Epias_Weather_Data.csv")

# Özellik önem skorlarını görselleştirme
print("\nRandom Forest Feature Importances:\n", feature_importances.sort_values(ascending=False))
print("\nMutual Information Scores:\n", mi_scores.sort_values(ascending=False))
