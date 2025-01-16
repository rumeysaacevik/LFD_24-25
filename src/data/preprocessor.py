import pandas as pd
import numpy as np

def load_data(file_path):
    """
    Veriyi verilen dosya yolundan yükler.
    Args:
        file_path (str): CSV dosyasının yolu.
    Returns:
        pd.DataFrame: Yüklenmiş veri çerçevesi.
    """
    try:
        data = pd.read_csv(file_path)
        print(f"Data successfully loaded from {file_path}. Shape: {data.shape}")
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None


def convert_to_datetime(data, column_name):
    """
    Belirtilen sütunu datetime formatına dönüştürür.
    Args:
        data (pd.DataFrame): Veri çerçevesi.
        column_name (str): Tarih/saat sütunu adı.
    Returns:
        pd.DataFrame: Güncellenmiş veri çerçevesi.
    """
    try:
        data[column_name] = pd.to_datetime(data[column_name])
        print(f"{column_name} column converted to datetime format.")
        return data
    except Exception as e:
        print(f"Error converting to datetime: {e}")
        return None


def check_missing_values(data):
    """
    Eksik değerleri kontrol eder ve özetler.
    Args:
        data (pd.DataFrame): Veri çerçevesi.
    Returns:
        None
    """
    missing = data.isnull().sum()
    print("Missing values per column:")
    print(missing)


def fill_missing_values(data, strategy='mean'):
    """
    Eksik değerleri belirtilen yöntemle doldurur.
    Args:
        data (pd.DataFrame): Veri çerçevesi.
        strategy (str): Doldurma stratejisi ('mean', 'median', 'mode').
    Returns:
        pd.DataFrame: Eksik değerleri doldurulmuş veri çerçevesi.
    """
    try:
        if strategy == 'mean':
            for column in data.select_dtypes(include=[np.number]).columns:
                data[column] = data[column].fillna(data[column].mean())
        elif strategy == 'median':
            for column in data.select_dtypes(include=[np.number]).columns:
                data[column] = data[column].fillna(data[column].median())
        elif strategy == 'mode':
            for column in data.columns:
                data[column] = data[column].fillna(data[column].mode()[0])
        else:
            raise ValueError(f"Unsupported strategy: {strategy}")

        print(f"Missing values filled using {strategy} strategy.")
        return data
    except Exception as e:
        print(f"Error filling missing values: {e}")
        return None


def drop_duplicates(data):
    """
    Veri çerçevesindeki tekrar eden satırları kaldırır.
    Args:
        data (pd.DataFrame): Veri çerçevesi.
    Returns:
        pd.DataFrame: Güncellenmiş veri çerçevesi.
    """
    try:
        before = data.shape[0]
        data = data.drop_duplicates()
        after = data.shape[0]
        print(f"Removed {before - after} duplicate rows.")
        return data
    except Exception as e:
        print(f"Error dropping duplicates: {e}")
        return None


def remove_columns_with_missing_values(data, threshold=0.5):
    """
    Eksik değer oranı belirtilen eşiğin üzerinde olan sütunları kaldırır.
    Args:
        data (pd.DataFrame): Veri çerçevesi.
        threshold (float): Eksik değer oranı eşiği (ör. 0.5 %50 demektir).
    Returns:
        pd.DataFrame: Eksik sütunlar kaldırılmış veri çerçevesi.
    """
    try:
        missing_ratio = data.isnull().mean()  # Eksik değer oranları
        columns_to_drop = missing_ratio[missing_ratio > threshold].index
        print(f"Columns removed due to missing values: {list(columns_to_drop)}")
        data = data.drop(columns=columns_to_drop)
        return data
    except Exception as e:
        print(f"Error removing columns with missing values: {e}")
        return None


def clean_data(file_path, datetime_column, fill_strategy='mean', threshold=None):
    """
    Veriyi temizleme işlemlerini birleştirir.
    Args:
        file_path (str): Veri dosyasının yolu.
        datetime_column (str): Tarih/saat sütunu adı.
        fill_strategy (str): Eksik değer doldurma stratejisi ('mean', 'median', 'mode').
        threshold (float): Eksik değer oranı eşiği. Belirtilirse, bu oranı aşan sütunlar kaldırılır.
    Returns:
        pd.DataFrame: Temizlenmiş veri çerçevesi.
    """
    # Veri yükleme
    data = load_data(file_path)

    if data is not None:
        # Tarih formatına dönüştürme
        data = convert_to_datetime(data, datetime_column)

        # Eksik değerleri kontrol etme
        check_missing_values(data)

        # Eksik sütunları kaldırma (threshold belirtilirse)
        if threshold is not None:
            missing_ratio = data.isnull().mean()  # Eksik değer oranları
            columns_to_drop = missing_ratio[missing_ratio > threshold].index
            print(f"Columns removed due to missing values: {list(columns_to_drop)}")
            data = data.drop(columns=columns_to_drop)

        # Eksik değerleri doldurma
        data = fill_missing_values(data, strategy=fill_strategy)

        # Tekrar eden satırları kaldırma
        data = drop_duplicates(data)

        print("Data cleaning completed.")
        return data
    else:
        print("Data loading failed. Cleaning skipped.")
        return None

# Temizlenmiş veriyi kaydetme
def save_cleaned_data(data, file_path):
    """
    Temizlenmiş veriyi belirtilen dosya yoluna kaydeder.
    Args:
        data (pd.DataFrame): Temizlenmiş veri çerçevesi.
        file_path (str): Kaydedilecek dosyanın yolu.
    """
    try:
        data.to_csv(file_path, index=False)
    except Exception as e:
        print(f"Error saving cleaned data: {e}")

# Örnek kullanım
if __name__ == "__main__":
    file_path = r"C:\Users\alpce\OneDrive\Belgeler\GitHub\LFD_24-25\data\raw\istanbul_hava_durumu.csv"
    datetime_column = "time"
    output_path = r"C:\Users\alpce\OneDrive\Belgeler\GitHub\LFD_24-25\data\raw\cleaned_istanbul_weather.csv"

    # Veriyi temizle
    cleaned_data = clean_data(file_path, datetime_column, fill_strategy='mean', threshold=0.5)

    # Temizlenmiş veriyi kaydet
    if cleaned_data is not None:
        save_cleaned_data(cleaned_data, output_path)

