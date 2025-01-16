import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Univariate Analiz

def univariate_analysis(data, numerical_columns, categorical_columns):
    """
    Univariate analiz yapar: histogram ve istatistiksel özetler.
    Args:
        data (pd.DataFrame): Veri çerçevesi.
        numerical_columns (list): Sayısal sütunlar.
        categorical_columns (list): Kategorik sütunlar.
    """
    # Sayısal değişkenler
    for col in numerical_columns:
        print(f"Univariate Analysis for {col}:")
        print(data[col].describe())
        print(f"Skewness: {data[col].skew()}, Kurtosis: {data[col].kurt()}")
        data[col].hist(bins=20)
        plt.title(f"{col} Histogram")
        plt.show()

    # Kategorik değişkenler
    for col in categorical_columns:
        print(f"Value counts for {col}:")
        print(data[col].value_counts())
        data[col].value_counts().plot(kind='bar')
        plt.title(f"{col} Bar Plot")
        plt.show()

# Multivariate Analiz

def multivariate_analysis(data, numerical_columns):
    """
    Multivariate analiz yapar: korelasyon matrisi ve scatter plot.
    Args:
        data (pd.DataFrame): Veri çerçevesi.
        numerical_columns (list): Sayısal sütunlar.
    """
    # Korelasyon matrisi
    correlation_matrix = data[numerical_columns].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
    plt.title("Correlation Matrix")
    plt.show()

    # Scatter plot
    for i, col1 in enumerate(numerical_columns):
        for col2 in numerical_columns[i + 1:]:
            plt.scatter(data[col1], data[col2])
            plt.title(f"{col1} vs {col2}")
            plt.xlabel(col1)
            plt.ylabel(col2)
            plt.show()

# Özellik Mühendisliği

def add_time_features(data, datetime_column):
    """
    Zaman tabanlı özellikler ekler: saat, gün, ay, sezon.
    Args:
        data (pd.DataFrame): Veri çerçevesi.
        datetime_column (str): Tarih/saat sütunu adı.
    Returns:
        pd.DataFrame: Güncellenmiş veri çerçevesi.
    """
    data['hour'] = pd.to_datetime(data[datetime_column]).dt.hour
    data['day'] = pd.to_datetime(data[datetime_column]).dt.day
    data['month'] = pd.to_datetime(data[datetime_column]).dt.month
    data['season'] = data['month'] % 12 // 3 + 1  # 1: Winter, 2: Spring, ...
    return data

def add_moving_avg_and_lags(data, column, window=3):
    """
    Hareketli ortalamalar ve lag özellikleri ekler.
    Args:
        data (pd.DataFrame): Veri çerçevesi.
        column (str): Hedef sütun.
        window (int): Hareketli ortalama için pencere boyutu.
    Returns:
        pd.DataFrame: Güncellenmiş veri çerçevesi.
    """
    data[f'{column}_rolling_mean'] = data[column].rolling(window).mean()
    for lag in range(1, 4):  # 1, 2, 3 gecikme
        data[f'{column}_lag_{lag}'] = data[column].shift(lag)
    return data

# Özellik Seçimi

def feature_importance_analysis(data, target_column):
    """
    Özellik önem sıralaması yapar: korelasyon ve mutual information kullanır.
    Args:
        data (pd.DataFrame): Veri çerçevesi.
        target_column (str): Hedef sütun.
    Returns:
        pd.DataFrame: Özellik önem sıralaması.
    """
    from sklearn.feature_selection import mutual_info_regression
    from sklearn.ensemble import RandomForestRegressor

    X = data.drop(columns=[target_column])
    y = data[target_column]

    # Korelasyon analizi
    correlation_scores = X.corrwith(y).sort_values(ascending=False)
    print("Correlation Scores:")
    print(correlation_scores)

    # Mutual Information
    mi_scores = mutual_info_regression(X, y)
    mi_scores = pd.Series(mi_scores, index=X.columns).sort_values(ascending=False)
    print("Mutual Information Scores:")
    print(mi_scores)

    # Random Forest Importance
    rf = RandomForestRegressor(random_state=42)
    rf.fit(X, y)
    rf_importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    print("Random Forest Feature Importances:")
    print(rf_importances)

# Örnek kullanım
if __name__ == "__main__":
    # Temizlenmiş veriyi yükle
    file_path = r"C:\Users\alpce\OneDrive\Belgeler\GitHub\LFD_24-25\data\raw\cleaned_istanbul_weather.csv"
    cleaned_data = pd.read_csv(file_path)

    # Univariate analiz
    numerical_cols = ['temp', 'prcp', 'rhum', 'wspd']
    categorical_cols = ['coco']
    univariate_analysis(cleaned_data, numerical_cols, categorical_cols)

    # Multivariate analiz
    multivariate_analysis(cleaned_data, numerical_cols)

    # Zaman tabanlı özellikler
    cleaned_data = add_time_features(cleaned_data, 'time')

    # Hareketli ortalama ve lag özellikleri
    cleaned_data = add_moving_avg_and_lags(cleaned_data, 'temp')

    # Özellik seçimi
    feature_importance_analysis(cleaned_data, target_column='temp')
