import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


def load_data(data_file):
    """
    Функція для завантаження даних з CSV файлу та підготовки їх до аналізу.
    """
    df = pd.read_csv(data_file)
    df['Month'] = pd.to_datetime(df['Month'])
    df.set_index('Month', inplace=True)
    return df


def decompose_time_series(df):
    """
    Функція для розкладання часового ряду на тренд, сезонність та залишкову складову.
    """
    decomposition = seasonal_decompose(df['Price Cocacola'], model='additive')
    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid
    return trend, seasonal, residual


def plot_decomposition(df, trend, seasonal, residual):
    """
    Функція для візуалізації розкладеного часового ряду на складові: тренд, сезонність та залишкова складова.
    """

    def plot_subplot(data, title, position):
        """
        Функція для побудови підграфіка.
        """
        plt.subplot(position)
        plt.plot(data)
        plt.title(title)

    plot_subplot(df['Price Cocacola'], 'Original', 411)
    plot_subplot(trend, 'Trend', 412)
    plot_subplot(seasonal, 'Seasonality', 413)
    plot_subplot(residual, 'Residuals', 414)

    plt.tight_layout()
    plt.show()


def evaluate_forecast_accuracy(df, forecast):
    """
    Функція для оцінки точності прогнозів за допомогою MSE
    """

    def calculate_mse(df, forecast):
        """
        Функція для обчислення середньоквадратичної помилки (MSE).
        """
        actual_values = df['Price Cocacola'].tail(6)
        mse = mean_squared_error(actual_values, forecast)
        return mse

    mse = calculate_mse(df, forecast)
    mean_production = df['Price Cocacola'].mean()
    print('Mean:', mean_production)
    print('Mean Squared Error:', mse)


def fit_arima_model(df):
    """
    Функція для підгонки моделі ARIMA до часового ряду та отримання прогнозу.
    """
    model_arima = ARIMA(df['Price Cocacola'], order=(5, 1, 0))
    model_fit_arima = model_arima.fit()

    # Отримання прогнозу на певну кількість кроків у майбутньому (у цьому випадку 6 кроків)
    forecast_arima = model_fit_arima.forecast(steps=6)
    print(f'ARIMA Forecast:\n{forecast_arima}')

    return forecast_arima


def fit_sarimax_model(df):
    """
    Функція для підгонки моделі SARIMAX до часового ряду та отримання прогнозу.
    """

    # Створення копії датафрейму
    df_copy = df.copy()

    model_sarimax = SARIMAX(df_copy['Price Cocacola'], order=(1, 1, 1), seasonal_order=(1, 1, 0, 12))
    results_sarimax = model_sarimax.fit()

    # Отримання прогнозу на певний часовий проміжок
    df_copy['SARIMAX Forecast'] = results_sarimax.predict(start=len(df_copy) - 10, end=len(df_copy) + 5, dynamic=True)
    df_copy[['Price Cocacola', 'SARIMAX Forecast']].plot()
    plt.show()


def check_stationarity(df):
    """
    Функція для перевірки стаціонарності часового ряду за допомогою тесту Дікі-Фуллера.

    Результати:
    Augmented Dickey-Fuller Statistic (float): Значення тесту Дікі-Фуллера.
    P-value (float): p-значення тесту Дікі-Фуллера.
    """
    result = adfuller(df['Price Cocacola'])
    print('Augmented Dickey-Fuller Statistic:', result[0])
    print('P-value:', result[1])


def fit_var_model(df):
    """
    Функція для підгонки моделі VAR (векторна авторегресійна модель) до декількох часових рядів.
    """
    df['Another_time_series'] = np.random.normal(0, 1, len(df)).cumsum() + np.cos(np.linspace(0, 10, len(df))) * 5
    model_var = VAR(df[['Price Cocacola', 'Another_time_series']])
    results_var = model_var.fit()
    print(results_var.summary())


def machine_learning(df):
    """
    Функція для прогнозування майбутніх значень часового ряду за допомогою машинного навчання.
    """
    df['trend'] = np.arange(len(df))
    for lag in range(1, 4):
        df[f'lag_{lag}'] = df['Price Cocacola'].shift(lag)

    df.dropna(inplace=True)

    X = df[['trend'] + [f'lag_{lag}' for lag in range(1, 4)]].values
    y = df['Price Cocacola'].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, shuffle=False)

    model_rf = RandomForestRegressor(n_estimators=200)
    model_rf.fit(X_train, y_train)

    predictions_rf = model_rf.predict(X_test)

    plt.figure(figsize=(10, 6))
    plt.plot(df.index[len(df) - len(y_test):], y_test, label='Actual')
    plt.plot(df.index[len(df) - len(y_test):], predictions_rf, label='Predicted')
    plt.legend()
    plt.show()


def analyze_time_series(data_file):
    warnings.filterwarnings('ignore')
    # Завантаження даних з файлу
    df = load_data(data_file)

    # Декомпозиція часового ряду на тренд, сезонність та залишкову складову
    trend, seasonal, residual = decompose_time_series(df)

    # Побудова графіка декомпозиції
    plot_decomposition(df, trend, seasonal, residual)

    # Підгонка моделі ARIMA та прогнозування
    forecast_arima = fit_arima_model(df)

    # Оцінка точності прогнозу моделі ARIMA
    evaluate_forecast_accuracy(df, forecast_arima)

    # Підгонка моделі SARIMA
    fit_sarimax_model(df)

    # Перевірка стаціонарності часового ряду
    check_stationarity(df)

    # Підгонка векторної авторегресійної моделі
    fit_var_model(df)

    # Використання методів машинного навчання для аналізу часового ряду
    machine_learning(df)


analyze_time_series('cocacola-price.csv')
