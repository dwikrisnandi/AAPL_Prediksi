# -*- coding: utf-8 -*-
"""Copy of AAPL_Prediksi.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1PtoGHYSsALDBBYWLHuUkRzz6dbsn2uI2

#Download data
"""

import json

# Membuat file kaggle.json dengan API token Anda
api_token = {"username":"krisnandi9998","key":"2d15c9eb5dd2bb51786765892d7218d9"}

# Membuat folder kaggle dan menyimpan token API
!mkdir -p ~/.kaggle
with open('/root/.kaggle/kaggle.json', 'w') as file:
    json.dump(api_token, file)

# Mengubah izin akses untuk file API
!chmod 600 ~/.kaggle/kaggle.json

# Verifikasi kredensial
!kaggle datasets list

# Unduh dataset menggunakan API Kaggle
!kaggle datasets download -d krupalpatel07/apple-stock-data
!unzip apple-stock-data.zip -d /content/apple-stock-data

"""#Membaca dan Menampilkan Data"""

# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv('apple-stock-data/AAPL.csv')

# Display first few rows
df.head()

"""5 data awal dari tahun 1980"""

# Display end few rows
df.tail()

"""5 data akhir berada di tahun 2024"""

# Display basic information about the dataset
df.info()

# Display summary statistics
df.describe()

# Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from statsmodels.tsa.arima.model import ARIMA
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

# Loading the dataset
data = pd.read_csv('apple-stock-data/AAPL.csv')
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Plotting the data
plt.figure(figsize=(10,6))
plt.plot(data['Close'])
plt.title('Apple Stock Prices')
plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.show()

"""Pada bagian ini, kita memuat dataset dan menampilkan grafik harga penutupan saham AAPL."""

# Plot the trading volume for AAPL
plt.figure(figsize=(14,8))
plt.plot(df['Date'], df['Volume'], label='Volume', color='orange')
plt.title('Volume Perdagangan Saham Apple (AAPL)', fontsize=16)
plt.xlabel('Tanggal', fontsize=12)
plt.ylabel('Volume Perdagangan', fontsize=12)
plt.legend(loc='upper right')
plt.show()

"""Pada bagian ini, kita memuat dataset dan menampilkan grafik tren distribusi perdagangan saham AAPL.



"""

# Calculate the 50-day and 200-day moving averages
df['MA50'] = df['Close'].rolling(window=50).mean()
df['MA200'] = df['Close'].rolling(window=200).mean()

# Plot closing price with moving averages
plt.figure(figsize=(14,8))
plt.plot(df['Date'], df['Close'], label='Close Price')
plt.plot(df['Date'], df['MA50'], label='50-Day MA', color='green')
plt.plot(df['Date'], df['MA200'], label='200-Day MA', color='red')
plt.title('Tren Harga Penutupan dengan Rata-rata Bergerak untuk Saham Apple', fontsize=16)
plt.xlabel('Tanggal', fontsize=12)
plt.ylabel('Harga Penutupan (USD)', fontsize=12)
plt.legend(loc='upper left')
plt.show()

"""pada bagian ini menghitung dan memplot rata-rata bergerak 50 hari dan 200 hari untuk melihat tren jangka pendek dan jangka panjang. Rata-rata bergerak membantu mengidentifikasi arah tren dan mencegah kebingungan akibat fluktuasi jangka pendek. Jika rata-rata bergerak 50 hari melampaui rata-rata 200 hari, ini bisa menunjukkan potensi tren bullish.

#Praproses Data
"""

# Preprocessing for machine learning models
data_ml = data[['Close']]

# Feature Engineering: Using previous days as features for regression models
data_ml['Close_shifted'] = data_ml['Close'].shift(1)
data_ml.dropna(inplace=True)

# Splitting the data into training and test sets
train, test = train_test_split(data_ml, test_size=0.2, shuffle=False)

# Separate features and labels
X_train, y_train = train[['Close_shifted']], train['Close']
X_test, y_test = test[['Close_shifted']], test['Close']

# Scale data for machine learning models
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

"""Pada tahap ini, kita melakukan praproses pada data, termasuk penggeseran data untuk membuat fitur dan label yang sesuai. Data kemudian dibagi menjadi set pelatihan dan pengujian.

#Linear Regression
"""

# Linear Regression
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_test_scaled)

"""Implementasi model Linear Regression untuk prediksi harga saham.

#Random Forest Regression
"""

# Random Forest Regression
rf = RandomForestRegressor(n_estimators=100)
rf.fit(X_train_scaled, y_train)
y_pred_rf = rf.predict(X_test_scaled)

"""Implementasi model Random Forest untuk prediksi harga saham.

#ARIMA Model
"""

# ARIMA Model
model_arima = ARIMA(train['Close'], order=(1,1,3))
model_arima_fit = model_arima.fit()
y_pred_arima = model_arima_fit.forecast(steps=len(test))

"""Implementasi model ARIMA untuk prediksi deret waktu pada harga saham.

#LSTM Model
"""

from tensorflow.keras.callbacks import EarlyStopping
# LSTM Model Preparation
train_set = data[['Close']].values

# Scaling for LSTM
scaler_lstm = MinMaxScaler()
train_scaled = scaler_lstm.fit_transform(train_set)

# Prepare dataset for LSTM
X_lstm, y_lstm = [], []
for i in range(60, len(train_scaled)):
    X_lstm.append(train_scaled[i-60:i, 0])
    y_lstm.append(train_scaled[i, 0])

X_lstm, y_lstm = np.array(X_lstm), np.array(y_lstm)
X_lstm = np.reshape(X_lstm, (X_lstm.shape[0], X_lstm.shape[1], 1))

# Build LSTM Model
lstm_model = Sequential()
lstm_model.add(LSTM(units=50, return_sequences=True, input_shape=(X_lstm.shape[1], 1)))
lstm_model.add(LSTM(units=50))
lstm_model.add(Dense(units=25))
lstm_model.add(Dense(1))

lstm_model.compile(optimizer='adam', loss='mean_squared_error')
# Add early stopping
early_stop = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)

lstm_model.fit(X_lstm, y_lstm, epochs=50, batch_size=64, callbacks=[early_stop])

# Predicting using LSTM
inputs = data['Close'][len(data)-len(test)-60:].values
inputs = inputs.reshape(-1,1)
inputs = scaler_lstm.transform(inputs)

X_test_lstm = []
for i in range(60, len(inputs)):
    X_test_lstm.append(inputs[i-60:i, 0])

X_test_lstm = np.array(X_test_lstm)
X_test_lstm = np.reshape(X_test_lstm, (X_test_lstm.shape[0], X_test_lstm.shape[1], 1))
y_pred_lstm = lstm_model.predict(X_test_lstm)
y_pred_lstm = scaler_lstm.inverse_transform(y_pred_lstm)

"""Implementasi model LSTM untuk memprediksi harga saham dengan data historis.

#Evaluasi Model
"""

# Evaluating Models (Mean Squared Error)
from sklearn.metrics import mean_squared_error

mse_lr = mean_squared_error(y_test, y_pred_lr)
mse_rf = mean_squared_error(y_test, y_pred_rf)
mse_arima = mean_squared_error(y_test, y_pred_arima)
mse_lstm = mean_squared_error(y_test, y_pred_lstm)

print(f"Linear Regression MSE: {mse_lr}")
print(f"Random Forest MSE: {mse_rf}")
print(f"ARIMA MSE: {mse_arima}")
print(f"LSTM MSE: {mse_lstm}")

from sklearn.metrics import r2_score

# Menghitung R-squared
r2_lr = r2_score(y_test, y_pred_lr)
r2_rf = r2_score(y_test, y_pred_rf)
r2_arima = r2_score(y_test, y_pred_arima)
r2_lstm = r2_score(y_test, y_pred_lstm)

print(f"Linear Regression R-squared: {r2_lr}")
print(f"Random Forest R-squared: {r2_rf}")
print(f"ARIMA R-squared: {r2_arima}")
print(f"LSTM R-squared: {r2_lstm}")

# Fungsi untuk plot prediksi vs harga aktual
def plot_predictions(model_name, actual, predicted, data, training_data_len):
    plt.figure(figsize=(16, 8))
    plt.title(f'Prediksi vs Harga Aktual untuk {model_name}', fontsize=16)
    plt.xlabel('Tanggal', fontsize=12)
    plt.ylabel('Harga Penutupan dalam USD', fontsize=12)
    plt.plot(data.index[training_data_len:], actual, label='Harga Aktual')
    plt.plot(data.index[training_data_len:], predicted, label=f'Prediksi {model_name}', color='red')
    plt.legend(loc='lower right')
    plt.show()

# Assuming training_data_len is calculated somewhere before this part
training_data_len = len(data) - len(test)  # Example: Calculate based on test set length

# Import necessary libraries
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# Calculate MSE for each model
mse_lr = mean_squared_error(y_test, y_pred_lr)
mse_rf = mean_squared_error(y_test, y_pred_rf)
mse_arima = mean_squared_error(y_test, y_pred_arima)
mse_lstm = mean_squared_error(y_test, y_pred_lstm)

# Store MSE values in a dictionary
mse_values = {
    "Linear Regression": mse_lr,
    "Random Forest": mse_rf,
    "ARIMA": mse_arima,
    "LSTM": mse_lstm
}

# Find the model with the lowest MSE
best_model = min(mse_values, key=mse_values.get)

# Print the best model and its MSE
print(f"Model terbaik adalah: {best_model} dengan MSE: {mse_values[best_model]}")

# Plot predictions for LSTM
plot_predictions("LSTM", y_test, y_pred_lstm, data, training_data_len)

# Plot predictions for Linear Regression
plot_predictions("Linear Regression", y_test, y_pred_lr, data, training_data_len)

# Plot predictions for Random Forest
plot_predictions("Random Forest", y_test, y_pred_rf, data, training_data_len)

# Plot predictions for ARIMA
plot_predictions("ARIMA", y_test, y_pred_arima, data, training_data_len)