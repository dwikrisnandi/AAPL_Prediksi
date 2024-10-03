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

"""Kode ini digunakan untuk mengonfigurasi API Kaggle dan mengunduh dataset. Pertama, modul json diimpor untuk menangani data JSON. Kemudian, token API yang berisi username dan key disimpan dalam file kaggle.json di folder .kaggle. Izin akses untuk file tersebut diubah agar hanya pemilik yang dapat membacanya. Setelah itu, kredensial diverifikasi dengan menjalankan perintah untuk menampilkan daftar dataset yang tersedia. Selanjutnya, dataset yang diinginkan diunduh menggunakan API Kaggle, dan akhirnya, file ZIP yang diunduh diekstrak ke folder yang ditentukan. Pastikan untuk mengganti username dan key dengan kredensial API Anda sendiri dan memastikan paket kaggle telah terinstal.

#Membaca dan Menampilkan Data
"""

# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv('apple-stock-data/AAPL.csv')

# Display first few rows
df.head()

"""Kode ini digunakan untuk memuat dan menampilkan dataset yang berisi data saham Apple. Pertama, beberapa pustaka diimpor, yaitu pandas untuk manipulasi data, matplotlib.pyplot untuk visualisasi, dan seaborn untuk membuat grafik yang lebih menarik. Selanjutnya, dataset dimuat dari file CSV yang terletak di folder apple-stock-data dengan nama file AAPL.csv menggunakan fungsi pd.read_csv(). Terakhir, beberapa baris pertama dari dataset ditampilkan dengan menggunakan metode head(), yang memungkinkan pengguna untuk melihat struktur dan isi awal dari data yang dimuat."""

# Display end few rows
df.tail()

"""Kode ini digunakan untuk menampilkan beberapa baris terakhir dari dataset yang telah dimuat sebelumnya. Dengan menggunakan metode tail(), pengguna dapat melihat data di bagian akhir dari DataFrame df. Ini berguna untuk memeriksa apakah dataset telah dimuat dengan benar dan untuk memahami struktur serta nilai-nilai yang terdapat di bagian akhir dataset, seperti tanggal terakhir dan harga saham terakhir yang tercatat."""

# Display basic information about the dataset
df.info()

# Display summary statistics
df.describe()

"""Output ini memberikan informasi ringkas mengenai DataFrame yang dimuat dari dataset saham Apple. Berikut ini adalah penjelasan dari setiap bagian:

Informasi Umum tentang DataFrame:

Tipe: pandas.core.frame.DataFrame menunjukkan bahwa objek ini adalah DataFrame dari pustaka Pandas.
RangeIndex: Terdapat 10.987 entri dalam dataset, dengan indeks mulai dari 0 hingga 10.986.
Jumlah Kolom: Terdapat 7 kolom dalam DataFrame.
Detail Kolom:

Unnamed: 0: Kolom ini adalah integer (int64) dan tidak memiliki nilai kosong (10987 non-null).

Date: Kolom bertipe objek (object) yang juga tidak memiliki nilai kosong. Ini biasanya berisi tanggal.

Open, High, Low, Close: Kolom-kolom ini bertipe float (float64) yang mencatat harga saham pada rentang tertentu.

Volume: Kolom ini juga bertipe integer (int64) yang mencatat jumlah saham yang diperdagangkan.

Statistik Deskriptif:

Menyediakan statistik seperti count (jumlah), mean (rata-rata), std (deviasi standar), min (nilai minimum), 25%, 50% (median), 75%, dan max (nilai maksimum) untuk setiap kolom.

Kolom "Open" memiliki rata-rata harga 21.302671 dan deviasi standar 45.148670, dengan nilai maksimum mencapai 236.531998.

Kolom "Volume" memiliki rata-rata sekitar 318.349.900, dengan nilai maksimum mencapai 7.421.641.000.

Secara keseluruhan, output ini memberikan gambaran umum yang komprehensif tentang struktur dan statistik dasar dari dataset yang dimuat, yang dapat membantu dalam analisis lebih lanjut.
"""

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

"""Grafik menunjukkan tren harga penutupan saham Apple sejak 1980 hingga 2025. Dari visualisasi ini, terlihat bahwa harga saham Apple mengalami pertumbuhan yang signifikan, terutama setelah tahun 2010. Peningkatan tajam dalam dekade terakhir mencerminkan keberhasilan perusahaan dalam inovasi produk, seperti iPhone, serta pertumbuhan global Apple sebagai raksasa teknologi. Pergerakan harga ini menggambarkan peningkatan nilai pasar Apple dan minat investor yang tinggi."""

# Plot the trading volume for AAPL
plt.figure(figsize=(14,8))
plt.plot(df['Date'], df['Volume'], label='Volume', color='orange')
plt.title('Volume Perdagangan Saham Apple (AAPL)', fontsize=16)
plt.xlabel('Tanggal', fontsize=12)
plt.ylabel('Volume Perdagangan', fontsize=12)
plt.legend(loc='upper right')
plt.show()

"""Grafik menggambarkan volume perdagangan saham Apple (AAPL) selama periode waktu yang sama. Lonjakan volume perdagangan menandakan adanya aktivitas tinggi dari investor, baik itu berupa pembelian besar-besaran ataupun aksi jual. Lonjakan terbesar mungkin terkait dengan peristiwa besar seperti rilis produk baru, perubahan dalam manajemen, atau kondisi pasar global. Volume perdagangan ini dapat menjadi indikator minat pasar terhadap saham Apple dalam jangka waktu tertentu."""

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

"""pada bagian ini grafik mengombinasikan harga penutupan saham Apple dengan dua moving averages (rata-rata bergerak) yakni 50-day dan 200-day. Moving averages digunakan untuk melacak tren harga dan memuluskan fluktuasi jangka pendek. Grafik ini menunjukkan bahwa saham Apple memiliki tren bullish yang konsisten, dengan harga penutupan secara berkala berada di atas 50-day dan 200-day moving average, menandakan tren naik yang kuat dalam jangka panjang.

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

"""1. Bagian pertama adalah bertujuan untuk menyiapkan dataset yang hanya berisi satu kolom, yaitu kolom penutupan saham `('Close')`. Dataset ini memiliki banyak kolom, namun untuk keperluan prediksi harga hanya fokus pada kolom ini demngan kode
```
data_ml = data[['Close']]
```
2. Bagian kedua adalah menerapkan teknik **Feature Engineering** dengan fungsi `Shift(1)` untuk menggeser kolom `('Close')` kebawah satu baris, yang berari nilai penutupan hari sebelumnya digunakan sebagai fitur dengan kode
```
data_ml.dropna(inplace=True)
```
3. Pada tahap ketiga melakukan pembagian data keladam set pelatihan dan pengujian dengan fungsi `train_test_split` dengan proporsi 80% data latih dan 20% data uji dengan kode
```
train, test = train_test_split(data_ml, test_size=0.2, shuffle=False)
```
4. Tahap ke empat memisahkan fitur dan label, dinama fitur(x) dan label(y).
  - X_train dan X_test: Fitur yang digunakan untuk memprediksi, yaitu harga penutupan dari hari sebelumnya ('Close_shifted').
  - y_train dan y_test: Label atau target yang akan diprediksi, yaitu harga penutupan hari ini ('Close').
  
  dengan memisahkan fitur dan label bertujuan agar supaya model bisa mempelajari hubungan antara hatga sebelumnya dan harga yang akan datang dengan kode
```
X_train, y_train = train[['Close_shifted']], train['Close']
X_test, y_test = test[['Close_shifted']], test['Close']
```

5. Yang terakhir menggunakan normalasi data dengan MinMaxScaler agar menskalakan data ke rentang antara 0 dan 1, dilakukan untuk mengurangi perbedaan skala yang terlalu besar dengan kode
```
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

#Linear Regression
"""

# Linear Regression
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_test_scaled)

"""Linear Regression adalah salah satu algoritma yang paling sederhana dan paling umum digunakan dalam analisis regresi. Tujuan dari regresi linier adalah untuk memodelkan hubungan antara satu atau lebih variabel independen (fitur) dan variabel dependen (target)

penjelasan kode
1. Mendifinisikan `Linear Regreassion` kedalam variabel `lr`
2. Pelatihan model dengan metode `fit` dengan data yang sudah dinormalisasi `(X_train_scaled)` dan label `(y_train)`
3. Prediksi menggunakan metode `predict` untuk menghasilkan perbedaan data uji `(X_train_scaled)` dan hasil prediksi disimpan dalam variabel `y_pred_lr`

#Random Forest Regression
"""

# Random Forest Regression
rf = RandomForestRegressor(n_estimators=100)
rf.fit(X_train_scaled, y_train)
y_pred_rf = rf.predict(X_test_scaled)

"""Random Forest Regression adalah metode ensemble yang digunakan untuk regresi, yang menggabungkan hasil dari banyak pohon keputusan (decision trees) untuk memberikan prediksi yang lebih akurat dan stabil.

penjelasan kode
1. mendifinisikan `Random Forest` kedalam variabel `rf`
2. Membuat objek kelas `Random Forest` dengan parameter `n_estimators=100` yang berati menggunakan 100 pohon keputusa
3. Pelatihan model dengan metode fit dengan data yang sudah dinormalisasi `(X_train_scaled)` dan label `(y_train)`
4. Prediksi menggunakan metode `predict` untuk menghasilkan perbedaan data uji `(X_train_scaled)` dan hasil prediksi disimpan dalam variabel `y_pred_rf`

#ARIMA Model
"""

# ARIMA Model
model_arima = ARIMA(train['Close'], order=(1,1,3))
model_arima_fit = model_arima.fit()
y_pred_arima = model_arima_fit.forecast(steps=len(test))

"""ARIMA adalah model statistik yang digunakan untuk analisis deret waktu dan peramalan. Model ini menggabungkan tiga komponen utama: autoregressive (AR), differencing (I), dan moving average (MA).

Penjelasan kode
1. Membuat objek dari kelas ARIMA, untuk memberikan data pelatihan `(train['Close'])` dan parameter model dengan urutan `(1,1,3)`.
2. Metode `fit()` digunakan untuk melatih model ARIMA pada data pelatihan. Selama proses ini, model akan menghitung koefisien AR dan MA yang optimal berdasarkan data pelatihan.
3. Metode `forecast()` untuk memprediksi nilai masa depan berdasarkan model yang telah dibangun. Parameter `steps=len(test)` menunjukkan bahwa dalam proyek ini ingin memprediksi jumlah langkah yang sama dengan jumlah data dalam set pengujian. Hasil prediksi akan disimpan dalam variabel `y_pred_arima`.

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

"""- Persiapan Data
Data yang digunakan (dalam hal ini adalah harga penutupan, data[['Close']]) diambil dalam bentuk array. Data ini merupakan data time series yang akan diprediksi.

- Scaling untuk LSTM (MinMaxScaler)
Data di-scaling menggunakan MinMaxScaler untuk membawa nilai ke dalam rentang [0, 1]. Ini penting dalam LSTM untuk mempercepat konvergensi dan menghindari masalah numerik.

- Mempersiapkan Dataset untuk LSTM
 - Dataset disiapkan dengan menciptakan X_lstm dan y_lstm. Dalam hal ini, untuk setiap 60 nilai sebelumnya dari data terukur, model akan memprediksi nilai selanjutnya. Ini dilakukan dengan loop yang mengisi array dengan data yang sesuai:
  ```
  python
  For i in range(60, len(train_scaled)):
    X_lstm.append(train_scaled[i-60:i, 0])
    y_lstm.append(train_scaled[i, 0])
  ```
 - Reshape Data: Setelah itu, bentuk data X_lstm diubah agar sesuai dengan input yang diharapkan oleh LSTM, yaitu (samples, time steps, features).

- Membangun Model LSTM
Model LSTM dibangun menggunakan Sequential dari Keras. Struktur model adalah sebagai berikut:
 - LSTM Layer 1: Menambahkan layer LSTM pertama dengan 50 unit dan pengaturan return_sequences=True untuk memungkinkan layer berikutnya menerima output dari layer ini.
 - LSTM Layer 2: Layer LSTM kedua tanpa pengaturan return_sequences.
 - Dense Layer 1: Layer Dense dengan 25 unit.
 - Output Layer: Layer Dense dengan 1 unit untuk menghasilkan prediksi nilai harga penutupan.

- Komplikasi Model (Compile)
Model dikompilasi dengan menggunakan optimizer Adam dan loss function mean_squared_error untuk mengukur seberapa baik model memprediksi data.

- Menambahkan Early Stopping
EarlyStopping digunakan untuk mencegah overfitting dengan menghentikan pelatihan jika tidak ada peningkatan dalam loss selama 5 epoch.

- Melatih Model (Fit)
Model dilatih pada dataset yang sudah dipersiapkan dengan 50 epoch dan batch size 64, serta menggunakan callback early stopping.

- Prediksi menggunakan LSTM
Setelah model dilatih, dilakukan prediksi dengan data uji yang diambil dari 60 nilai terakhir dari data asli. Data tersebut juga di-scaling agar sesuai dengan format yang digunakan dalam model. Selanjutnya, hasil prediksi di-inverse transform untuk mengembalikan ke skala asli.

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

"""Evaluasi ini menggunakan MSE mengukur rata-rata kuadrat perbedaan antara nilai yang diprediksi oleh model dan nilai aktual. Metrik ini memberikan bobot lebih pada kesalahan yang lebih besar, sehingga memberikan gambaran yang lebih jelas tentang seberapa besar kesalahan prediksi."""

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

"""Evaluasi ini menggunkan R-squared adalah metrik yang menunjukkan seberapa baik model menjelaskan variasi dalam data. Nilai R² berkisar antara 0 dan 1, di mana nilai yang lebih tinggi menunjukkan bahwa model menjelaskan proporsi yang lebih besar dari variasi data. Nilai negatif menunjukkan bahwa model tidak mampu menjelaskan variabilitas data lebih baik daripada model rata-rata."""

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

"""Pada bagian ini bertujuan mencari model terbaik berdasarkan evaluasi MSE, dimana menghasilkan bahwa Linear Regeression adalah model terbaik dengan nilai MSE sebesar 3.8 dimana semakin kecil nilai MSE semaik baik model yang d buat."""

# Plot predictions for LSTM
plot_predictions("LSTM", y_test, y_pred_lstm, data, training_data_len)

"""Grafik ini menggunakan model LSTM (Long Short-Term Memory), sebuah jenis jaringan saraf tiruan yang cocok untuk data berurutan. Grafik ini menunjukkan bahwa prediksi LSTM sangat mendekati harga aktual, dengan model yang lebih akurat dalam menangkap fluktuasi harga yang cepat dan pola yang tidak teratur. LSTM tampaknya menjadi model yang lebih unggul dalam memahami pola harga saham yang rumit dibandingkan dengan model prediksi sebelumnya."""

# Plot predictions for Linear Regression
plot_predictions("Linear Regression", y_test, y_pred_lr, data, training_data_len)

"""Grafik ini menunjukkan hasil prediksi menggunakan model Linear Regression dibandingkan dengan harga aktual. Hasil prediksi cukup akurat dengan mengikuti tren harga secara keseluruhan. Regresi linear tampak lebih baik dalam mengikuti tren umum kenaikan harga saham Apple. Namun, seperti kebanyakan model linear, ia mungkin tidak sepenuhnya menangkap pergerakan harga yang kompleks dan volatil dalam jangka pendek."""

# Plot predictions for Random Forest
plot_predictions("Random Forest", y_test, y_pred_rf, data, training_data_len)

"""Grafik ini menggunakan model Random Forest untuk memprediksi harga saham. Grafik ini menunjukkan bahwa model Random Forest cenderung memprediksi nilai harga yang lebih konservatif, terutama dalam periode awal prediksi. Hal ini mengindikasikan bahwa model ini mungkin memiliki keterbatasan dalam memprediksi lonjakan harga yang tajam atau peristiwa besar yang memengaruhi pasar saham."""

# Plot predictions for ARIMA
plot_predictions("ARIMA", y_test, y_pred_arima, data, training_data_len)

"""Grafik ini membandingkan prediksi harga saham Apple menggunakan model ARIMA (AutoRegressive Integrated Moving Average) dengan harga aktualnya. Model ARIMA tampaknya kurang dapat menangkap fluktuasi harga yang lebih signifikan pada data historis saham Apple, terutama pada periode setelah 2016. Grafik ini menunjukkan bahwa meskipun ARIMA dapat memprediksi tren dasar, model ini mungkin kurang akurat dalam memprediksi pergerakan harga saham yang lebih dinamis dan volatil.


"""