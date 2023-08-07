# Laporan Akhir Kecerdasan Buatan - Kelompok 4

**Anggota Kelompok :**

1. Ashmaa'Ulia Safitri M - 3.34.21.0.06

2. Hanna Maghfiroh - 3.34.21.0.11

   

## Domain Proyek

Data ada dimanapun, mulai dari sosial media sampai transaksi bank. Semua itu melakukan pemprosesan data. Dengan data yang jumlahnya sangatlah banyak , muncullah masalah yang data untuk mengatur data tersebut. Data Science datang untuk mengatur, mengeksplor dan analisis data menjadi informasi yang berharga dan bisa digunakan. Data Science merupakan salah satu pekerjaan terpopuler abad 21, maka dari itu pekerjaan menjadi Data Scientist begitu dibutuhkan . Yang dimana untuk Data Scientist juga memiliki tingkatan Gaji yang mewakili seberapa besar tanggung jawab seorang Data Scientist tersebut.


## Business Understanding

### Problem Statements

Menjelaskan pernyataan masalah latar belakang:

- Bagaimana cara <em> preprocessing </em> pada data Dataset  "Data Science Job Salaries" yang akan digunakan untuk membuat model yang baik?
- Bagaimana cara memilih/membuat model yang terbaik untuk memprediksi Gaji dari Seorang Data Scientist?

### Goals

Menjelaskan tujuan dari pernyataan masalah:

- Melakukan <em> preprocessing </em> data sehingga data tersebut siap untuk di latih oleh model <em> Machine Learning </em>
- Menggunakan library python untuk melakukan permodelan dengan algoritma Linear Regression, Ridge Regression , Lasso Regression, Random Forest Regression, dan XGBoost serta menggunakan RMSE sebagai metrik evaluasi nya.


- Untuk <em> preprocessing </em> data dapat dilakukan beberapa teknik, diantaranya :

  - Melakukan drop kolom pada kolom yang tidak penting / yang tidak berpengaruh pada prediksi gaji.
  - Melakukan EDA(Explorasi Data Analis) berupa menganalisis feature-feature menggunakan <em>Univarate Analysis</em> dan <em>Multivariate Analysis</em>
  - Melakukan pembagian dataset menjadi dua bagian dengan rasio 7:3 / 70% untuk train dan 30% untuk test.
  
- Untuk Pemilihan model terbaik data dapat dilakukan beberapa teknik, diantaranya :

  - Menghitung metric yang akan menjadikan patokan kita untuk memilih model terbaik menggunakan <em>RMSE</em>

  - Berikut adalah rumus untuk menghitung RMSE

    ![image-20230719141305734](C:\Users\ACER\AppData\Roaming\Typora\typora-user-images\image-20230719141305734.png)

    ​																				Gambar 1. <em>Rumus RMSE</em>

  - Rumus diatas dapat dihitung langsung menggunakan library python yaitu sklearn metrics

Setelah goals dicapai, selanjutnya adalah tahap implementasi. 

## Data Understanding

Data yang digunakan adalah data yang berasal dari kaggle, data ini berisikan gaji data scientist beserta tingkatan pekerjaan dan lain - lain berikut adalah datanya [<em> Data Science Job Salaries </em>](https://www.kaggle.com/datasets/ruchi798/data-science-job-salaries)

### Variabel-variabel pada  <em> Data Science Job Salaries </em> Datasets adalah sebagai berikut:

- work_year: The year the salary was paid
- experience_level: EN Entry-level(Junior) MI Mid-level(Intermediate) SE Senior-level(Expert) EX Executive-level(Director)
- job_title: The role worked in during the year
- salary: The total gross salary amount paid
- salary_currency: The currency of the salary paid as an ISO 4217 currency code
- salary_in_usd: The salary in USD
- employee_residence: Employee's primary country of residence in during the work year
- remote_ratio: 0 No remote work (less than 20%) 50 Partially remote 100 Fully remote (more than 80%)
- company_location: The country of the mployer's main office or contracting branch
- company_size: The average number of people that worked for the company during the year: S (less than 50 employees) M (50-250 employees) L (more than 250 employees)


### Analisis Deskriptif

Tabel 1. <em> Generative Describe Statistics </em>

|       | Unnamed: 0 |   work_year |       salary | salary_in_usd | remote_ratio |
| ----: | ---------: | ----------: | -----------: | ------------: | -----------: |
| count | 607.000000 |  607.000000 | 6.070000e+02 |    607.000000 |    607.00000 |
|  mean | 303.000000 | 2021.405272 | 3.240001e+05 | 112297.869852 |     70.92257 |
|   std | 175.370085 |    0.692133 | 1.544357e+06 |  70957.259411 |     40.70913 |
|   min |   0.000000 | 2020.000000 | 4.000000e+03 |   2859.000000 |      0.00000 |
|   25% | 151.500000 | 2021.000000 | 7.000000e+04 |  62726.000000 |     50.00000 |
|   50% | 303.000000 | 2022.000000 | 1.150000e+05 | 101570.000000 |    100.00000 |
|   75% | 454.500000 | 2022.000000 | 1.650000e+05 | 150000.000000 |    100.00000 |
|   max | 606.000000 | 2022.000000 | 3.040000e+07 | 600000.000000 |    100.00000 |

### <em> Visualization </em>

Berikut adalah kolerasi antar fitur yang terdapat pada datasets

![image-20230719144206510](C:\Users\ACER\AppData\Roaming\Typora\typora-user-images\image-20230719144206510.png)

​																				Gambar 2. Korelasi antar Kolom/Fitur

## Data Preparation

Teknik Data Preparation yang Dilakukan adalah sebagai berikut:

- <em> TrainTestSplit </em>() : Membagi dataset menjadi data latih (train) dan data uji (test) merupakan hal yang harus dilakukan sebelum membuat model. Mempertahankan sebagian data yang ada untuk menguji seberapa baik generalisasi model terhadap data baru.

Result:

Untuk Train Test Split kita bisa menggunakan potongan kode berikut:

```
from sklearn.model_selection import train_test_split

y_target = df['salary_in_usd']
X_features = df.drop(['salary_in_usd', 'job_title', 'employee_residence', 'company_location'], axis=1, inplace=False)
log_y_target = np.log1p(y_target)
X_features_ohe = pd.get_dummies(X_features)
X_train, X_test, y_train, y_test = train_test_split(X_features_ohe, log_y_target, test_size=0.3, random_state=156)
```

## Modeling

Pada tahap ini, akan dikembangkan model <em> Machine Learning </em> dengan melakukan perbandingan 5 algoritma yaitu  Linear Regression, Ridge Regression , Lasso Regression, Random Forest Regression, dan XGBoost .Selanjutnya dari 5 Algoritma tersebut akan kita evaluasi performa metric nya di tahap Evaluation untuk menentukan model terbaik. 

### Model yang digunakan

#### Parameters

- <em> n_estimator </em>= Jumlah tree di forest. [4]
- <em>max_depth </em> = kedalaman atau panjang pohon. Ia merupakan ukuran seberapa banyak pohon dapat membelah (splitting) untuk membagi setiap node ke dalam jumlah pengamatan yang diinginkan.[4]
- <em> n_jobs </em> = jumlah job (pekerjaan) yang digunakan secara paralel. Ia merupakan komponen untuk mengontrol thread atau proses yang berjalan secara paralel. n_jobs=-1 artinya semua proses berjalan secara paralel.[4]
- <em> subsample </em> = Fraksi sampel yang akan digunakan untuk menyesuaikan basis individu peserta didik. Jika lebih kecil dari 1,0 ini menghasilkan Stochastic Gradient Boosting. subsampel berinteraksi dengan parameter n_estimators. Memilih subsampel <1,0 mengarah pada pengurangan varians dan peningkatan bias. Nilai harus dalam rentang (0.0, 1.0]. [3] 

#### Models

- <em>Linear Regression</em> = Suatu regresi linear yang digunakan untuk memprediksi hubungan antara dua variabel dalam penelitian kuantitatif.
- <em>Ridge Regression</em> = Teknik regresi dalam machine learning yang digunakan untuk menstabilkan parameter regresi karena adanya multikolinieritas.
- <em>Lasso Regression</em> = Digunakan untuk meminimalkan nilai residual dari model regresi dengan menambahkan penalti pada koefisien variabel yang tidak signifikan atau tidak berpengaruh terhadap variabel dependen.

- <em>Random Forest</em> = Random Forest dapat digunakan sebagai regresi dengan memperluas 'tree' sepenuhnya sehingga setiap daun memiliki tepat satu nilai. Breiman menyarankan untuk membuat regresi random forest dengan cara memperluas pohon secara acak. Kemudian sebuah prediksi secara sederhana mengembalikan variabel respon individual dari distribusi dapat dibangun jika 'forest' cukup besar. Satu peringatan bahwa perkembangan 'tree' sepenuhnya dapat menutupi atau melebihi kapasitas: jika itu terjadi, intervalnya akan sia-sia, seperti prediksi. Hal yang diharapkan adalah sama seperti akurasi dan presisi. [2]
  - Kelebihan
    - Menghasilkan eror yang lebih rendah.
    - Memberikan hasil yang bagus dalam klasifikasi.
    - Dapat mengatasi data training dalam jumlah sangat besar secara efisien.
    - Metode yang efektif untuk mengestimasi hilangnya data.
    - Dapat memperkiraan variabel apa yang penting dalam klasifikasi.
    - Menyediakan metode eksperimental untuk mendeteksi interaksi variabel.
  - Kekurangan
    - Waktu pemrosesan yang lama karena menggunakan data yang banyak dan membangun model tree yang banyak pula untuk membentuk random trees karena menggunakan single processor.
    - Interpretasi yang sulit dan membutuhkan mode penyetelan yang tepat untuk data.
    - Ketika digunakan untuk regresi, mereka tidak dapat memprediksi di luar kisaran dalam data percobaan, hal ini di mungkinkan data terlalu cocok dengan kumpulan data pengganggu (noisy).

  - Parameter
    - n_estimator = 100 
    - max_depth = Infinite / None (node diperluas sampai semua semua daun kurang dari sampel)
    - n_jobs = -1
  - <em>XGBoost</em> = XGBoostadalah implementasi sumber terbuka yang populer dan efisien dari algoritma pohon yang ditingkatkan gradien. Peningkatan gradien adalah algoritma pembelajaran yang diawasi, yang mencoba memprediksi variabel target secara akurat dengan menggabungkan perkiraan serangkaian model yang lebih sederhana dan lebih lemah. Saat menggunakangradien meningkatkanuntuk regresi, peserta didik yang lemah adalah pohon regresi, dan setiap pohon regresi memetakan titik data input ke salah satu daunnya yang berisi skor berkelanjutan. XGBoost meminimalkan fungsi obyektif yang diatur (L1 dan L2) yang menggabungkan fungsi kehilangan cembung (berdasarkan perbedaan antara output yang diprediksi dan target) dan istilah penalti untuk kompleksitas model (dengan kata lain, fungsi pohon regresi). Pelatihan berlangsung berulang, menambahkan pohon baru yang memprediksi residu atau kesalahan pohon sebelumnya yang kemudian digabungkan dengan pohon sebelumnya untuk membuat prediksi akhir. Ini disebut peningkatan gradien karena menggunakan algoritma turunan gradien untuk meminimalkan kerugian saat menambahkan model baru. Di bawah ini adalah ilustrasi singkat tentang bagaimana gradient tree boosting bekerja. [1]
      ![XGBoost Works](https://docs.aws.amazon.com/id_id/sagemaker/latest/dg/images/xgboost_illustration.png)
                                                                                 Gambar 3. <em> How XGBoost Works? </em>
  - Parameter
      - n_estimators = 100 (Default)
      - subsample = None (Default)
      - max_depth = None (Default)

## Evaluation

Model yang digunakan adalah model regressi, sesuai penjelasan diatas saya akan menggunakan beberapa metrik untuk evaluasi, berikut adalah list nya:

- Root Mean Squared Error (RMSE)

### Root Mean Squared Error (RMSE)

<em>Root Mean Squared Error (RMSE)</em> merupakan salah satu cara untuk mengevaluasi model regresi dengan mengukur tingkat akurasi hasil perkiraan suatu model. RMSE dihitung dengan mengkuadratkan error (prediksi – observasi) dibagi dengan jumlah data (= rata-rata), lalu diakarkan.

Nilai RMSE rendah menunjukkan bahwa variasi nilai yang dihasilkan oleh suatu model prakiraan mendekati variasi nilai obeservasinya. RMSE menghitung seberapa berbedanya seperangkat nilai. Semakin kecil nilai RMSE, semakin dekat nilai yang diprediksi dan diamati.

Kelebihan dari RMSE yaitu memiliki tingkat sensitivitas yang cukup tinggi. Sedangkan kekurangannya RMSE tidak menggambarkan kesalahan rata-rata saja namun memiliki implikasi lain yang lebih sulit untuk diurai dan dipahami.

![image-20230719141305734](C:\Users\ACER\AppData\Roaming\Typora\typora-user-images\image-20230719141305734.png)

​													                                  Gambar 4. <em>Rumus RMSE</em>

Diketahui:

- n = Jumlah Data
- yi = Actual Value / Nilai Sebenarnya
- ŷp = Predicted Value / Nilai Prediksi


### Final Report

Setelah melalui berbagai tahapan evaluasi diputuskan bahwa model terbaik yang akan digunakan adalah Ridge Regression sesuai dengan perhitungan matrix yang telah dijabarkan diatas. Berikut hasil akhir evaluasi metrik dari 5 Model .

Tabel 1. <em> Final Result of Model </em>

| id   | Model_Name               | rmse  |
| ---- | ------------------------ | ----- |
| 0    | Linear Regression        | 0.535 |
| 1    | Ridge Regression         | 0.527 |
| 2    | Lasso Regression         | 0.761 |
| 3    | Random Forest Regression | 0.565 |
| 4    | XGBoost Regression       | 0.694 |

![image-20230719141900297](C:\Users\ACER\AppData\Roaming\Typora\typora-user-images\image-20230719141900297.png)

Gambar 5. <em>G</em>rafik Summary Model

## Daftar Referensi

Referensi

[1] Amazon Web Services. "<em>AWS Sagemaker Docs</em>". https://docs.aws.amazon.com/id_id/sagemaker [accessed Nov. 4 2022]

[2] Plaosan. S. V. "Random Forest". Learning Box. http://learningbox.coffeecup.com/05_2_randomforest.html [accessed Nov. 4 2022]

[3] Boisberranger. J. D, et al., "Scikit Learn Documentations." https://scikit-learn.org/stable/ [accessed Nov. 4 2022]

[4] Dicoding. "Kelas Machine Learning Terapan." https://www.dicoding.com/academies/319 [accessed Nov. 4 2022]