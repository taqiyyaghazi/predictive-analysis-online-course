# Laporan Proyek Machine Learning - Ghazi Taqiyya Al Anshari

## Domain Proyek

Platform pembelajaran daring telah menjadi sangat populer, terutama di era digital ini, di mana akses ke pendidikan berkualitas bisa didapatkan dari mana saja dan kapan saja. Namun, tantangan utama yang dihadapi platform ini adalah tingkat penyelesaian kursus yang rendah. Salah satu permasalahan yang sering terjadi adalah banyak pengguna yang mendaftar kursus tapi tidak menyelesaikannya. Memahami faktor-faktor yang mempengaruhi penyelesaian kursus dapat membantu platform untuk meningkatkan strategi mereka dalam mendukung pengguna menyelesaikan kursus mereka.

Penyelesaian kursus yang rendah dapat berdampak negatif pada reputasi dan keberlanjutan platform pembelajaran daring. Pengguna yang tidak menyelesaikan kursus mungkin merasa tidak puas, yang dapat menyebabkan penurunan retensi pengguna dan pendapatan. Untuk menyelesaikan permasalahan tersebut salah satu yang dapat dilakukan adalah identifikasi pola-pola keterlibatan pengguna yang berhubungan dengan penyelesaian kursus. Informasi ini dapat digunakan untuk mengembangkan intervensi yang lebih efektif dan personalisasi pengalaman belajar untuk meningkatkan tingkat penyelesaian kursus.

Salah satu cara yang dapat digunakan untuk mengidentifikasi pola-pola keterlibatan pengguna yang berhubungan dengan penyelesaian kursus yaitu metode klasifikasi menggunakan machine learning. Dengan menggunakan pendekatan klasifikasi, platform pembelajaran daring dapat lebih proaktif dalam mendukung pengguna untuk menyelesaikan kursus mereka. Penerapan model machine learning ini memungkinkan personalisasi pengalaman belajar dan intervensi yang lebih tepat sasaran, yang pada akhirnya dapat meningkatkan tingkat penyelesaian kursus, retensi pengguna, dan keberhasilan platform secara keseluruhan.

## Business Understanding

### Problem Statements

Berdasarkan permasalahan yang ada, muncul beberapa problem statement yaitu:

- Apa saja faktor utama yang mempengaruhi penyelesaian kursus daring?
- Bagaimana kita dapat memprediksi apakah seorang pengguna akan menyelesaikan kursus atau tidak berdasarkan keterlibatannya dalam pembelajaran?

### Goals

Berdasarkan problem statement yang ada, maka ditentukan beberapa goals yang harus dicapai yaitu

- Dapat mengidentifikasi dan menganalisis faktor-faktor utama yang mempengaruhi penyelesaian kursus daring.
- Dapat mengembangkan model prediktif yang dapat memprediksi status penyelesaian kursus pengguna berdasarkan keterlibatan pengguna dalam pembelajaran

## Data Understanding

Dataset yang digunakan dalam proyek ini menggambarkan keterlibatan pengguna dalam pembelajaran dari platform kursus daring. Dataset ini mencakup informasi demografi pengguna, data khusus kursus, dan metrik keterlibatan.
Dataset yang digunakan berasal dari platform Kaggle dengan judul [Predict Online Course Engagement Dataset](https://www.kaggle.com/datasets/rabieelkharoua/predict-online-course-engagement-dataset) yang diunggah oleh Rabie El Kharoua.

### Variabel-variabel pada Predict Online Course Engagement Dataset adalah sebagai berikut:

- UserID: Pengidentifikasi unik untuk setiap pengguna.
- CourseCategory: Kategori kursus yang diambil oleh pengguna (misalnya, Pemrograman, Bisnis, Seni).
- TimeSpentOnCourse: Total waktu yang dihabiskan oleh pengguna pada kursus dalam jam.
- NumberOfVideosWatched: Jumlah total video yang ditonton oleh pengguna.
- NumberOfQuizzesTaken: Jumlah total kuis yang diambil oleh pengguna.
- QuizScores: Skor rata-rata yang dicapai pengguna dalam kuis (persentase).
- CompletionRate: Persentase konten kursus yang diselesaikan oleh pengguna.
- DeviceType: Jenis perangkat yang digunakan oleh pengguna (Desktop (0) atau Seluler (1)).
- CourseCompletion (Variabel Target): Status penyelesaian kursus (0: Belum Selesai, 1: Selesai).

### Eksplorasi Predict Online Course Engagement Dataset

Berdasarkan hasil eksplorasi data didapatkan beberapa informasi sebagai berikut:

- Dataset yang digunakan berjumlah 9000 data dengan tidak ada data yang bernilai null.
- Dataset memiliki fitur kategorikal yaitu CourseCategory
- Dataset memiliki fitur numerikal yaitu UserID, TimeSpentOnCourse, NumberOfVideosWatched, NumberOfQuizzesTaken, QuizScores, CompletionRate, DeviceType
- Rata-rata user menghabiskan waktu di kelas adalah 50 jam
- Rata-rata video yang ditonton oleh user yaitu 10 video
- Rata-rata user mengerjakan kuis sebanyak 5 kali
- Rata-rata skor kuis user adalah 74
- Rata-rata completion rate user adalah 50%
- Kategori kelas yang paling banyak dibeli adalah kategori bisnis
- 60.36% user tidak selesai kelas dan 39.64% menyelesaikan kelas
- Berdasarkan matriks korelasi diketahui bahwa Top 3 hal yang paling mempengaruhi CourseCompletion adalah CompletionRate, QuizScores, dan NumberOfQuizzesTaken

## Data Preparation

Sebelum melakukan pelatihan model dengan data yang dimiliki perlu dilakukan proses Data Preparation. Hal tersebut dilakukan agar pealatihan model lebih efisien dan dapat memberikan hasil yang terbaik. Beberapa tahapan Data Preparation yang dilakukan adalah sebagai berikut

- **Menghapus kolom user id**, user id tidak diperlukan dalam pelatihan model karena tidak mencerminkan perilaku pengguna
- **Encoding kolom CourseCategory**, agar data course category dapat digunakan dalam pelatihan perlu di konversi ke dalam numerik. Untuk mengkonversi digunakan metode One Hot Encoder karena agar menghindari asumsi urutan atau hubungan ordinal antara kategori. Ini berarti bahwa setiap kategori diwakili secara independen dan tidak ada asumsi bahwa satu kategori lebih besar atau lebih kecil dari kategori lain.
- **Split data train dan test**, untuk mengevaluasi model hasil training maka diperlukan pembagian data menjadi data train dan test. Data train digunakan untuk pelatihan model sedangkan data test digunakan untuk mengevaluasi model hasil training. Dalam proyek ini digunakan perbandingan 80:20 untuk pembagian dataset.
- **Scalling Data**, Scalling data menggunakan StandartScaller yang memiliki cara kerja menormalkan fitur sehingga mereka memiliki rata-rata (mean) 0 dan deviasi standar 1. Scalling dilakukan untuk membantu agar fitur dengan skala yang berbeda tidak mendominasi proses pelatihan model.

## Modeling

Tahapan ini mencakup pemilihan dan penerapan model Machine Learning untuk memprediksi penyelesaian kursus.

### K-Nearest Neighbors (KNN)

KNN adalah algoritma klasifikasi yang memprediksi kelas data berdasarkan kedekatannya dengan data pelatihan terdekat. Setiap titik data diklasifikasikan berdasarkan mayoritas kelas dari K tetangga terdekat.

- Kelebihan:
  1. KNN intuitif dan tidak memerlukan asumsi tentang distribusi data.
  2. Menyesuaikan dengan distribusi data yang berbeda karena tidak memerlukan parameter model yang eksplisit.
- Kelemahan:
  1. Kinerja KNN dapat terpengaruh oleh fitur yang memiliki skala yang berbeda.
  2. KNN bisa menjadi lambat pada dataset besar karena perhitungan jarak untuk setiap prediksi.

Dalam analisis ini digunakan nilai K = 3 sehingga dalam proses klasifikasi dilakukan perhitungan jarak terdekat antar data kemudian diklasifikasikan berdasarkan 3 titik terdekatnya.

### Decision Tree

Decision Tree adalah algoritma klasifikasi yang membuat keputusan dengan memecah data menjadi subset yang lebih kecil berdasarkan fitur. Proses ini diulang hingga setiap subset memiliki satu kelas target.

- Kelebihan:
  1. Struktur pohon keputusan membuat model mudah dipahami.
  2. Dapat menangani fitur numerik dan kategorikal.
- Kelemahan:
  1. Model bisa menjadi sangat kompleks dan menangkap noise dari data.
  2. Sedikit perubahan pada data pelatihan bisa menyebabkan perubahan besar pada struktur pohon.

Dalam analisis ini digunakan parameter bawaan dari fungsi DecisionTreeClassifier yaitu min_sample_split = 2 yang berarti memperbolehkan pohon membagi node selama ada setidaknya 2 sampel di node tersebut, yang memungkinkan pohon untuk sangat kompleks. Selain itu parameter bawaan lainnya yaitu min_samples_leaf = 1 yang berarti memungkinkan leaf node untuk hanya berisi satu sampel, yang juga memungkinkan pohon menjadi sangat dalam dan kompleks.

### Random Forest

Random Forest adalah ensemble learning method yang menggabungkan banyak pohon keputusan untuk meningkatkan akurasi dan mengurangi overfitting. Setiap pohon dibangun dengan subset acak dari data dan fitur.

- Kelebihan:
  1. Mengurangi variabilitas dengan rata-rata prediksi dari banyak pohon.
  2. Bekerja dengan baik pada data non-linear dan dengan fitur yang beragam.
- Kelemahan:
  1. Pelatihan dan prediksi bisa lebih lambat dibandingkan model individu seperti pohon keputusan.
  2. Sulit untuk memahami kontribusi setiap pohon dalam prediksi.

Dalam analisis ini digunakan parameter n_estimator = 100 yang berarti model akan terdiri dari 100 pohon keputusan yang digabungkan untuk memberikan prediksi akhir.

### Gradient Boosting

Gradient Boosting adalah algoritma ensemble yang membangun model prediksi dengan cara iteratif. Setiap model baru yang ditambahkan adalah model yang memperbaiki kesalahan model sebelumnya.

- Kelebihan:
  1. Gradient Boosting sering menghasilkan model dengan akurasi tinggi dibandingkan dengan algoritma lain karena kemampuannya untuk mengurangi bias secara bertahap.
  2. Dapat digunakan dengan berbagai jenis model dasar dan bisa menangani data yang non-linear.
  3. Mampu mengontrol trade-off antara bias dan variansi, membantu mencegah overfitting.
- Kelemahan:
  1. Proses iteratif dan tuning parameter bisa memerlukan waktu dan sumber daya komputasi yang signifikan.
  2. Model yang dihasilkan sering kali kompleks dan sulit untuk diinterpretasikan secara langsung.

Dalam analisis ini digunakan beberapa parameter bawaan dari fungsi GradientBoostingClassifier yaitu :

- n_estimators = 100 yang berarti model Gradient Boosting akan terdiri dari 100 pohon keputusan, yang diiterasi untuk memperbaiki kesalahan prediksi sebelumnya.
- subsample = 1.0 yang berarti setiap pohon menggunakan seluruh data pelatihan, tanpa subsampling.
- min_samples_split = 2 dan min_samples_leaf = 1 yang berarti memperbolehkan pohon untuk membagi node selama ada setidaknya 2 sampel dan membuat leaf node yang sangat spesifik.
- max_depth = 3 yang berarti setiap pohon hanya bisa memiliki kedalaman hingga 3 level, menjaga kompleksitas pohon tetap terkendali.

## Evaluation

Pada tahap evaluasi ini, kami menilai kinerja model Gradient Boosting menggunakan metrik akurasi. Akurasi adalah salah satu metrik evaluasi yang paling umum digunakan dalam klasifikasi untuk mengukur sejauh mana model memprediksi kelas dengan benar.

Akurasi adalah rasio jumlah prediksi yang benar terhadap jumlah total prediksi. Dalam konteks klasifikasi biner, akurasi dihitung dengan rumus berikut:

\[
\text{Akurasi} = \frac{\text{Jumlah Prediksi Benar}}{\text{Jumlah Total Prediksi}}
\]

Dari 4 model yang dilatih menghasilkan akurasi sebagai berikut:
| | KNN | DecisionTree | RandomForest | Gradient Boost |
| ----------- | ----------- | ----------- | ----------- | ----------- |
| Train Accuracy | 0.92 |1|1|0.96|
| Test Accuracy | 0.84 |0.91|0.95|0.95|

Dari hasil akurasi tersebut didapatkan bahwa model dengan algoritma Gradient Boost memiliki performa terbaik karena memiliki akurasi paling tinggi dan memiliki rentang antara akurasi data train dan test paling kecil. Dari model terbaik dapat diambil seberapa penting fitur-fitur berpengaruh terhadap penyelesaian kursus sebagai berikut:

| Feature                    | Importance |
| -------------------------- | ---------- |
| QuizScores                 | 0.259075   |
| CompletionRate             | 0.227334   |
| NumberOfQuizzesTaken       | 0.218364   |
| NumberOfVideosWatched      | 0.169483   |
| TimeSpentOnCourse          | 0.125386   |
| CourseCategory_Science     | 0.000339   |
| DeviceType                 | 0.000019   |
| CourseCategory_Arts        | 0.000000   |
| CourseCategory_Business    | 0.000000   |
| CourseCategory_Health      | 0.000000   |
| CourseCategory_Programming | 0.000000   |

Dari tabel berikut dapat diambil 3 fitur yang paling memengaruhi penyelesaian kursus yaitu QuizScores, CompletionRate, dan NumberOfQuizzesTaken.

Dari hasil evaluasi, dapat diambil kesimpulan bahwa:

1. Faktor-faktor utama yang mempengaruhi penyelesaian kursus daring yaitu quiz scores, completion rate, dan number of quizzes taken.
2. Model yang dihasilkan dapat memprediksi apakah seorang pengguna akan menyelesaikan kursus atau tidak berdasarkan keterlibatannya dalam pembelajaran dengan baik dengan akurasi sebesar 95%.
