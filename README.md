# Real-Time Object Detection for the Blind (Alat Bantu Tunanetra)

Proyek ini adalah sistem deteksi objek real-time yang dirancang sebagai alat bantu navigasi bagi tunanetra. Sistem menggunakan **ESP32-CAM** sebagai sensor visual (kamera) dan **Aplikasi Android** sebagai otak pemrosesan berbasis **Computer Vision** dan **Edge AI**.

## 🚀 Fitur Utama
- **Real-Time Detection**: Deteksi objek dengan performa rata-rata **25 FPS**.
- **Edge AI Processing**: Menggunakan model **TensorFlow Lite (Int8 Quantized)** untuk inferensi cepat di perangkat mobile.
- **Hardware Acceleration**: Mendukung **Android NNAPI Delegate** untuk pemanfaatan NPU/GPU hardware.
- **Distance Estimation**: Estimasi jarak objek dalam satuan centimeter (cm).
- **Audio Guidance**: Output suara otomatis melalui **Text-to-Speech (TTS)** saat objek terdeteksi dalam radius $\le$ 500 cm.
- **System Profiling**: Fitur log terminal di aplikasi dan integrasi **Android Logcat (Tag: LAPORAN_TA)** untuk pemantauan latensi TFLite dan Audio secara presisi.

## 🛠️ Tech Stack
### Hardware
- **ESP32-CAM**: Sebagai modul kamera dan server streaming.
- **Android Smartphone**: Perangkat pemroses (Tested on Samsung S24 FE).

### Software & Libraries
- **Language**: Kotlin (Android), C++ (ESP32).
- **UI Framework**: Jetpack Compose.
- **AI Engine**: Lite RT (TensorFlow Lite Runtime).
- **Communication**: HTTP Streaming (MJPEG over HTTP).
- **Image Processing**: TensorFlow Lite Support Library.

## 🏗️ Arsitektur Sistem
1. **ESP32-CAM** menangkap frame gambar dan bertindak sebagai server HTTP MJPEG.
2. **Android App** melakukan *request* stream ke IP ESP32-CAM melalui jaringan WiFi.
3. Setiap frame yang diterima di-*decode* dan diproses oleh **ImageProcessor**.
4. **TFLite Interpreter** (dengan NNAPI) melakukan inferensi untuk mengenali objek.
5. Hasil deteksi ditampilkan di UI dan jika memenuhi syarat jarak, sistem memicu **TTS Audio**.

## 📸 Tampilan Log Terminal
Aplikasi dilengkapi dengan terminal internal untuk memantau performa sistem:
- `Frame #` : Menampilkan urutan frame yang diproses.
- `TFLite` : Latensi murni mesin AI (ms).
- `Proc` : Total waktu pemrosesan dari input ke output (ms).
- `SUMMARY` : Ringkasan FPS dan rata-rata latensi per detik.

## ⚙️ Cara Menjalankan
1. **Setup ESP32-CAM**:
   - Flash ESP32-CAM IDE Code ke ESP32-CAM.
   - Pastikan ESP32-CAM dan HP Android berada di **jaringan WiFi yang sama**.
2. **Konfigurasi IP**:
   - Buka `MainActivity.kt`.
   - Cari variabel `streamUrl` dan sesuaikan IP-nya dengan IP yang didapat ESP32-CAM (contoh: `http://192.168.x.x:81/stream`).
3. **Build & Run**:
   - Build proyek menggunakan **Android Studio**.
   - Install `.apk` ke perangkat Android.
   - Klik **Connect** pada aplikasi untuk memulai streaming dan deteksi.
