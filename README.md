# RonaAI Backend

## Deskripsi

Backend untuk aplikasi RonaAI - sistem analisis kesehatan kulit berbasis AI yang menggabungkan diagnostik berbasis kecerdasan buatan dengan manajemen perawatan kulit personal.

## Teknologi

- **Framework**: FastAPI (Python 3.12+)
- **Database**: SQLAlchemy ORM (SQLite default, dapat dikonfigurasi untuk PostgreSQL)
- **Autentikasi**: JWT Authentication dengan Python-JOSE
- **Server**: Uvicorn ASGI server
- **AI & Machine Learning**: 
  - Model ensemble untuk deteksi kondisi kulit
  - Integrasi dengan Gemini API untuk analisis lanjutan
- **Library Utama**: Pydantic, bcrypt, python-dotenv, TensorFlow 2.19.0

## Struktur Direktori

```
backend/
├── app/                  # Package aplikasi utama
│   ├── api/              # Router dan endpoint API
│   │   └── endpoints/    # Implementasi endpoint API
│   ├── core/             # Komponen inti (konfigurasi, keamanan)
│   ├── db/               # Setup database dan manajemen sesi
│   ├── models/           # Model SQLAlchemy ORM
│   ├── schemas/          # Skema Pydantic untuk validasi data
│   ├── services/         # Logika bisnis dan layer service (AI agent, model ML)
│   ├── uploads/          # Direktori untuk unggahan pengguna (gambar profil/kulit)
│   └── main.py           # Entry point aplikasi FastAPI
├── .python-version       # Menentukan versi Python (3.12)
├── pyproject.toml        # Metadata proyek dan dependensi (PEP 621)
├── requirements.txt      # Dependensi dengan versi tetap
└── Dockerfile            # Konfigurasi Docker untuk backend
```

## Endpoint API

Backend menyediakan beberapa endpoint API yang dikelompokkan berdasarkan fungsinya:

- **Auth**: Registrasi, login, dan manajemen autentikasi
- **Analysis**: Analisis gambar kulit dan riwayat analisis
- **Profile**: Manajemen profil pengguna
- **Progress**: Pelacakan kemajuan perawatan kulit
- **Journals**: Pencatatan jurnal perawatan kulit
- **Products**: Manajemen produk perawatan kulit
- **Skin**: Manajemen profil kulit pengguna

## Model Database

- **User**: Informasi pengguna dan autentikasi
- **Skin**: Profil kulit pengguna (tipe kulit, masalah)
- **Analysis**: Hasil analisis gambar kulit
- **Journals**: Catatan jurnal perawatan kulit
- **Products**: Produk perawatan kulit yang direkomendasikan

## Fitur AI

- **Deteksi Kondisi Kulit**: Menggunakan model ensemble untuk mengidentifikasi berbagai kondisi kulit seperti jerawat, komedo, bintik hitam, kulit kering, pori-pori membesar, dll.
- **Analisis Kulit**: Menggunakan Gemini API untuk memberikan analisis mendalam tentang kondisi kulit dan rekomendasi perawatan.
- **Metrik Analisis**: Menyediakan metrik kuantitatif seperti hidrasi kulit, keseragaman tekstur, visibilitas pori-pori, dan skor kesehatan keseluruhan.

## Konfigurasi

Konfigurasi aplikasi dikelola melalui file `.env` dan `app/core/config.py`. Beberapa konfigurasi utama meliputi:

- `DATABASE_URL`: URL koneksi database (default: SQLite)
- `SECRET_KEY`: Kunci rahasia untuk enkripsi JWT
- `ALGORITHM`: Algoritma enkripsi (default: HS256)
- `ACCESS_TOKEN_EXPIRE_MINUTES`: Masa berlaku token akses
- `ALLOWED_ORIGINS`: Daftar origin yang diizinkan untuk CORS

## Dokumentasi API

Setelah server backend berjalan, dokumentasi API tersedia di:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`