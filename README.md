# LeafMedic — Medicinal Plant Leaf Identification

End-to-end project: ML (TensorFlow MobileNetV2), Flask backend, Expo mobile app.

---

## Project structure

```
leafmedic/
├── ml/
│   ├── data/
│   │   ├── raw/
│   │   │   ├── neem/
│   │   │   ├── tulsi/
│   │   │   └── aloe/
│   │   └── splits/
│   │       ├── train/{neem,tulsi,aloe}
│   │       ├── val/{neem,tulsi,aloe}
│   │       └── test/{neem,tulsi,aloe}
│   ├── src/
│   │   ├── preprocess.py
│   │   ├── train.py
│   │   └── test_model.py
│   ├── models/
│   │   ├── plant_model.keras
│   │   └── class_names.txt
│   └── requirements.txt
├── backend/
│   ├── app.py
│   ├── model/
│   │   ├── plant_model.keras
│   │   └── class_names.txt
│   └── requirements.txt
└── mobile-app/
    ├── app/
    │   ├── _layout.tsx
    │   ├── index.tsx
    │   ├── result.tsx
    │   ├── history.tsx
    │   └── about.tsx
    ├── components/
    │   ├── ResultCard.tsx
    │   ├── LoadingSkeleton.tsx
    │   └── ErrorState.tsx
    ├── constants/
    │   └── theme.ts
    ├── utils/
    │   └── historyStorage.ts
    ├── package.json
    ├── app.json
    ├── babel.config.js
    ├── tsconfig.json
    └── README.md
```

---

## Step-by-step setup and run

## Production deployment

The production shape is:

```text
Mobile app -> public HTTPS backend -> Flask API -> TensorFlow model
```

This repo includes a root `Dockerfile` and `railway.json` for deploying the
backend as a container. The Docker image copies only the `backend/` folder,
including `backend/model/plant_model.keras` and `backend/model/class_names.txt`.

### Deploy backend on Railway

1. Push this project to GitHub.
2. Create a new Railway project and choose "Deploy from GitHub repo".
3. Railway should detect the root `Dockerfile`.
4. Add these environment variables if you want to override defaults:
   ```text
   PREDICTION_CONFIDENCE_THRESHOLD=0.6
   CORS_ORIGINS=*
   WEB_CONCURRENCY=1
   WEB_THREADS=2
   WEB_TIMEOUT=120
   ```
5. Open the deployed service settings and generate a public domain.
6. Test the API:
   ```text
   https://YOUR-RAILWAY-DOMAIN/health
   ```

If deployment fails because TensorFlow/model memory is too high, use a larger
Railway plan or deploy the same Dockerfile to a VPS/Fly.io/Hugging Face Spaces.

### Connect the mobile app to hosted backend

Create `mobile-app/.env`:

```text
EXPO_PUBLIC_API_URL=https://YOUR-RAILWAY-DOMAIN
```

Then start Expo:

```powershell
cd c:\Users\Acer\Desktop\new\leafmedic\mobile-app
npm start
```

For an installable Android app, build with EAS after setting the same API URL
in the build environment.

### 1. ML pipeline (Python 3.11+ required)

**1.1 Create virtualenv and install ML deps**

```powershell
cd c:\Users\Acer\Desktop\new\leafmedic\ml
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

**1.2 Add raw leaf images**

- Put images in:
  - `ml/data/raw/neem/`
  - `ml/data/raw/tulsi/`
  - `ml/data/raw/aloe/`
  - `ml/data/raw/unknown/` for leaves that should not match any supported plant
- Use `.jpg`, `.png`, etc. (at least a few dozen per class for better accuracy).
  The `unknown` class should contain varied non-supported leaves, not random non-leaf objects.

**1.3 Split data into train/val/test**

```powershell
# From repo root
python ml/src/preprocess.py
```

- This creates `ml/data/splits/train`, `val`, `test` with 70% / 15% / 15% per class.

**1.4 Train the model**

```powershell
python ml/src/train.py
```

- Saves:
  - `ml/models/plant_model.keras`
  - `ml/models/class_names.txt`

**1.5 Test inference (optional)**

```powershell
python ml/src/test_model.py
# Or: python ml/src/test_model.py path/to/leaf.jpg
```

---

### 2. Backend (Flask)

**2.1 Copy model into backend**

```powershell
copy c:\Users\Acer\Desktop\new\leafmedic\ml\models\plant_model.keras c:\Users\Acer\Desktop\new\leafmedic\backend\model\
copy c:\Users\Acer\Desktop\new\leafmedic\ml\models\class_names.txt c:\Users\Acer\Desktop\new\leafmedic\backend\model\
```

**2.2 Install and run**

```powershell
cd c:\Users\Acer\Desktop\new\leafmedic\backend
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

- API: `http://localhost:5000`
- Endpoints: `GET /health`, `POST /predict` (multipart form key `image` or `file`).
- CORS is enabled for web and mobile.
- Low-confidence predictions are returned as `unknown`. Tune this with
  `PREDICTION_CONFIDENCE_THRESHOLD` before starting the backend, for example
  `$env:PREDICTION_CONFIDENCE_THRESHOLD="0.7"; python app.py`.

---

### 3. Mobile app (Expo)

**3.1 Install and start**

```powershell
cd c:\Users\Acer\Desktop\new\leafmedic\mobile-app
npm install
npm start
```

- Press `w` for web, or run Android emulator and press `a`.
- For a physical device, use your PC’s IP and set `EXPO_PUBLIC_API_URL=http://YOUR_IP:5000` (e.g. in `.env` or in the run command) so the app can reach the backend.

**3.2 Web**

```powershell
npm run web
```

- Backend must be reachable at the URL the app uses (localhost for same machine).

---

## Quick reference: preprocess only

To **only** run the preprocessing step (split raw images into train/val/test):

1. Open PowerShell and go to project root:
   ```powershell
   cd c:\Users\Acer\Desktop\new\leafmedic
   ```
2. Ensure raw images are in `ml/data/raw/neem`, `ml/data/raw/tulsi`, `ml/data/raw/aloe`.
3. Run:
   ```powershell
   python ml/src/preprocess.py
   ```
4. Check `ml/data/splits/` for `train`, `val`, and `test` subfolders with class folders inside.

---

## Troubleshooting

- **"Raw data directory not found"**  
  Create `ml/data/raw/` and subfolders `neem`, `tulsi`, `aloe`, then add images and run `preprocess.py` again.

- **"Model not found" (backend)**  
  Copy `plant_model.keras` and `class_names.txt` from `ml/models/` to `backend/model/` after training.

- **CORS / network errors in app**  
  Use your machine’s IP (not `localhost`) for `EXPO_PUBLIC_API_URL` when testing on a real device or another machine.

- **TensorFlow / CUDA issues on Windows**  
  Use a recent TensorFlow 2.x; CPU-only is fine. If you see DLL errors, try `pip install tensorflow --upgrade` or use WSL2 for training.

- **Expo "Unable to resolve module"**  
  Run `npm install` and `npx expo start -c` (clear cache).

- **Android emulator can’t reach backend**  
  Use `10.0.2.2:5000` as API URL for Android emulator (emulator’s alias for host localhost).
