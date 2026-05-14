# data_ina

Application d'analyse thematique du corpus INA.

## Etat courant

- `backend/ina_api` expose l'API FastAPI.
- `frontend/` contient l'interface React/Vite.
- `deploy.sh` deploie l'application sur le VPS.
- `legacy/streamlit/app.py` est l'application Streamlit legacy, conservee comme fallback pendant la transition.

## Lancer en developpement

Option simple, depuis la racine :

```powershell
.\scripts\dev.ps1
```

Ou avec deux terminaux :

```powershell
python -m uvicorn ina_api.main:app --host 127.0.0.1 --port 8000
```

```powershell
cd frontend
$env:VITE_API_BASE_URL="http://127.0.0.1:8000"
npm.cmd run dev -- --port 5173
```

URLs :

- Frontend : `http://127.0.0.1:5173`
- API docs : `http://127.0.0.1:8000/docs`

## Deploiement

Sur le VPS :

```bash
cd ~/data_ina
bash deploy.sh
```

Voir [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) pour le runbook complet.

## Verifications

```powershell
python -m pytest -q backend/tests --basetemp C:\tmp\pytest-data-ina
```

```powershell
cd frontend
npm.cmd run lint
npm.cmd run test
npm.cmd run build
```

Si `WinError 10048` apparait, le port est deja utilise :

```powershell
netstat -ano | findstr :8000
Stop-Process -Id LE_PID -Force
```
