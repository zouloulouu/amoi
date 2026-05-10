# data_ina frontend

Frontend Vite + React + TypeScript pour l'API FastAPI `data_ina`.

## Installation

```powershell
cd C:\Users\vince\OneDrive\Bureau\data_ina\frontend
npm.cmd install
```

## Developpement

Lancement recommande depuis la racine du projet :

```powershell
.\scripts\dev.ps1
```

Lancement manuel :

```powershell
cd C:\Users\vince\OneDrive\Bureau\data_ina
$env:INA_CORS_ORIGINS="http://localhost:5173,http://127.0.0.1:5173,http://localhost:5174,http://127.0.0.1:5174"
python -m uvicorn ina_api.main:app --host 127.0.0.1 --port 8000
```

```powershell
cd C:\Users\vince\OneDrive\Bureau\data_ina\frontend
$env:VITE_API_BASE_URL="http://127.0.0.1:8000"
npm.cmd run dev -- --port 5173
```

Ports par defaut :

- API : `http://127.0.0.1:8000`
- Frontend : `http://127.0.0.1:5173`
- Vite peut proposer `5174` si `5173` est occupe.

## API types

Avec le backend lance sur `localhost:8000` :

```powershell
npm.cmd run gen:api
```

## Tests

```powershell
npm.cmd run lint
npm.cmd run test
npm.cmd run build
```

Smoke Playwright read-only, avec API et frontend deja lances :

```powershell
npm.cmd run test:e2e
```

## Depannage

`WinError 10048` signifie que le port est deja utilise :

```powershell
netstat -ano | findstr :8000
Stop-Process -Id LE_PID -Force
```
