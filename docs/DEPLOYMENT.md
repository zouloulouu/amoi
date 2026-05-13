# Déploiement production

Stack cible — **tout sur un seul VPS** :

```
Internet
  │
  └─→ tondomaine.fr   (VPS OVH, nginx)
        │
        ├── /          →  /var/www/data_ina/  (frontend dist statique)
        └── /api/*     →  http://127.0.0.1:8000  (uvicorn dans Docker)
```

**Un seul domaine, un seul serveur, un seul compte (OVH).** CORS pas nécessaire (même origine).

Streamlit Cloud peut continuer à tourner en parallèle pendant la transition.

---

## 0 — Prérequis avant déploiement

Tu dois avoir, dans cet ordre :

1. **Un VPS** OVH (Ubuntu 24.04, voir spec ci-dessous)
2. **Un domaine** (n'importe quel registrar)
3. **Un token HuggingFace** (optionnel — utile si tu ne copies pas un snapshot local sur le VPS)

---

## 1 — Commander le VPS OVH

| Plan | vCPU | RAM | SSD | Prix | Conseil |
|---|---|---|---|---|---|
| **VPS Starter** | 2 | 4 Go | 80 Go | ~6 €/mois | Suffisant pour MVP, peut serrer pendant le pré-warming |
| **VPS Comfort** | 4 | 8 Go | 80 Go | ~12 €/mois | **Recommandé** — confortable pour les 11 thèmes |
| **VPS Elite** | 8 | 16 Go | 160 Go | ~24 €/mois | Overkill pour ton usage actuel |

**OS** : **Ubuntu 24.04 LTS** (Docker out-of-the-box, support 5 ans).

**Datacenter** : Gravelines (GRA) ou Strasbourg (SBG) pour latence FR optimale.

---

## 2 — Acheter un domaine

| Registrar | Avantage | TLD .fr | TLD .com |
|---|---|---|---|
| **OVH** | Tout au même endroit | ~7 €/an | ~10 €/an |
| **Cloudflare Registrar** | Prix coûtant, zero markup | ~9 €/an | ~10 €/an |

---

## 3 — Configurer DNS

Une fois VPS provisionné (tu as son IP, ex `51.83.12.34`) et domaine acheté, ajouter ces records :

| Type | Name | Value | TTL |
|---|---|---|---|
| A | `@` (apex) | `51.83.12.34` (ton IP VPS) | 600 |
| A | `www` | `51.83.12.34` | 600 |

→ `tondomaine.fr` et `www.tondomaine.fr` pointent vers ton VPS. Compte ~5 min de propagation.

---

## 4 — Provisionnement VPS (Ubuntu 24.04)

SSH dans le VPS (`ssh ubuntu@51.83.12.34`), puis :

```bash
# Créer un user non-root
sudo adduser --gecos "" ina
sudo usermod -aG sudo ina

# Pousser ta clé SSH publique
sudo mkdir -p /home/ina/.ssh
sudo cp ~/.ssh/authorized_keys /home/ina/.ssh/
sudo chown -R ina:ina /home/ina/.ssh
sudo chmod 700 /home/ina/.ssh
sudo chmod 600 /home/ina/.ssh/authorized_keys

# Désormais : ssh ina@51.83.12.34
exit
ssh ina@51.83.12.34

# Mises à jour + outils
sudo apt update && sudo apt upgrade -y
sudo apt install -y git curl ufw

# Firewall : SSH + HTTP + HTTPS uniquement
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow 22/tcp
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw enable

# Docker + Compose plugin
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker ina
# Reconnect pour que le groupe docker prenne effet
exit
ssh ina@51.83.12.34
docker --version  # check

# Node.js (pour build frontend) + npm
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt install -y nodejs
node --version
npm --version

# nginx + certbot (HTTPS Let's Encrypt)
sudo apt install -y nginx certbot python3-certbot-nginx
```

---

## 5 — Déployer le backend (Docker)

```bash
# Cloner le repo
cd ~
git clone https://github.com/zouloulouu/amoi.git data_ina
cd data_ina

# Configurer les secrets
cp .env.example .env
nano .env
# Remplir :
#   HF_TOKEN=<ton token HF>
#   INA_CORS_ORIGINS=                  # peut rester vide : tout vient du même origin
#   INA_HF_REPO_ID=zouloulouu/data_ina_clean
#   INA_DICTIONARY_PATH=/app/state/dictionaries.json
#   INA_TAGGING_CACHE_MAXSIZE=4        # prudent pour VPS 8 Go
#   INA_DISABLE_PREWARM=0              # passer à 1 si pression mémoire au boot

# Option A : laisser FastAPI télécharger le corpus depuis HF au boot (lent au 1er boot, ~1 min)
# Option B : copier un snapshot local depuis ta machine :
#   rsync -avz data/clean/ ina@51.83.12.34:~/data_ina/data/clean/

# Build + start
docker compose up -d --build

# Vérifier le boot
docker compose logs -f api
# Attendre "Corpus loaded: ..." et "Application startup complete."

curl -fsS http://127.0.0.1:8000/health
# → {"status":"ok","data_loaded":true,...}
```

---

## 6 — Build le frontend sur le VPS

```bash
cd ~/data_ina/frontend
npm ci          # installation reproductible depuis package-lock.json
npm run build   # build production avec .env.production (VITE_API_BASE_URL=/api)

# Déployer les fichiers statiques pour nginx
sudo mkdir -p /var/www/data_ina
sudo cp -r dist/* /var/www/data_ina/
sudo chown -R www-data:www-data /var/www/data_ina
```

---

## 7 — Configurer nginx (frontend + reverse proxy)

```bash
sudo nano /etc/nginx/sites-available/data_ina
```

Coller :

```nginx
server {
    listen 80;
    server_name tondomaine.fr www.tondomaine.fr;

    # Augmenter timeouts pour les analyses longues sur cold start
    proxy_connect_timeout 60s;
    proxy_send_timeout    300s;
    proxy_read_timeout    300s;
    client_max_body_size  10M;

    # Frontend statique : SPA routing → toutes les routes inconnues servent index.html
    root /var/www/data_ina;
    index index.html;

    # Cache long pour les assets versionnés par Vite (hash dans le nom de fichier)
    location /assets/ {
        try_files $uri =404;
        add_header Cache-Control "public, max-age=31536000, immutable";
    }

    # API : reverse proxy vers uvicorn dans Docker
    # /api/health → http://127.0.0.1:8000/health
    location /api/ {
        rewrite ^/api/(.*)$ /$1 break;
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Host              $host;
        proxy_set_header X-Real-IP         $remote_addr;
        proxy_set_header X-Forwarded-For   $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    # SPA routing : toute autre route → index.html (gérée par React Router)
    location / {
        try_files $uri $uri/ /index.html;
        add_header X-Content-Type-Options "nosniff";
        add_header Referrer-Policy "strict-origin-when-cross-origin";
    }
}
```

Activer + recharger :

```bash
sudo ln -s /etc/nginx/sites-available/data_ina /etc/nginx/sites-enabled/
sudo rm -f /etc/nginx/sites-enabled/default   # virer le site par défaut nginx
sudo nginx -t                                  # tester la config
sudo systemctl reload nginx

# Tester en HTTP
curl -fsS http://tondomaine.fr/api/health
curl -I http://tondomaine.fr/
# Doit retourner du HTML pour la racine, du JSON pour /api/health
```

---

## 8 — HTTPS Let's Encrypt

```bash
sudo certbot --nginx -d tondomaine.fr -d www.tondomaine.fr
# certbot met à jour automatiquement le fichier nginx pour ajouter HTTPS + redirect HTTP→HTTPS
```

Vérifier : ouvrir `https://tondomaine.fr` dans le navigateur. Le cadenas doit être vert.

---

## 9 — Script de déploiement (pour les mises à jour ultérieures)

Créer `~/data_ina/scripts/deploy.sh` :

```bash
#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/.."

echo "==> Pull from GitHub"
git pull

echo "==> Rebuild backend Docker"
docker compose up -d --build

echo "==> Build frontend"
cd frontend
npm ci
npm run build

echo "==> Deploy frontend to nginx"
sudo cp -r dist/* /var/www/data_ina/
sudo chown -R www-data:www-data /var/www/data_ina

echo "==> Reload nginx"
sudo systemctl reload nginx

echo "==> Done. Smoke test:"
curl -fsS https://tondomaine.fr/api/health
```

Rendre exécutable :

```bash
chmod +x ~/data_ina/scripts/deploy.sh
```

Désormais pour déployer : `~/data_ina/scripts/deploy.sh`.

---

## 10 — Vérifier la chaîne complète

```bash
# Backend via reverse proxy
curl -fsS https://tondomaine.fr/api/health
curl -fsS https://tondomaine.fr/api/themes
curl -fsS https://tondomaine.fr/api/docs   # Swagger UI

# Frontend
# Ouvrir https://tondomaine.fr/ dans le navigateur
# DevTools → Network → vérifier que les appels XHR vont vers /api/...
```

---

## 11 — Troubleshooting rapide

| Symptôme | Diagnostic |
|---|---|
| `502 Bad Gateway` sur /api | Container down ou pas encore healthy. `docker compose ps` puis `docker compose logs api` |
| Frontend → écran blanc | Vérifier `/var/www/data_ina/index.html` existe. Vérifier que `try_files ... /index.html` est bien dans nginx |
| Frontend → 404 sur navigation directe | `try_files $uri $uri/ /index.html;` manque dans le block `location /` |
| `/api/health` répond `degraded` | Corpus pas chargé. Voir `docker compose logs api` pour les `issues` |
| Mémoire saturée | Pré-warming consomme trop. Augmenter swap ou prendre VPS Comfort |
| Certificat expiré | certbot renouvelle auto via cron. Forcer : `sudo certbot renew --force-renewal` |

---

## 12 — Backup et rollback

### Backup `dictionaries.json` (cron horaire)

```bash
mkdir -p ~/dictionaries-backup
crontab -e
# Append:
0 * * * * cp ~/data_ina/dictionaries.json ~/dictionaries-backup/$(date +\%Y\%m\%d-\%H).json
```

### Rollback rapide

```bash
cd ~/data_ina
git log --oneline -10
git checkout <commit-hash>
./scripts/deploy.sh
```

Tag git `pre-migration` reste l'option « tout retour à zéro » la plus sûre.
