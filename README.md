# NovaCraft AI - Générateur IA d'Images et Vidéos (Next.js + FastAPI)

Projet local complet suivant le cahier des charges :

- **Frontend** : Next.js (App Router, TailwindCSS 4, React 19) avec interface façon Leonardo.ai
- **Backend** : FastAPI + Diffusers pour piloter Stable Diffusion (SD1.5 & SDXL)
- **Offline first** : les modèles sont chargés depuis des fichiers `.safetensors` stockés localement

```
[ Next.js UI ]  --fetch-->  [ FastAPI /generate ]  --Diffusers-->  [ SD Models ]
```

## Démarrage rapide

### Frontend

```powershell
cd frontend
npm install
npm run dev
# URL par défaut : http://localhost:3000
```

Vous pouvez ajuster l’URL du backend avec `NEXT_PUBLIC_API_BASE_URL` (créez `.env.local`).

### Backend

Voir `backend/README.md` pour l'installation de Torch + Diffusers et les variables modèle (`MODEL_SDXL`, etc.). Le service expose `POST /generate` qui renvoie les images en base64.

**Démarrer le backend :**

```powershell
cd backend
python main.py
# Ou avec uvicorn directement :
python -m uvicorn main:app --reload --port 8000
```

Le backend sera accessible sur `http://localhost:8000`

### Ollama (IA de texte pour améliorer les prompts)

Ollama doit être installé et démarré pour utiliser la fonctionnalité de chat IA locale.

**Démarrer Ollama :**

```powershell
# Si Ollama est installé mais pas en service :
ollama serve
```

Ou vérifiez qu'il tourne déjà :
```powershell
# Testez si Ollama répond
curl http://127.0.0.1:11434/api/tags
```

Par défaut, Ollama écoute sur `http://127.0.0.1:11434`. Vous pouvez configurer l'URL via les variables d'environnement `OLLAMA_BASE_URL` et `OLLAMA_MODEL` dans le backend.

## Démarrage complet (3 terminaux)

**Terminal 1 - Ollama :**
```powershell
ollama serve
```

**Terminal 2 - Backend FastAPI :**
```powershell
cd backend
python main.py
```

**Terminal 3 - Frontend Next.js :**
```powershell
cd frontend
npm run dev
```

> **Note :** Si Ollama est installé en service Windows, il démarre automatiquement et vous n'avez pas besoin du Terminal 1.

## Fonctionnalités couvertes

### Interface utilisateur
- **Prompt + Negative Prompt** : champs texte avec presets rapides (portrait, paysage, architecture, art conceptuel)
- **Paramètres essentiels** : sampler (Euler, DPM++, UniPC, DDIM), steps (10-60), CFG Scale (1-14), résolution (512/768/1024 ou dimensions personnalisables), seed, image count (1-4)
- **Sélecteur de modèles** façon CivitAI, avec rafraîchissement dynamique (scan des dossiers `backend/models` et infobulle s’ils sont absents)
- **Historique persistant** : sauvegarde automatique dans localStorage, chargement des prompts précédents avec un clic
- **Galerie responsive** : grille adaptative avec export PNG et copie base64
- **Indicateur de progression + file de jobs** : toutes les générations passent par la queue (pause/reprise/annulation)
- **Barre VRAM** : estimation en temps réel de la mémoire consommée selon la résolution, les LoRA, le mode vidéo/img2img, etc.
- **Image de base (img2img)** : possibilité d’importer une image avant le prompt avec contrôle de la force (denoise)
- **Vidéo** : mode Image→Vidéo ou Texte→Vidéo (génère automatiquement l’image de référence)
- **Chat IA locale** : conversation avec Ollama, possibilité de joindre une image pour analyse
- **Scan des modèles/LoRA** : bouton “Actualiser” qui détecte les nouveaux `.safetensors` déposés côté backend sans redémarrage

### Backend FastAPI
- **Endpoint `/generate`** : génération d'images (supporte désormais `init_image_base64` + `init_strength` pour du img2img)
- **Endpoint `/generate-video`** : génération vidéo (mode `img2vid` ou `text2vid`)
- **Endpoints `/models`, `/models/scan`, `/loras`, `/loras/scan`** : gestion dynamique des modèles/LoRA présents sur disque
- **Endpoints `/jobs`** : file persistante (crash recovery, pause/reprise/start/cancel/clear-completed)
- **Endpoint `/health`** : vérification de l'état du serveur et du device (CPU/GPU)
- **Téléchargement automatique** : récupération des modèles depuis Hugging Face si absents
- **Support CPU/GPU** : détection automatique avec optimisation (float16 sur GPU, float32 sur CPU), VAE/Attention Slicing et CPU offload (limité à 1 modèle en cache pour 8Go de VRAM). Utilisez `USE_CPU_OFFLOAD=true` si besoin.

> Par défaut, une large sélection de modèles (SDXL, Pony, Illustrious, etc.) est autorisée côté serveur. Pour restreindre ou activer d'autres mélanges SD1.5/SDXL/Pony, définissez la variable d’environnement `ENABLED_MODELS` (ex : `sdxl,realistic-vision,cyberrealistic-pony`) et relancez FastAPI.

Les poids des modèles sont lourds (plusieurs Go). Si vous manquez de place, vous pouvez supprimer certains fichiers dans `backend/models/`.

### Modèles "Cloud" (Récupérables automatiquement)
Ces modèles peuvent être supprimés sans crainte. Le backend les re-téléchargera depuis Hugging Face dès que vous les utiliserez ou via le script dédié.
- **SDXL Base** (`sdxlTurbo_fullVersion.safetensors`)
- **SDXL Turbo** (`sd_xl_turbo_1.0.safetensors`)
- **RealVis XL** (`RealVisXL_V4.0.safetensors`)
- **Realistic Vision** (`Realistic_Vision_V5.1-noVAE.safetensors`)
- **DreamShaper / MeinaMix / ChilloutMix**
- **Stable Diffusion 1.5** (`v1-5-pruned.safetensors`)

Pour tout re-télécharger d'un coup :
```powershell
cd backend
python download_models.py --all
```

### Modèles "Locaux" (À ne PAS supprimer)
Ne supprimez pas les fichiers suivants car ils n'ont pas de source automatique (sauf si vous les avez téléchargés vous-même ailleurs) :
- Le contenu du dossier `backend/models/lora/`.
- Les modèles spécifiques comme `waiIllustriousSDXL`, `pony-no-score`, `lucentxlPony`, etc.

## Tests

### Backend

```powershell
cd backend
pip install -r requirements-dev.txt
pytest
```

### Frontend

```powershell
cd frontend
npm install
npm run test:unit
```

