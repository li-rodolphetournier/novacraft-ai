# Générateur IA d’Images (Next.js + FastAPI)

Projet local complet suivant le cahier des charges :

- **Frontend** : Next.js 15 (App Router, Tailwind) avec interface façon Leonardo.ai
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

Voir `backend/README.md` pour l’installation de Torch + Diffusers et les variables modèle (`MODEL_SDXL`, etc.). Le service expose `POST /generate` qui renvoie les images en base64.

## Fonctionnalités couvertes

### Interface utilisateur
- **Prompt + Negative Prompt** : champs texte avec presets rapides (portrait, paysage, architecture, art conceptuel)
- **Paramètres essentiels** : sampler (Euler, DPM++, UniPC, DDIM), steps (10-60), CFG Scale (1-14), résolution (512/768/1024), seed, image count (1-4)
- **Sélecteur de modèles** façon CivitAI (Realistic Vision, DreamShaper, MeinaMix, Juggernaut XL, SDXL base)
- **Historique persistant** : sauvegarde automatique dans localStorage, chargement des prompts précédents avec un clic
- **Galerie responsive** : grille adaptative avec export PNG et copie base64
- **Indicateur de progression** : barre de progression pendant la génération
- **Vérification des modèles** : détection automatique des modèles disponibles via l'endpoint `/models`

### Backend FastAPI
- **Endpoint `/generate`** : génération d'images avec retour base64
- **Endpoint `/models`** : liste des modèles disponibles avec statut d'installation
- **Endpoint `/health`** : vérification de l'état du serveur et du device (CPU/GPU)
- **Téléchargement automatique** : récupération des modèles depuis Hugging Face si absents
- **Support CPU/GPU** : détection automatique avec optimisation (float16 sur GPU, float32 sur CPU)

> Par défaut, seul le preset **SDXL** est autorisé côté serveur pour assurer la meilleure qualité. Lorsque vous voudrez activer d’autres mélanges SD1.5/SDXL, définissez la variable d’environnement `ENABLED_MODELS` (ex : `sdxl,realistic-vision`) et relancez FastAPI.

### Téléchargement auto des modèles

Depuis `backend/`, utilisez `python download_models.py --all` pour récupérer les poids listés dans le registre (Hugging Face Hub). Le backend tente aussi le téléchargement à la volée si un fichier manque.

