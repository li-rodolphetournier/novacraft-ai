# Backend – FastAPI + Diffusers

Ce serveur expose la route `POST /generate` pour piloter Stable Diffusion (SD1.5 ou SDXL) en local. Il s'appuie sur `diffusers` et charge les poids fournis par l'utilisateur depuis CivitiAI ou HuggingFace.

## Installation

```powershell
cd backend
python -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
pip install -r requirements.txt
# Installe Torch adapté à votre GPU / CPU
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

Déposez vos fichiers `.safetensors` dans `backend/models/` (ou ailleurs) puis pointez les variables d'environnement correspondantes avant de lancer le serveur :

```powershell
$env:MODEL_SDXL="models\sdxl_base_1.0.safetensors"
$env:MODEL_REALISTIC_VISION="models\realisticVision_v51VAE.safetensors"
# etc.
```

👉 Par défaut, seul **SDXL** est activé pour garantir la meilleure qualité. Pour débloquer d'autres presets plus tard, renseignez la variable `ENABLED_MODELS` avec une liste de clés séparées par des virgules :

```powershell
$env:ENABLED_MODELS="sdxl,realistic-vision,dreamshaper"
```

### Téléchargement automatique depuis Hugging Face

Le script `download_models.py` permet de télécharger automatiquement les modèles gratuits depuis Hugging Face.

**Lister les modèles disponibles :**
```powershell
cd backend
python download_models.py --list
```

**Télécharger un modèle spécifique :**
```powershell
python download_models.py sdxl
python download_models.py realistic-vision dreamshaper
```

**Télécharger tous les modèles disponibles :**
```powershell
python download_models.py --all
```

**Modèles gratuits disponibles depuis Hugging Face :**

**SDXL (haute qualité) :**
- `sdxl` - Stable Diffusion XL Base 1.0
- `sdxl-turbo` - SDXL Turbo (génération rapide)
- `dreamshaper-xl` - DreamShaper XL v2.0
- `juggernaut-xl` - Juggernaut XL v9
- `realvis-xl` - RealVisXL V4.0

**SD 1.5 (légers, rapides) :**
- `sd-1.5-base` - Stable Diffusion 1.5 Base
- `realistic-vision` - Realistic Vision V5.1
- `dreamshaper` - DreamShaper 8
- `chilloutmix` - ChilloutMix
- `deliberate` - Deliberate v2
- `meinamik` - MeinaMix V11

Au runtime, `main.py` essaie aussi de télécharger le fichier manquant avant de lever une erreur.

## File de jobs et reprise automatique

Les générateurs fonctionnent maintenant via une file persistante :

- `POST /generate` : crée un job d'image (retourne `job_id`)
- `POST /generate-video` : crée un job vidéo
- `GET /jobs` / `GET /jobs/{id}` : suivent la progression (statut, frames/images, résultat)
- `POST /jobs/{id}/pause` / `resume` / `start` : pause ou reprise (le bouton *start* force la priorité)
- `POST /jobs/{id}/cancel` : annule un job en file ou en cours
- `DELETE /jobs/{id}` : supprime un job (pending, terminé, annulé, etc.) de la file
- `POST /jobs/resume-all` : relance tous les jobs interrompus après un crash/veille
- `POST /jobs/clear-completed` : nettoie les jobs terminés/annulés

### Génération vidéo : image → vidéo ou texte → vidéo

`POST /generate-video` prend désormais un champ `mode` :

- `img2vid` (par défaut) : vous fournissez `init_image_base64`.
- `text2vid` : le backend génère une image de référence à partir de vos réglages (`image_settings`) avant de lancer Stable Video Diffusion. Idéal pour partir d’un simple prompt.

Exemple en mode texte :

```json
{
  "prompt": "A cyberpunk skyline at night",
  "negative_prompt": "low quality, blurry",
  "mode": "text2vid",
  "num_frames": 8,
  "fps": 6,
  "resolution": "512x512",
  "seed": -1,
  "image_settings": {
    "model": "sdxl",
    "steps": 30,
    "cfg_scale": 7,
    "sampler": "euler",
    "clip_skip": 2,
    "resolution": "1024x1024",
    "additional_loras": []
  }
}
```

Le job calcule d’abord l’image (steps configurables), puis enchaîne la conversion en vidéo tout en conservant la progression dans la file.

## Lancer le serveur

```powershell
uvicorn main:app --reload --port 8000
```

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

## Format de la requête

```json
{
  "prompt": "A cinematic shot of a retro sci-fi city",
  "negative_prompt": "blurry, low quality, watermark",
  "sampler": "euler",
  "steps": 30,
  "cfg_scale": 7.5,
  "resolution": "1024x1024",
  "seed": -1,
  "image_count": 2,
  "model": "sdxl"
}
```

La requête crée désormais un **job**. La réponse ressemble à :

```json
{
  "job_id": "job-1a2b3c4d",
  "status": "pending",
  "type": "image"
}
```

Utilisez ensuite `GET /jobs/{job_id}` pour récupérer l'avancement et les images une fois le job terminé.

