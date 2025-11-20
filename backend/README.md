# NovaCraft AI Backend – FastAPI + Diffusers

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

> Tous les jobs survivent au redémarrage. Utilisez `/jobs/resume-all` pour relancer automatiquement la file après une coupure.

### Scan automatique des modèles et LoRA

Déposez vos fichiers `.safetensors` dans les dossiers :
- `backend/models/` pour les checkpoints
- `backend/models/lora/` pour les LoRA

Puis déclenchez un scan (sans redémarrer) :

```powershell
# Depuis PowerShell
Invoke-RestMethod -Method Post http://localhost:8000/models/scan
Invoke-RestMethod -Method Post http://localhost:8000/loras/scan

# ou via curl
curl -X POST http://localhost:8000/models/scan
curl -X POST http://localhost:8000/loras/scan
```

Le frontend dispose également d’un bouton “Actualiser” qui appelle ces endpoints. Les entrées disparues (fichiers supprimés) sont retirées automatiquement.

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

### Mode img2img (image de base)

`POST /generate` accepte en option :
- `init_image_base64` : image de départ (PNG/JPEG encodé base64)
- `init_strength` : force (0.05–0.95). Plus la valeur est haute, plus l’IA s’éloigne de l’image initiale.

Cela correspond au bouton “Image de base (img2img)” dans l’interface.

## Lancer le serveur

```powershell
uvicorn main:app --reload --port 8000
```

## Chat IA locale + analyse d'images

L'endpoint `POST /chat` relaie vos messages vers Ollama (`OLLAMA_BASE_URL`, par défaut `http://127.0.0.1:11434`). Deux variables permettent de choisir les modèles :

| Variable | Par défaut | Usage |
| --- | --- | --- |
| `OLLAMA_MODEL` | `llama3.2:3b` | Modèle texte pour les conversations classiques |
| `OLLAMA_VISION_MODEL` | `llava:7b` | Modèle multimodal utilisé automatiquement dès qu'un message contient une image |

```powershell
$env:OLLAMA_MODEL="llama3.1:8b"
$env:OLLAMA_VISION_MODEL="llava:13b"
$env:OLLAMA_VISION_MODEL="llava:13b"
```

> ⚠️ Pensez à installer les modèles correspondants côté Ollama (`ollama pull llava:7b`). Si vous recevez “Je ne vois pas l’image…”, vérifiez que le modèle multimodal est bien présent ou remplacez `OLLAMA_VISION_MODEL` par un modèle vision compatible (LLaVA, Llava-phi, llama3.2-vision, etc.).

Chaque message peut transporter une liste `images` contenant des chaînes base64. Le frontend encode automatiquement vos fichiers, vous n'avez rien à faire.

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

