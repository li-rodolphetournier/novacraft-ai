# Guide de téléchargement des modèles

## ✅ Modèles téléchargés avec succès

Ces modèles ont été téléchargés automatiquement depuis Hugging Face :
- ✅ `sdxl` - Stable Diffusion XL Base 1.0
- ✅ `sdxl-turbo` - SDXL Turbo
- ✅ `realvis-xl` - RealVisXL V4.0
- ✅ `sd-1.5-base` - Stable Diffusion 1.5 Base

## ⚠️ Modèles nécessitant un téléchargement manuel

Certains modèles ne sont pas disponibles sur Hugging Face ou nécessitent une authentification. Tu peux les télécharger depuis **Civitai** (https://civitai.com) :

### Modèles SDXL

1. **DreamShaper XL** (`dreamshaper-xl`)
   - Civitai: https://civitai.com/models/112902
   - Télécharge le fichier `.safetensors`
   - Place-le dans `backend/models/` avec le nom `DreamShaperXL_v2.0.safetensors`

2. **Juggernaut XL** (`juggernaut-xl`)
   - Civitai: https://civitai.com/models/133005
   - Télécharge le fichier `.safetensors`
   - Place-le dans `backend/models/` avec le nom `Juggernaut-XL-v9.safetensors`

### Modèles SD 1.5

3. **ChilloutMix** (`chilloutmix`)
   - Civitai: https://civitai.com/models/6424
   - Télécharge le fichier `.safetensors`
   - Place-le dans `backend/models/` avec le nom `chilloutmix.safetensors`

4. **Deliberate** (`deliberate`)
   - Civitai: https://civitai.com/models/4823
   - Télécharge le fichier `.safetensors`
   - Place-le dans `backend/models/` avec le nom `deliberate_v2.safetensors`

## 📝 Modèles corrigés (maintenant disponibles)

Ces modèles ont été ajoutés à la configuration et devraient fonctionner maintenant :
- ✅ `realistic-vision` - Realistic Vision V5.1
- ✅ `dreamshaper` - DreamShaper 8
- ✅ `meinamik` - MeinaMix V11

Relance `python download_models.py realistic-vision dreamshaper meinamik` pour les télécharger.

## 🔧 Résolution des problèmes

### Erreur "KeyError"
Si tu vois une erreur `KeyError`, cela signifie que le modèle n'est pas dans `MODEL_CONFIG`. Les modèles `realistic-vision`, `dreamshaper` et `meinamik` ont été ajoutés, relance le script.

### Erreur "401 Unauthorized"
Certains modèles nécessitent une authentification Hugging Face :
```powershell
huggingface-cli login
```

### Erreur "404 Not Found"
Le repository n'existe pas sur Hugging Face. Télécharge le modèle depuis Civitai (voir ci-dessus).

### Fichier introuvable
Le nom du fichier peut être différent. Vérifie sur le repository Hugging Face ou Civitai le nom exact du fichier `.safetensors`.

## 📦 Alternative : Téléchargement depuis Hugging Face

Si un modèle est disponible sur Hugging Face mais avec un nom de fichier différent :

1. Va sur https://huggingface.co/[REPO_ID]/tree/main
2. Trouve le fichier `.safetensors`
3. Télécharge-le manuellement
4. Place-le dans `backend/models/` avec le nom attendu (voir `MODEL_CONFIG` dans `model_registry.py`)

## ✅ Vérifier les modèles téléchargés

```powershell
cd backend
python check_models.py
```

Cela liste tous les modèles configurés et indique s'ils sont présents ou manquants.

