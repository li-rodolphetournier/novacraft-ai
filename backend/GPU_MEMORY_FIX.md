# Guide de résolution des problèmes de mémoire GPU

## Problème : Erreur cuDNN `CUDNN_STATUS_INTERNAL_ERROR_HOST_ALLOCATION_FAILED`

Cette erreur indique que cuDNN (bibliothèque d'accélération GPU) ne peut pas allouer de la mémoire, souvent à cause de fragmentation ou de saturation.

## Solutions implémentées

### 1. Système de cache intelligent (LRU)
- **Maximum 1 modèle en cache** avec 8GB VRAM
- Déchargement automatique des modèles non utilisés
- Nettoyage de la mémoire avant chaque chargement

### 2. Optimisations mémoire
- ✅ **VAE Slicing** : Découpe le VAE pour économiser la mémoire
- ❌ **VAE Tiling** : Désactivé (cause des erreurs cuDNN)
- ✅ **Attention Slicing** : Découpe l'attention
- ✅ **Nettoyage agressif** : Garbage collection + `torch.cuda.empty_cache()`

### 3. Gestion d'erreurs améliorée
- Détection automatique des erreurs cuDNN/PyTorch
- Nettoyage automatique en cas d'erreur
- Suggestions personnalisées selon les paramètres

## Solutions selon le problème

### Option A : Forcer CPU Offload (plus stable)

Si tu continues à avoir des erreurs cuDNN, force l'utilisation de CPU offload :

```powershell
# Dans PowerShell, avant de lancer le serveur
$env:USE_CPU_OFFLOAD="true"
python -m uvicorn main:app --reload --port 8000
```

**Avantages :**
- ✅ Plus stable, évite les erreurs cuDNN
- ✅ Économise la VRAM

**Inconvénients :**
- ⚠️ Plus lent (mais toujours plus rapide que CPU pur)

### Option B : Réduire les paramètres de génération

Si tu veux garder le GPU direct, réduis :

1. **Résolution** : Utilise 1024x1024 au lieu de 1536x1536
2. **Nombre d'images** : Génère 1 image à la fois au lieu de 2-4
3. **Steps** : Utilise 20-30 steps au lieu de 50+

### Option C : Nettoyer manuellement le cache

Si tu as des problèmes, appelle l'endpoint de nettoyage :

```bash
curl -X POST http://localhost:8000/clear-cache
```

Ou depuis le frontend, tu peux ajouter un bouton qui appelle cet endpoint.

## Vérifier l'état de la mémoire

Appelle `/health` pour voir l'état actuel :

```bash
curl http://localhost:8000/health
```

Tu verras :
- `vram_total_gb` : VRAM totale
- `vram_allocated_gb` : VRAM utilisée
- `vram_reserved_gb` : VRAM réservée
- `vram_free_gb` : VRAM libre
- `cached_models` : Modèles actuellement en cache
- `cache_size` : Nombre de modèles en cache

## Recommandations pour 8GB VRAM

Avec une RTX 4070 (8GB), voici les paramètres recommandés :

**Génération rapide (GPU direct) :**
- Résolution : 1024x1024
- Images : 1
- Steps : 20-30
- Modèle : SDXL Turbo (plus rapide)

**Génération stable (CPU offload) :**
- Résolution : 1024x1024 ou 1536x1536
- Images : 1-2
- Steps : 30-50
- Modèle : N'importe quel SDXL

## Dépannage

### Si tu as toujours des erreurs cuDNN :

1. **Force CPU offload** :
   ```powershell
   $env:USE_CPU_OFFLOAD="true"
   ```

2. **Redémarre le serveur**

3. **Nettoie le cache** :
   ```bash
   curl -X POST http://localhost:8000/clear-cache
   ```

4. **Vérifie qu'un seul modèle est en cache** :
   ```bash
   curl http://localhost:8000/health
   ```

### Si les générations sont trop lentes :

1. **Désactive CPU offload** (si activé) :
   ```powershell
   $env:USE_CPU_OFFLOAD="false"
   ```

2. **Utilise SDXL Turbo** au lieu de SDXL standard

3. **Réduis les steps** à 20-25

## Notes techniques

- Le système nettoie automatiquement la mémoire avant et après chaque génération
- Le cache est limité à 1 modèle pour éviter la saturation
- VAE tiling est désactivé car il cause des erreurs cuDNN sur certaines configurations
- Le garbage collection Python est forcé après chaque nettoyage GPU

