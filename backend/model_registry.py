from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict

from huggingface_hub import hf_hub_download


ModelKey = str

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
LORA_DIR = MODELS_DIR / "lora"
STORAGE_DIR = BASE_DIR / "storage"
STORAGE_DIR.mkdir(parents=True, exist_ok=True)
CUSTOM_MODELS_FILE = STORAGE_DIR / "custom_models.json"
CUSTOM_LORAS_FILE = STORAGE_DIR / "custom_loras.json"

for custom_path in (CUSTOM_MODELS_FILE, CUSTOM_LORAS_FILE):
    if not custom_path.exists():
        custom_path.write_text("[]", encoding="utf-8")


def _load_json_list(path: Path) -> list[Dict[str, Any]]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []


def _save_json_list(path: Path, data: list[Dict[str, Any]]) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _slugify(name: str, existing: set[str], prefix: str) -> str:
    base = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or prefix
    candidate = base
    counter = 1
    while candidate in existing:
        candidate = f"{base}-{counter}"
        counter += 1
    return candidate


MODEL_CONFIG: Dict[ModelKey, Dict[str, Any]] = {
    "sdxl": {
        "path": os.getenv("MODEL_SDXL", "models/sdxlTurbo_fullVersion.safetensors"),
        "variant": "sdxl",
    },
    "cyberrealistic-pony": {
        # Base SDXL (ex: sdxlTurbo_fullVersion) + LoRA cyberrealisticPony_v141
        "path": os.getenv(
            "MODEL_CYBERREALISTIC_BASE", "models/sdxlTurbo_fullVersion.safetensors"
        ),
        "variant": "sdxl",
        "lora_path": os.getenv(
            "MODEL_CYBERREALISTIC_LORA", "models/cyberrealisticPony_v141.safetensors"
        ),
        "lora_label": "CyberRealistic Pony v1.41",
    },
    "tsunade-il": {
        # Base SDXL (par ex. waiIllustriousSDXL) + LoRA Tsunade_iL
        "path": os.getenv(
            "MODEL_TSUNADE_BASE", "models/waiIllustriousSDXL_v140.safetensors"
        ),
        "variant": "sdxl",
        "lora_path": os.getenv(
            "MODEL_TSUNADE_LORA", "models/Tsunade_iL.safetensors"
        ),
        "lora_label": "Tsunade iL",
    },
    "wai-illustrious-sdxl": {
        "path": os.getenv(
            "MODEL_WAI_ILLUSTRIOUS_SDXL", "models/waiIllustriousSDXL_v140.safetensors"
        ),
        "variant": "sdxl",
    },
    "wan22-enhanced-nsfw-camera": {
    # Base SDXL (par ex. ton sdxlTurbo_fullVersion) + LoRA wan22
    "path": os.getenv(
        "MODEL_WAN22_BASE", "models/sdxlTurbo_fullVersion.safetensors"
    ),
    "variant": "sdxl",
    "lora_path": os.getenv(
        "MODEL_WAN22_LORA",
        "models/wan22EnhancedNSFWCameraPrompt_nsfwFASTMOVEFP8High.safetensors",
        ),
        "lora_label": "wan22 Enhanced NSFW Camera",
    },
    "hassaku-xl-illustrious-v32": {
        "path": os.getenv(
            "MODEL_HASSAKU_XL_ILLUSTRIOUS_V32",
            "models/hassakuXLIllustrious_v32.safetensors",
        ),
        "variant": "sdxl",
    },
    "duchaiten-pony-xl": {
        "path": os.getenv(
            "MODEL_DUCHAITEN_PONY_XL", "models/pony-no-score_v4.0.safetensors"
        ),
        "variant": "sdxl",
    },
    "lucentxl-pony": {
        "path": os.getenv(
            "MODEL_LUCENTXL_PONY", "models/lucentxlPonyByKlaabu_b20.safetensors"
        ),
        "variant": "sdxl",
    },
    "ponydiffusion-v6-xl": {
        "path": os.getenv(
            "MODEL_PONYDIFFUSION_V6_XL", "models/ponyDiffusionV6XL_v6StartWithThisOne.safetensors"
        ),
        "variant": "sdxl",
    },
    "ishtars-gate-nsfw-sfw": {
        "path": os.getenv(
            "MODEL_ISHTARS_GATE", "models/ishtarsGateNSFWSFW_v10.safetensors"
        ),
        "variant": "sdxl",
    },
    "sdxl-turbo": {
        "path": os.getenv("MODEL_SDXL_TURBO", "models/sdxl_turbo_1.0.safetensors"),
        "variant": "sdxl",
    },
    "diving-illustrious": {
        "path": os.getenv(
            "MODEL_DIVING_ILLUSTRIOUS", "models/divingIllustrious_nijiMutedColorReal.safetensors"
        ),
        "variant": "sdxl",
    },
    "dreamshaper-xl": {
        "path": os.getenv(
            "MODEL_DREAMSHAPER_XL", "models/DreamShaperXL_v2.0.safetensors"
        ),
        "variant": "sdxl",
    },
    "juggernaut-xl": {
        "path": os.getenv(
            "MODEL_JUGGERNAUT_XL", "models/Juggernaut-XL-v9.safetensors"
        ),
        "variant": "sdxl",
    },
    "realvis-xl": {
        "path": os.getenv("MODEL_REALVIS_XL", "models/RealVisXL_V4.0.safetensors"),
        "variant": "sdxl",
    },
    "sd-1.5-base": {
        "path": os.getenv("MODEL_SD_1_5_BASE", "models/v1-5-pruned.safetensors"),
        "variant": "sd15",
    },
    "chilloutmix": {
        "path": os.getenv("MODEL_CHILLOUTMIX", "models/chilloutmix.safetensors"),
        "variant": "sd15",
    },
    "deliberate": {
        "path": os.getenv("MODEL_DELIBERATE", "models/deliberate_v2.safetensors"),
        "variant": "sd15",
    },
    "realistic-vision": {
        "path": os.getenv(
            "MODEL_REALISTIC_VISION", "models/Realistic_Vision_V5.1-noVAE.safetensors"
        ),
        "variant": "sd15",
    },
    "dreamshaper": {
        "path": os.getenv(
            "MODEL_DREAMSHAPER", "models/DreamShaper_8_pruned.safetensors"
        ),
        "variant": "sd15",
    },
    "meinamik": {
        "path": os.getenv("MODEL_MEINAMIK", "models/MeinaMix_v11.safetensors"),
        "variant": "sd15",
    },
}

MODEL_SOURCES: Dict[ModelKey, Dict[str, str]] = {
    "sdxl": {
        "repo_id": "stabilityai/stable-diffusion-xl-base-1.0",
        "filename": "sdxl_base_1.0.safetensors",
    },
    "realistic-vision": {
        "repo_id": "SG161222/Realistic_Vision_V5.1_noVAE",
        "filename": "Realistic_Vision_V5.1-noVAE.safetensors",
    },
    "dreamshaper": {
        "repo_id": "Lykon/dreamshaper-8",
        "filename": "DreamShaper_8_pruned.safetensors",
    },
    "meinamik": {
        "repo_id": "Meina/MeinaMix_V11",
        "filename": "MeinaMix_v11.safetensors",
    },
    "sdxl-turbo": {
        "repo_id": "stabilityai/sdxl-turbo",
        "filename": "sd_xl_turbo_1.0_fp16.safetensors",
    },
    "dreamshaper-xl": {
        "repo_id": "Lykon/DreamShaper-XL",
        "filename": "DreamShaperXL_v2.0.safetensors",
    },
    "juggernaut-xl": {
        "repo_id": "RunDiffusion/Juggernaut-XL-v9",
        "filename": "Juggernaut-XL-v9.safetensors",
    },
    "realvis-xl": {
        "repo_id": "SG161222/RealVisXL_V4.0",
        "filename": "RealVisXL_V4.0.safetensors",
    },
    "sd-1.5-base": {
        "repo_id": "runwayml/stable-diffusion-v1-5",
        "filename": "v1-5-pruned.safetensors",
    },
    "chilloutmix": {
        "repo_id": "Lykon/chilloutmix",
        "filename": "chilloutmix.safetensors",
    },
    "deliberate": {
        "repo_id": "XpucT/Deliberate",
        "filename": "deliberate_v2.safetensors",
    },
}

LORA_LIBRARY: Dict[str, Dict[str, Any]] = {
    "expressive-h": {
        "label": "ExpressiveH (Hentai LoRa Style)",
        "path": os.getenv("LORA_EXPRESSIVE_H", "models/lora/Expressive_H.safetensors"),
        "default_weight": 0.45,
        "description": "Style anime/hentai très expressif",
        "nsfw": True,
    },
    "incase-ponyxl": {
        "label": "Incase Style [PonyXL]",
        "path": os.getenv(
            "LORA_INCASE_PONYXL", "models/lora/incase-ilff-v3-4_ponyxl.safetensors"
        ),
        "default_weight": 0.5,
        "description": "Style Incase v3.0 optimisé PonyXL",
        "nsfw": True,
    },
    "g0th1c-pxl": {
        "label": "G0th1c PXL",
        "path": os.getenv("LORA_G0TH1C_PXL", "models/lora/g0th1cPXL.safetensors"),
        "default_weight": 0.5,
        "description": "Style gothique optimisé PonyXL",
        "nsfw": True,
    },
    "bs-pony-alpha": {
        "label": "BS Pony Alpha 1.0",
        "path": os.getenv("LORA_BS_PONY_ALPHA", "models/lora/BS- Pony_alpha1.0_rank4.safetensors"),
        "default_weight": 0.6,
        "description": "LoRA BS Pony alpha (rank4)",
        "nsfw": True,
    },
    "ps-alpha": {
        "label": "PS Alpha 1.0",
        "path": os.getenv("LORA_PS_ALPHA", "models/lora/PS_alpha1.0_rank4.safetensors"),
        "default_weight": 0.55,
        "description": "LoRA PS alpha polyvalente (rank4)",
        "nsfw": True,
    },
    "pts-alpha": {
        "label": "PTS Alpha 1.0",
        "path": os.getenv("LORA_PTS_ALPHA", "models/lora/PTS_alpha1.0_rank4.safetensors"),
        "default_weight": 0.55,
        "description": "LoRA PTS alpha (rank4)",
        "nsfw": True,
    },
    "wan22-i2v": {
        "label": "Wan 2.2 I2V A14B",
        "path": os.getenv(
            "LORA_WAN22_I2V",
            "models/lora/Wan_2_2_I2V_A14B_HIGH_lightx2v_4step_lora_v1030_rank_64_bf16.safetensors",
        ),
        "default_weight": 0.4,
        "description": "Optimisée pour les conversions image→vidéo rapides.",
        "nsfw": False,
    },
    "morii-gothic": {
        "label": "MoriiMee Gothic Pony",
        "path": os.getenv(
            "LORA_MORII_GOTHIC",
            "models/lora/MoriiMee_Gothic_Niji_Style__Pony_LoRA_V1_Redux.safetensors",
        ),
        "default_weight": 0.45,
        "description": "Style gothique kawaï (MoriiMee).",
        "nsfw": False,
    },
    "incase-style-alt": {
        "label": "Incase Style PonyXL (alt)",
        "path": os.getenv(
            "LORA_INCASE_STYLE_ALT",
            "models/lora/incase_style_v3-1_ponyxl_ilff.safetensors",
        ),
        "default_weight": 0.5,
        "description": "Variante Incase stylisée (style v3.1).",
        "nsfw": True,
    },
}


CUSTOM_MODELS_REGISTRY: list[Dict[str, Any]] = _load_json_list(CUSTOM_MODELS_FILE)
for entry in CUSTOM_MODELS_REGISTRY:
    key = entry.get("key")
    path = entry.get("path")
    if not key or not path:
        continue
    MODEL_CONFIG[key] = {
        "path": path,
        "variant": entry.get("variant", "sdxl"),
        "label": entry.get("label", key.replace("-", " ").title()),
        **{k: entry.get(k) for k in ("lora_path", "lora_label") if entry.get(k)},
    }

CUSTOM_LORA_REGISTRY: list[Dict[str, Any]] = _load_json_list(CUSTOM_LORAS_FILE)
for entry in CUSTOM_LORA_REGISTRY:
    key = entry.get("key")
    path = entry.get("path")
    if not key or not path:
        continue
    LORA_LIBRARY[key] = {
        "label": entry.get("label", key.replace("-", " ").title()),
        "path": path,
        "default_weight": entry.get("default_weight", 0.5),
        "description": entry.get("description", "LoRA importé depuis le dossier local."),
        "nsfw": bool(entry.get("nsfw", False)),
    }


def scan_custom_models() -> list[Dict[str, Any]]:
    existing_paths = {str(Path(config["path"]).expanduser().resolve()) for config in MODEL_CONFIG.values()}
    added_entries: list[Dict[str, Any]] = []
    removed_keys: list[str] = []

    # Retirer les entrées dont le fichier n'existe plus
    for entry in list(CUSTOM_MODELS_REGISTRY):
        path = entry.get("path")
        key = entry.get("key")
        if not path or not key:
            continue
        if not Path(path).exists() and key in MODEL_CONFIG:
            CUSTOM_MODELS_REGISTRY.remove(entry)
            MODEL_CONFIG.pop(key, None)
            removed_keys.append(key)

    if not MODELS_DIR.exists():
        if removed_keys:
            _save_json_list(CUSTOM_MODELS_FILE, CUSTOM_MODELS_REGISTRY)
        return added_entries

    for file_path in MODELS_DIR.glob("*.safetensors"):
        resolved = str(file_path.resolve())
        if resolved in existing_paths:
            continue
        existing_paths.add(resolved)
        key = _slugify(file_path.stem, set(MODEL_CONFIG.keys()), "custom-model")
        variant_hint = "sdxl" if "xl" in file_path.stem.lower() or "sdxl" in file_path.stem.lower() else "sd15"
        entry = {
            "key": key,
            "path": resolved,
            "variant": variant_hint,
            "label": file_path.stem.replace("_", " ").replace("-", " ").title(),
        }
        CUSTOM_MODELS_REGISTRY.append(entry)
        MODEL_CONFIG[key] = {
            "path": resolved,
            "variant": variant_hint,
            "label": entry["label"],
        }
        added_entries.append(entry)

    if added_entries:
        _save_json_list(CUSTOM_MODELS_FILE, CUSTOM_MODELS_REGISTRY)
    elif removed_keys:
        _save_json_list(CUSTOM_MODELS_FILE, CUSTOM_MODELS_REGISTRY)
    return added_entries


def scan_custom_loras() -> list[Dict[str, Any]]:
    existing_paths = {
        str(Path(config["path"]).expanduser().resolve()) for config in LORA_LIBRARY.values()
    }
    added_entries: list[Dict[str, Any]] = []
    removed_keys: list[str] = []

    for entry in list(CUSTOM_LORA_REGISTRY):
        path = entry.get("path")
        key = entry.get("key")
        if not path or not key:
            continue
        if not Path(path).exists() and key in LORA_LIBRARY:
            CUSTOM_LORA_REGISTRY.remove(entry)
            LORA_LIBRARY.pop(key, None)
            removed_keys.append(key)

    if not LORA_DIR.exists():
        if removed_keys:
            _save_json_list(CUSTOM_LORAS_FILE, CUSTOM_LORA_REGISTRY)
        return added_entries

    for file_path in LORA_DIR.glob("*.safetensors"):
        resolved = str(file_path.resolve())
        if resolved in existing_paths:
            continue
        existing_paths.add(resolved)
        key = _slugify(file_path.stem, set(LORA_LIBRARY.keys()), "custom-lora")
        entry = {
            "key": key,
            "path": resolved,
            "label": file_path.stem.replace("_", " ").replace("-", " ").title(),
            "default_weight": 0.5,
            "description": "LoRA importé automatiquement.",
            "nsfw": False,
        }
        CUSTOM_LORA_REGISTRY.append(entry)
        LORA_LIBRARY[key] = {
            "label": entry["label"],
            "path": resolved,
            "default_weight": entry["default_weight"],
            "description": entry["description"],
            "nsfw": entry["nsfw"],
        }
        added_entries.append(entry)

    if added_entries:
        _save_json_list(CUSTOM_LORAS_FILE, CUSTOM_LORA_REGISTRY)
    elif removed_keys:
        _save_json_list(CUSTOM_LORAS_FILE, CUSTOM_LORA_REGISTRY)
    return added_entries


def ensure_model_file(model_key: ModelKey) -> Path:
    """Ensure the model file exists locally; download from Hugging Face if possible."""
    config = MODEL_CONFIG[model_key]
    model_path = Path(config["path"]).expanduser()

    if model_path.exists():
        return model_path

    source = MODEL_SOURCES.get(model_key)
    if not source:
        return model_path

    target_dir = model_path.parent
    target_dir.mkdir(parents=True, exist_ok=True)

    try:
        downloaded = hf_hub_download(
            repo_id=source["repo_id"],
            filename=source["filename"],
            local_dir=str(target_dir),
        )
    except FileNotFoundError as err:
        raise FileNotFoundError(
            f"Fichier '{source['filename']}' introuvable dans le repository '{source['repo_id']}'. "
            f"Le fichier peut avoir un nom différent ou ne pas être disponible. "
            f"Vérifiez sur https://huggingface.co/{source['repo_id']}/tree/main pour voir les fichiers disponibles."
        ) from err
    except Exception as err:  # pylint: disable=broad-except
        error_msg = str(err)
        if "401" in error_msg or "Unauthorized" in error_msg:
            raise FileNotFoundError(
                f"Authentification requise pour '{source['repo_id']}'. "
                f"Connectez-vous avec 'huggingface-cli login' ou téléchargez le fichier manuellement depuis "
                f"https://huggingface.co/{source['repo_id']}"
            ) from err
        elif "404" in error_msg or "Not Found" in error_msg:
            raise FileNotFoundError(
                f"Repository '{source['repo_id']}' introuvable. "
                f"Vérifiez que le repository existe sur https://huggingface.co/{source['repo_id']} "
                f"ou téléchargez le modèle depuis Civitai (https://civitai.com)"
            ) from err
        else:
            raise FileNotFoundError(
                f"Impossible de télécharger automatiquement {model_key} depuis '{source['repo_id']}'. "
                f"Erreur: {error_msg}. "
                f"Téléchargez le fichier manuellement depuis Hugging Face ou Civitai."
            ) from err

    downloaded_path = Path(downloaded)
    if downloaded_path != model_path:
        downloaded_path.replace(model_path)

    return model_path


def ensure_lora_file(lora_key: str) -> Path:
    """Retourne le chemin du fichier LoRA et vérifie son existence."""
    if lora_key not in LORA_LIBRARY:
        raise KeyError(f"LoRA '{lora_key}' non déclarée dans LORA_LIBRARY.")
    config = LORA_LIBRARY[lora_key]
    lora_path = Path(config["path"]).expanduser()
    if not lora_path.exists():
        raise FileNotFoundError(
            f"LoRA '{config['label']}' introuvable. Placez le fichier à {lora_path}"
        )
    return lora_path

