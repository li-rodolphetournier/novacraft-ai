from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Literal

from huggingface_hub import hf_hub_download


ModelKey = Literal[
    "realistic-vision",
    "dreamshaper",
    "meinamik",
    "sdxl",
    "cyberrealistic-pony",
    "tsunade-il",
    "wai-illustrious-sdxl",
    "wan22-enhanced-nsfw-camera",
    "hassaku-xl-illustrious-v32",
    "duchaiten-pony-xl",
    "lucentxl-pony",
    "ponydiffusion-v6-xl",
    "ishtars-gate-nsfw-sfw",
]

MODEL_CONFIG: Dict[ModelKey, Dict[str, str]] = {
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
}

LORA_LIBRARY: Dict[str, Dict[str, str | float]] = {
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
}


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
            local_dir_use_symlinks=False,
        )
    except Exception as err:  # pylint: disable=broad-except
        raise FileNotFoundError(
            f"Impossible de télécharger automatiquement {model_key}. "
            "Téléchargez le fichier depuis Hugging Face ou définissez MODEL_* vers un chemin valide."
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

