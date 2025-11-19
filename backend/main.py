from __future__ import annotations

import base64
import json
import os
import random
import tempfile
import time
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Literal, Tuple
import uuid

import imageio
import numpy as np
import requests
import torch
from diffusers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerDiscreteScheduler,
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
    StableVideoDiffusionPipeline,
    UniPCMultistepScheduler,
)
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel, Field

from job_manager import (
    JobCancelledError,
    JobManager,
    JobPauseRequested,
)
from model_registry import (
    LORA_LIBRARY,
    MODEL_CONFIG,
    MODEL_SOURCES,
    ModelKey,
    ensure_lora_file,
    ensure_model_file,
)


SamplerName = Literal["euler", "dpmpp_2s", "unipc", "ddim"]
ResolutionKey = Literal["512x512", "768x768", "1024x1024", "1536x1536"]


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")

# Chemins de stockage
STORAGE_DIR = Path(__file__).parent / "storage"
STORAGE_IMAGES_DIR = STORAGE_DIR / "images"
STORAGE_VIDEOS_DIR = STORAGE_DIR / "videos"
STORAGE_IMAGES_JSON = STORAGE_DIR / "gallery-images.json"
STORAGE_VIDEOS_JSON = STORAGE_DIR / "gallery-videos.json"
STORAGE_HISTORY_JSON = STORAGE_DIR / "history.json"
JOBS_STORAGE_DIR = STORAGE_DIR / "jobs"
MAX_HISTORY_ENTRIES = 500

# Créer les dossiers s'ils n'existent pas
STORAGE_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
STORAGE_VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
STORAGE_HISTORY_JSON.parent.mkdir(parents=True, exist_ok=True)

for json_path in (STORAGE_IMAGES_JSON, STORAGE_VIDEOS_JSON, STORAGE_HISTORY_JSON):
    if not json_path.exists():
        json_path.write_text("[]", encoding="utf-8")

# Gestionnaire de jobs persistant
job_manager = JobManager(JOBS_STORAGE_DIR)


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=3, description="Main positive prompt")
    negative_prompt: str | None = Field(
        default="", description="What to avoid in the image"
    )
    sampler: SamplerName = Field(default="euler")
    steps: int = Field(default=30, ge=5, le=100)
    cfg_scale: float = Field(default=7.0, ge=1, le=20)
    resolution: ResolutionKey | str = Field(default="1024x1024")  # Accepte aussi des résolutions custom
    seed: int = Field(default=-1)
    clip_skip: int = Field(default=2, ge=1, le=12)
    image_count: int = Field(default=1, ge=1, le=4)
    model: ModelKey = Field(default="sdxl")
    width: int | None = Field(default=None, ge=256, le=2048, description="Custom width (overrides resolution)")
    height: int | None = Field(default=None, ge=256, le=2048, description="Custom height (overrides resolution)")
    additional_loras: list[LoRAInput] = Field(default_factory=list)


class GeneratedImage(BaseModel):
    id: str
    seed: int
    base64: str
    model: ModelKey | None = None
    sampler: SamplerName | None = None
    steps: int | None = None
    cfg_scale: float | None = None
    resolution: str | None = None
    prompt: str | None = None
    negative_prompt: str | None = None
    clip_skip: int | None = None
    loras: list["AppliedLoRA"] | None = None


class GenerateResponse(BaseModel):
    images: list[GeneratedImage]
    model: ModelKey
    sampler: SamplerName
    steps: int
    cfg_scale: float
    resolution: ResolutionKey | str
    duration_seconds: float


class LoRAInput(BaseModel):
    key: str = Field(..., description="Identifiant interne (LORA_LIBRARY)")
    weight: float = Field(default=0.5, ge=-3.0, le=3.0)


class AppliedLoRA(BaseModel):
    key: str
    label: str
    weight: float
    type: str = "LoRA"


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    model: str | None = None
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    max_tokens: int | None = Field(default=None, ge=16, le=4096)


class ChatResponseModel(BaseModel):
    message: ChatMessage
    raw: Dict[str, Any] | None = None


class GenerateVideoRequest(BaseModel):
    prompt: str = Field(..., min_length=3)
    negative_prompt: str | None = ""
    num_frames: int = Field(default=8, ge=6, le=16)  # Réduit pour économiser VRAM
    fps: int = Field(default=6, ge=3, le=30)
    resolution: ResolutionKey | str = "512x512"
    seed: int = -1
    init_image_base64: str | None = None
    mode: Literal["img2vid", "text2vid"] = "img2vid"
    image_settings: GenerateRequest | None = None


class GeneratedVideo(BaseModel):
    mp4_base64: str
    id: str | None = None


class GenerateVideoResponse(BaseModel):
    video: GeneratedVideo
    duration_seconds: float


RESOLUTIONS: Dict[ResolutionKey, Tuple[int, int]] = {
    "512x512": (512, 512),
    "768x768": (768, 768),
    "1024x1024": (1024, 1024),
    "1536x1536": (1536, 1536),
}


SAMPLERS = {
    "euler": EulerDiscreteScheduler,
    "dpmpp_2s": DPMSolverMultistepScheduler,
    "unipc": UniPCMultistepScheduler,
    "ddim": DDIMScheduler,
}


# Cache des pipelines avec gestion de mémoire limitée
# Avec 8GB VRAM, on ne peut garder qu'un seul modèle SDXL en mémoire
PIPELINE_CACHE: Dict[ModelKey, StableDiffusionPipeline | StableDiffusionXLPipeline] = {}
PIPELINE_CACHE_ORDER: list[ModelKey] = []  # Ordre d'utilisation (LRU)
MAX_CACHED_MODELS = 1  # Maximum 1 modèle en cache avec 8GB VRAM
VIDEO_PIPELINE: StableVideoDiffusionPipeline | None = None

ENABLED_MODELS = {
    key.strip()
    for key in os.getenv(
        "ENABLED_MODELS",
        "sdxl,sdxl-turbo,dreamshaper-xl,realistic-vision,dreamshaper,cyberrealistic-pony,tsunade-il,wai-illustrious-sdxl,wan22-enhanced-nsfw-camera,hassaku-xl-illustrious-v32,duchaiten-pony-xl,lucentxl-pony,ponydiffusion-v6-xl,ishtars-gate-nsfw-sfw",
    ).split(",")
    if key.strip()
}


def _clear_gpu_memory():
    """Nettoie la mémoire GPU en libérant les caches."""
    if DEVICE == "cuda":
        try:
            # Libère le cache PyTorch
            torch.cuda.empty_cache()
            # Synchronise pour s'assurer que toutes les opérations sont terminées
            torch.cuda.synchronize()
            # Force la collecte de garbage Python
            import gc
            gc.collect()
            # Libère à nouveau après garbage collection
            torch.cuda.empty_cache()
        except Exception:
            pass


def _unload_pipeline(model_key: ModelKey):
    """Décharge un pipeline de la mémoire GPU."""
    if model_key in PIPELINE_CACHE:
        pipe = PIPELINE_CACHE[model_key]
        try:
            # Déplace le pipeline sur CPU pour libérer la VRAM
            if hasattr(pipe, "to"):
                pipe = pipe.to("cpu")
            # Supprime les références
            del pipe
            del PIPELINE_CACHE[model_key]
            if model_key in PIPELINE_CACHE_ORDER:
                PIPELINE_CACHE_ORDER.remove(model_key)
            print(f"[INFO] Modèle '{model_key}' déchargé de la mémoire GPU")
        except Exception as e:
            print(f"[WARN] Erreur lors du déchargement de '{model_key}': {e}")
        finally:
            _clear_gpu_memory()


def load_pipeline(model_key: ModelKey):
    """Charge un pipeline avec gestion intelligente de la mémoire GPU."""
    # Si le modèle est déjà en cache, le mettre en premier (LRU)
    if model_key in PIPELINE_CACHE:
        if model_key in PIPELINE_CACHE_ORDER:
            PIPELINE_CACHE_ORDER.remove(model_key)
        PIPELINE_CACHE_ORDER.append(model_key)
        return PIPELINE_CACHE[model_key]

    # Si le cache est plein, décharger le modèle le moins récemment utilisé
    while len(PIPELINE_CACHE) >= MAX_CACHED_MODELS and PIPELINE_CACHE_ORDER:
        oldest_key = PIPELINE_CACHE_ORDER.pop(0)
        if oldest_key != model_key:  # Ne pas décharger celui qu'on charge
            _unload_pipeline(oldest_key)

    # Nettoyer la mémoire GPU avant de charger un nouveau modèle
    _clear_gpu_memory()

    config = MODEL_CONFIG[model_key]
    model_path = ensure_model_file(model_key)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model weights not found for {model_key}! Expected at {model_path}"
        )

    variant = config["variant"]

    if variant == "sdxl":
        pipe: StableDiffusionXLPipeline | StableDiffusionPipeline = (
            StableDiffusionXLPipeline.from_single_file(
                model_path, torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32
            )
        )
    else:
        pipe = StableDiffusionPipeline.from_single_file(
            model_path, torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32
        )

    # Initialise le suivi des LoRAs déjà chargés
    pipe._loaded_loras = set()  # type: ignore[attr-defined]
    pipe._default_lora_adapter = None  # type: ignore[attr-defined]
    pipe._default_lora_label = None  # type: ignore[attr-defined]
    pipe._default_lora_weight = 1.0  # type: ignore[attr-defined]

    # Charge un éventuel LoRA associé (ex: Tsunade_iL sur base waiIllustrious SDXL)
    lora_path_str = config.get("lora_path")
    if lora_path_str:
        lora_path = Path(lora_path_str).expanduser()
        if lora_path.exists():
            try:
                adapter_name = f"default-{model_key}"
                lora_label = config.get("lora_label", adapter_name)
                print(f"[INFO] Chargement du LoRA '{lora_label}' depuis {lora_path.name}...")
                with torch.enable_grad():
                    pipe.load_lora_weights(lora_path, adapter_name=adapter_name)
                pipe.set_adapters([adapter_name])
                pipe._loaded_loras.add(adapter_name)  # type: ignore[attr-defined]
                pipe._default_lora_adapter = adapter_name  # type: ignore[attr-defined]
                pipe._default_lora_label = lora_label  # type: ignore[attr-defined]
                pipe._default_lora_weight = float(config.get("lora_weight", 1.0))  # type: ignore[attr-defined]
                print(f"[INFO] ✅ LoRA '{lora_label}' chargé avec succès")
            except ValueError as err:
                error_msg = str(err)
                if "Invalid LoRA checkpoint" in error_msg or "not a valid LoRA" in error_msg:
                    print(f"[WARN] ⚠️  Le fichier '{lora_path.name}' n'est pas un LoRA valide.")
                    print(f"       Il peut s'agir d'un modèle complet ou d'un format incompatible.")
                    print(f"       Le modèle '{model_key}' sera chargé sans ce LoRA.")
                    print(f"       Erreur: {error_msg}")
                else:
                    print(f"[WARN] ⚠️  Impossible de charger le LoRA '{lora_path.name}': {error_msg}")
            except Exception as err:  # pylint: disable=broad-except
                print(f"[WARN] ⚠️  Erreur lors du chargement du LoRA '{lora_path.name}': {type(err).__name__}: {err}")
                print(f"       Le modèle '{model_key}' sera chargé sans ce LoRA.")
        else:
            print(f"[WARN] ⚠️  LoRA '{lora_path}' introuvable pour {model_key}.")
            print(f"       Le modèle sera chargé sans ce LoRA.")
    else:
        if hasattr(pipe, "disable_lora"):
            pipe.disable_lora()

    # Optimisation GPU : utilise to("cuda") si assez de VRAM, sinon CPU offload
    if DEVICE == "cuda":
        # Vérifie la VRAM disponible (approximatif)
        try:
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            # Si 8GB ou plus VRAM, charge tout en GPU (plus rapide)
            # Avec 8GB exactement, on peut charger SDXL en GPU
            # Avec 8GB VRAM, on peut charger en GPU mais avec optimisations agressives
            # ou utiliser CPU offload pour plus de stabilité
            use_cpu_offload = os.getenv("USE_CPU_OFFLOAD", "false").lower() == "true"
            
            if use_cpu_offload or vram_gb < 7.5:
                # CPU offload : plus lent mais plus stable, évite les erreurs cuDNN
                pipe.enable_model_cpu_offload()
                print(f"[INFO] CPU offload activé ({vram_gb:.1f}GB VRAM disponible)")
            else:
                # GPU direct : plus rapide mais nécessite plus de gestion mémoire
                pipe = pipe.to("cuda")
                
                # Optimisations mémoire pour réduire la consommation VRAM
                if hasattr(pipe, "enable_vae_slicing"):
                    pipe.enable_vae_slicing()  # Découpe le VAE pour économiser la mémoire
                # NOTE: VAE tiling désactivé car peut causer des erreurs cuDNN
                # if hasattr(pipe, "enable_vae_tiling"):
                #     pipe.enable_vae_tiling()  # Peut causer CUDNN_STATUS_INTERNAL_ERROR
                if hasattr(pipe, "enable_attention_slicing"):
                    pipe.enable_attention_slicing(1)  # Découpe l'attention
                
                # Désactiver le cache cuDNN pour éviter les problèmes de fragmentation
                torch.backends.cudnn.benchmark = False
                torch.backends.cudnn.deterministic = False
                
                print(f"[INFO] ✅ Modèle chargé en GPU ({vram_gb:.1f}GB VRAM disponible)")
                print(f"[INFO] Optimisations mémoire activées (VAE slicing, attention slicing)")
        except Exception as e:
            # Fallback sur CPU offload si erreur
            print(f"[WARN] Erreur lors de la vérification VRAM: {e}")
            pipe.enable_model_cpu_offload()
            print(f"[INFO] CPU offload activé (fallback)")
    else:
        pipe = pipe.to("cpu")
        print(f"[INFO] Modèle chargé en CPU (pas de GPU détecté)")

    # Ajouter au cache
    PIPELINE_CACHE[model_key] = pipe
    PIPELINE_CACHE_ORDER.append(model_key)
    
    # Nettoyer la mémoire après chargement
    _clear_gpu_memory()
    
    return pipe


def get_video_pipeline() -> StableVideoDiffusionPipeline:
    global VIDEO_PIPELINE  # noqa: PLW0603
    if VIDEO_PIPELINE is not None:
        return VIDEO_PIPELINE

    pipe = StableVideoDiffusionPipeline.from_pretrained(
        "stabilityai/stable-video-diffusion-img2vid-xt",
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    )
    if DEVICE == "cuda":
        # Sequential CPU offload économise plus de VRAM que model_cpu_offload
        pipe.enable_sequential_cpu_offload()
    VIDEO_PIPELINE = pipe
    return pipe


def _ensure_model_enabled(model_key: ModelKey) -> None:
    if model_key not in ENABLED_MODELS:
        allowed = ", ".join(sorted(ENABLED_MODELS)) or "sdxl"
        raise RuntimeError(
            f"Le modèle '{model_key}' est désactivé. "
            f"Modèles disponibles actuellement: {allowed}. "
            "Définissez la variable d'environnement ENABLED_MODELS pour en activer d'autres."
        )


def _compute_image_dimensions(payload: GenerateRequest) -> tuple[int, int]:
    if payload.width and payload.height:
        width = (payload.width // 64) * 64
        height = (payload.height // 64) * 64
    else:
        if payload.resolution in RESOLUTIONS:
            width, height = RESOLUTIONS[payload.resolution]
        else:
            try:
                parts = payload.resolution.split("x")
                width = (int(parts[0]) // 64) * 64
                height = (int(parts[1]) // 64) * 64
            except (ValueError, IndexError):
                width, height = RESOLUTIONS["1024x1024"]
    width = max(64, width)
    height = max(64, height)
    return width, height


def _build_history_entry(
    images: list[Dict[str, Any]],
    payload: GenerateRequest,
    width: int,
    height: int,
    seed: int,
) -> Dict[str, Any]:
    if not images:
        return {}
    return {
        "id": str(uuid.uuid4()),
        "prompt": payload.prompt,
        "negative_prompt": payload.negative_prompt or "",
        "model": payload.model,
        "timestamp": datetime.now().isoformat(),
        "thumbnail_id": images[0]["id"],
        "settings": {
            "sampler": payload.sampler,
            "steps": payload.steps,
            "cfg_scale": payload.cfg_scale,
            "clip_skip": payload.clip_skip,
            "resolution": f"{width}x{height}",
            "seed": seed,
            "use_aspect_ratio": bool(payload.width and payload.height),
            "aspect_ratio": payload.resolution if not (payload.width and payload.height) else None,
            "custom_width": width,
            "custom_height": height,
            "loras": [
                {"key": lora.key, "weight": lora.weight}
                for lora in payload.additional_loras
            ],
        },
    }


def run_image_generation_job(payload: GenerateRequest, job_id: str | None = None) -> Dict[str, Any]:
    _ensure_model_enabled(payload.model)

    if job_id:
        job_manager.raise_if_interrupted(job_id)

    pipe = load_pipeline(payload.model)
    apply_sampler(pipe, payload.sampler)

    width, height = _compute_image_dimensions(payload)
    seed = payload.seed if payload.seed >= 0 else random.randint(0, 2**32 - 1)
    generator = torch.Generator(device=DEVICE).manual_seed(seed)
    total_steps = max(payload.steps, 1)

    applied_loras_meta: list[AppliedLoRA] = []
    start = time.perf_counter()

    if DEVICE == "cuda" and hasattr(pipe, "enable_vae_slicing"):
        try:
            pipe.enable_vae_slicing()
        except Exception:
            pass

    try:
        applied_loras_meta = _apply_request_loras(pipe, payload)
        _clear_gpu_memory()

        if job_id:
            job_manager.raise_if_interrupted(job_id)
            job_manager.update_progress(
                job_id,
                current=0,
                total=total_steps,
                message="Initialisation du pipeline…",
            )

            def diffusion_callback(step: int, _timestep: int, _kwargs: Dict[str, Any]) -> None:
                current_step = min(total_steps, step + 1)
                job_manager.update_progress(
                    job_id,
                    current=current_step,
                    total=total_steps,
                    message=f"Étape {current_step}/{total_steps}",
                )
                job_manager.raise_if_interrupted(job_id)
        else:
            diffusion_callback = None

        with torch.inference_mode():
            try:
                result = pipe(
                    prompt=payload.prompt,
                    negative_prompt=payload.negative_prompt or "",
                    guidance_scale=payload.cfg_scale,
                    num_inference_steps=payload.steps,
                    width=width,
                    height=height,
                    generator=generator,
                    num_images_per_prompt=payload.image_count,
                    callback=diffusion_callback,
                    callback_steps=1,
                )
            except (torch.cuda.OutOfMemoryError, RuntimeError) as err:
                _clear_gpu_memory()
                for cached_key in list(PIPELINE_CACHE.keys()):
                    if cached_key != payload.model:
                        _unload_pipeline(cached_key)
                _clear_gpu_memory()

                suggestions = []
                if payload.image_count > 1:
                    suggestions.append(f"réduire le nombre d'images ({payload.image_count} → 1)")
                if width * height > 1024 * 1024:
                    suggestions.append(f"réduire la résolution ({width}x{height} → 1024x1024)")
                if payload.steps > 30:
                    suggestions.append(f"réduire les steps ({payload.steps} → 30)")
                suggestion_text = ". ".join(suggestions) if suggestions else "réduire la résolution ou le nombre d'images"

                raise RuntimeError(
                    "Mémoire GPU insuffisante (erreur cuDNN/PyTorch). "
                    f"Suggestions: {suggestion_text}. "
                    f"Erreur: {err}"
                ) from err

    finally:
        _reset_pipeline_adapters(pipe)
        _clear_gpu_memory()

    duration = time.perf_counter() - start
    images_meta: list[Dict[str, Any]] = []

    for idx, image in enumerate(result.images):
        if job_id:
            job_manager.raise_if_interrupted(job_id)

        image_id = f"{payload.model}-{seed + idx}-{int(time.time())}"
        base64_data = pil_to_base64(image)
        _save_image_file(image_id, base64_data)

        image_meta = {
            "id": image_id,
            "seed": seed + idx,
            "model": payload.model,
            "sampler": payload.sampler,
            "steps": payload.steps,
            "cfg_scale": payload.cfg_scale,
            "resolution": f"{width}x{height}",
            "prompt": payload.prompt,
            "negative_prompt": payload.negative_prompt or "",
            "clip_skip": payload.clip_skip,
            "loras": [
                {"key": lora.key, "label": lora.label, "weight": lora.weight, "type": lora.type}
                for lora in applied_loras_meta
            ] if applied_loras_meta else [],
            "timestamp": datetime.now().isoformat(),
            "file_path": f"images/{image_id}.png",
            "base64": base64_data,
        }

        stored_images = _load_storage_json(STORAGE_IMAGES_JSON)
        stored_images.append(image_meta)
        stored_images = stored_images[-1000:]
        _save_storage_json(STORAGE_IMAGES_JSON, stored_images)

        images_meta.append(image_meta)

        if job_id:
            job_manager.update_progress(
                job_id,
                current=total_steps,
                total=total_steps,
                message=f"Sauvegarde image {idx + 1}/{payload.image_count}",
            )

    history_entry = _build_history_entry(images_meta, payload, width, height, seed)
    if history_entry:
        _append_history_entry(history_entry)

    return {
        "images": images_meta,
        "duration_seconds": duration,
        "history_entry_id": history_entry.get("id") if history_entry else None,
    }


def run_video_generation_job(payload: GenerateVideoRequest, job_id: str | None = None) -> Dict[str, Any]:
    if job_id:
        job_manager.raise_if_interrupted(job_id)

    pipe = get_video_pipeline()

    def _parse_resolution(value: ResolutionKey | str) -> tuple[int, int]:
        if value in RESOLUTIONS:
            return RESOLUTIONS[value]  # type: ignore[index]
        try:
            parts = str(value).lower().split("x")
            width = (int(parts[0]) // 64) * 64
            height = (int(parts[1]) // 64) * 64
            width = max(64, width)
            height = max(64, height)
            return width, height
        except (ValueError, IndexError):
            return RESOLUTIONS["512x512"]

    width, height = _parse_resolution(payload.resolution)
    max_video_size = 384

    seed = payload.seed if payload.seed >= 0 else random.randint(0, 2**32 - 1)
    generator = torch.Generator(device=DEVICE).manual_seed(seed)

    init_base64 = payload.init_image_base64
    prep_steps = 0
    settings_data: Dict[str, Any] | None = None

    if payload.mode == "text2vid" and not init_base64:
        settings_data = payload.image_settings.dict() if payload.image_settings else {}
        settings_data.update(
            {
                "prompt": payload.prompt,
                "negative_prompt": payload.negative_prompt or "",
                "resolution": settings_data.get("resolution") or payload.resolution,
                "seed": payload.seed,
                "image_count": 1,
            }
        )
        settings_data.setdefault("sampler", "euler")
        settings_data.setdefault("steps", 30)
        settings_data.setdefault("cfg_scale", 7.0)
        settings_data.setdefault("clip_skip", 2)
        settings_data.setdefault("model", "sdxl")
        settings_data.setdefault("additional_loras", [])
        prep_steps = settings_data["steps"]

        if settings_data.get("width") is None and settings_data.get("height") is None:
            parsed_w, parsed_h = _parse_resolution(settings_data["resolution"])
            settings_data["width"] = parsed_w
            settings_data["height"] = parsed_h

    total_steps = payload.num_frames + prep_steps
    offset = prep_steps

    if settings_data is not None:
        if job_id:
            job_manager.update_progress(
                job_id,
                current=0,
                total=total_steps,
                message="Génération de l'image de référence…",
            )
        image_request = GenerateRequest(**settings_data)
        image_result = run_image_generation_job(image_request, job_id=None)
        if not image_result["images"]:
            raise RuntimeError("Impossible de générer l'image de référence pour la vidéo.")
        init_base64 = image_result["images"][0]["base64"]

        if job_id:
            job_manager.update_progress(
                job_id,
                current=prep_steps,
                total=total_steps,
                message="Image de référence prête, conversion en vidéo…",
            )

    if not init_base64:
        raise RuntimeError("init_image_base64 est requis pour la vidéo.")

    try:
        img_bytes = base64.b64decode(init_base64)
        init_image = Image.open(BytesIO(img_bytes)).convert("RGB")
        target_w, target_h = init_image.size
        max_dim = max(target_w, target_h)
        if max_dim > max_video_size:
            scale = max_video_size / max_dim
            target_w = int(target_w * scale)
            target_h = int(target_h * scale)
        target_w = max(64, (target_w // 64) * 64)
        target_h = max(64, (target_h // 64) * 64)
        width, height = target_w, target_h
        init_image = init_image.resize((width, height))
    except Exception as err:  # pylint: disable=broad-except
        raise RuntimeError(f"Image invalide: {err}") from err

    if job_id:
        job_manager.raise_if_interrupted(job_id)

    start = time.perf_counter()
    try:
        with torch.inference_mode():
            result = pipe(
                image=init_image,
                num_frames=payload.num_frames,
                generator=generator,
                decode_chunk_size=1,
            )
    except RuntimeError as err:
        if "out of memory" in str(err).lower() or "cuda" in str(err).lower():
            raise RuntimeError(
                "VRAM insuffisante (OOM). Réduisez num_frames (8 max) ou la résolution (384x384 max)."
            ) from err
        raise RuntimeError(f"Erreur génération vidéo: {err}") from err
    duration = time.perf_counter() - start

    frames = result.frames[0]

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_file:
        tmp_path = tmp_file.name

    try:
        writer = imageio.get_writer(
            tmp_path, format="ffmpeg", fps=payload.fps, codec="libx264", pixelformat="yuv420p"
        )
        for idx, frame in enumerate(frames):
            if hasattr(frame, "convert"):
                frame_array = np.array(frame.convert("RGB"))
            else:
                frame_array = np.array(frame)
            writer.append_data(frame_array)
            if job_id:
                job_manager.update_progress(
                    job_id,
                    current=offset + idx + 1,
                    total=total_steps,
                    message=f"Frame {idx + 1}/{len(frames)}",
                )
        writer.close()

        with open(tmp_path, "rb") as f:
            mp4_bytes = f.read()
        mp4_b64 = base64.b64encode(mp4_bytes).decode("utf-8")

        video_id = f"video-{seed}-{int(time.time())}"
        _save_video_file(video_id, mp4_b64)

        video_meta = {
            "id": video_id,
            "seed": seed,
            "num_frames": payload.num_frames,
            "fps": payload.fps,
            "prompt": payload.prompt,
            "negative_prompt": payload.negative_prompt or "",
            "timestamp": datetime.now().isoformat(),
            "file_path": f"videos/{video_id}.mp4",
            "duration_seconds": duration,
            "mode": payload.mode,
        }

        stored_videos = _load_storage_json(STORAGE_VIDEOS_JSON)
        stored_videos.append(video_meta)
        stored_videos = stored_videos[-500:]
        _save_storage_json(STORAGE_VIDEOS_JSON, stored_videos)

    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

    return {
        "video": {
            "id": video_id,
            "mp4_base64": mp4_b64,
            "file_path": f"videos/{video_id}.mp4",
        },
        "duration_seconds": duration,
    }


def _image_job_executor(job_data: Dict[str, Any], manager: JobManager) -> None:
    payload = GenerateRequest(**job_data["payload"])
    result = run_image_generation_job(payload, job_id=job_data["id"])
    manager.set_result(job_data["id"], result)
    manager.update_progress(
        job_data["id"],
        current=max(payload.steps, 1),
        total=max(payload.steps, 1),
        message="Images générées",
    )


def _video_job_executor(job_data: Dict[str, Any], manager: JobManager) -> None:
    payload = GenerateVideoRequest(**job_data["payload"])
    result = run_video_generation_job(payload, job_id=job_data["id"])
    manager.set_result(job_data["id"], result)
    prep_steps = 0
    if payload.mode == "text2vid":
        prep_steps = payload.image_settings.steps if payload.image_settings else 30
    total_steps = payload.num_frames + prep_steps
    manager.update_progress(
        job_data["id"],
        current=total_steps,
        total=total_steps,
        message="Vidéo générée",
    )


job_manager.register_executor("image", _image_job_executor)
job_manager.register_executor("video", _video_job_executor)

def apply_sampler(
    pipe: StableDiffusionPipeline | StableDiffusionXLPipeline, sampler_name: SamplerName
):
    scheduler_cls = SAMPLERS[sampler_name]
    pipe.scheduler = scheduler_cls.from_config(pipe.scheduler.config)


def pil_to_base64(image) -> str:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def _ensure_adapter_loaded(pipe, adapter_name: str, lora_path: Path):
    """Charge un LoRA s'il n'est pas déjà chargé."""
    loaded = getattr(pipe, "_loaded_loras", set())
    if adapter_name not in loaded:
        try:
            with torch.enable_grad():
                pipe.load_lora_weights(lora_path, adapter_name=adapter_name)
            loaded.add(adapter_name)
            pipe._loaded_loras = loaded  # type: ignore[attr-defined]
        except ValueError as err:
            error_msg = str(err)
            if "Invalid LoRA checkpoint" in error_msg or "not a valid LoRA" in error_msg:
                raise ValueError(
                    f"Le fichier '{lora_path.name}' n'est pas un LoRA valide. "
                    f"Il peut s'agir d'un modèle complet ou d'un format incompatible. "
                    f"Erreur: {error_msg}"
                ) from err
            raise
        except Exception as err:
            raise RuntimeError(
                f"Erreur lors du chargement du LoRA '{lora_path.name}': {type(err).__name__}: {err}"
            ) from err


def _reset_pipeline_adapters(pipe):
    default_adapter = getattr(pipe, "_default_lora_adapter", None)
    default_weight = getattr(pipe, "_default_lora_weight", 1.0)
    if default_adapter:
        pipe.set_adapters([default_adapter], adapter_weights=[default_weight])
    else:
        if hasattr(pipe, "disable_lora"):
            pipe.disable_lora()


def _load_storage_json(json_path: Path) -> list[Dict[str, Any]]:
    """Charge les métadonnées depuis le JSON de stockage."""
    if json_path.exists():
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return []
    return []


def _save_storage_json(json_path: Path, data: list[Dict[str, Any]]) -> None:
    """Sauvegarde les métadonnées dans le JSON de stockage."""
    try:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except IOError as err:
        print(f"[WARN] Impossible de sauvegarder {json_path}: {err}")


def _append_history_entry(entry: Dict[str, Any]) -> None:
    """Ajoute une entrée d'historique persistée sur disque."""
    entries = _load_storage_json(STORAGE_HISTORY_JSON)
    entries.append(entry)
    entries = entries[-MAX_HISTORY_ENTRIES:]
    _save_storage_json(STORAGE_HISTORY_JSON, entries)


def _save_image_file(image_id: str, base64_data: str) -> Path:
    """Sauvegarde une image en fichier PNG et retourne le chemin."""
    img_bytes = base64.b64decode(base64_data)
    img_path = STORAGE_IMAGES_DIR / f"{image_id}.png"
    with open(img_path, "wb") as f:
        f.write(img_bytes)
    return img_path


def _save_video_file(video_id: str, base64_data: str) -> Path:
    """Sauvegarde une vidéo en fichier MP4 et retourne le chemin."""
    video_bytes = base64.b64decode(base64_data)
    video_path = STORAGE_VIDEOS_DIR / f"{video_id}.mp4"
    with open(video_path, "wb") as f:
        f.write(video_bytes)
    return video_path


def _apply_request_loras(
    pipe, payload: GenerateRequest
) -> list[AppliedLoRA]:
    applied: list[AppliedLoRA] = []
    adapters: list[str] = []
    weights: list[float] = []

    default_adapter = getattr(pipe, "_default_lora_adapter", None)
    if default_adapter:
        adapters.append(default_adapter)
        default_weight = getattr(pipe, "_default_lora_weight", 1.0)
        weights.append(default_weight)
        default_label = getattr(pipe, "_default_lora_label", default_adapter)
        applied.append(
            AppliedLoRA(
                key=default_adapter,
                label=default_label,
                weight=default_weight,
                type="LoRA",
            )
        )

    for lora_input in payload.additional_loras:
        config = LORA_LIBRARY.get(lora_input.key)
        if not config:
            print(f"[WARN] LoRA '{lora_input.key}' non trouvé dans la bibliothèque")
            continue
        try:
            lora_path = ensure_lora_file(lora_input.key)
        except FileNotFoundError as err:
            print(f"[WARN] ⚠️  Fichier LoRA introuvable pour '{lora_input.key}': {err}")
            continue
        adapter_name = f"user-{lora_input.key}"
        try:
            _ensure_adapter_loaded(pipe, adapter_name, lora_path)
            adapters.append(adapter_name)
            weights.append(lora_input.weight)
            applied.append(
                AppliedLoRA(
                    key=lora_input.key,
                    label=config["label"],
                    weight=lora_input.weight,
                    type=config.get("type", "LoRA"),
                )
            )
        except (ValueError, RuntimeError) as err:
            error_msg = str(err)
            if "Invalid LoRA checkpoint" in error_msg or "not a valid LoRA" in error_msg:
                print(f"[WARN] ⚠️  Le fichier LoRA '{lora_input.key}' n'est pas un LoRA valide.")
                print(f"       Il peut s'agir d'un modèle complet ou d'un format incompatible.")
                print(f"       Ce LoRA sera ignoré pour cette génération.")
            else:
                print(f"[WARN] ⚠️  Impossible de charger le LoRA '{lora_input.key}': {error_msg}")
                print(f"       Ce LoRA sera ignoré pour cette génération.")
            continue

    if adapters:
        pipe.set_adapters(adapters, adapter_weights=weights)
    else:
        if hasattr(pipe, "disable_lora"):
            pipe.disable_lora()

    return applied


app = FastAPI(title="Local Stable Diffusion Bridge", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _startup_events():
    job_manager.start()


@app.on_event("shutdown")
def _shutdown_events():
    job_manager.stop()


@app.get("/health")
def health():
    """Endpoint de santé avec informations sur le device."""
    try:
        # Informations détaillées sur le device
        device_info = {"device": DEVICE, "status": "ok"}
        if DEVICE == "cuda" and torch.cuda.is_available():
            try:
                device_info["gpu_name"] = torch.cuda.get_device_name(0)
                props = torch.cuda.get_device_properties(0)
                device_info["vram_total_gb"] = round(props.total_memory / (1024**3), 2)
                device_info["vram_allocated_gb"] = round(torch.cuda.memory_allocated(0) / (1024**3), 2)
                device_info["vram_reserved_gb"] = round(torch.cuda.memory_reserved(0) / (1024**3), 2)
                device_info["vram_free_gb"] = round(device_info["vram_total_gb"] - device_info["vram_reserved_gb"], 2)
                device_info["cached_models"] = list(PIPELINE_CACHE.keys())
                device_info["cache_size"] = len(PIPELINE_CACHE)
            except Exception as e:
                device_info["gpu_error"] = str(e)
        return device_info
    except Exception as e:
        return {"status": "error", "device": DEVICE, "error": str(e)}


@app.post("/clear-cache")
def clear_cache():
    """Force le nettoyage du cache des modèles et de la mémoire GPU."""
    try:
        cleared = []
        for model_key in list(PIPELINE_CACHE.keys()):
            _unload_pipeline(model_key)
            cleared.append(model_key)
        _clear_gpu_memory()
        return {
            "status": "ok",
            "cleared_models": cleared,
            "message": f"{len(cleared)} modèle(s) déchargé(s) de la mémoire GPU",
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


@app.get("/models")
def list_models():
    """List all registered models with their installation status."""
    models_info = []
    for model_key in MODEL_CONFIG.keys():
        config = MODEL_CONFIG[model_key]
        model_path = Path(config["path"]).expanduser()
        is_installed = model_path.exists()
        is_enabled = model_key in ENABLED_MODELS

        models_info.append(
            {
                "key": model_key,
                "variant": config["variant"],
                "installed": is_installed,
                "enabled": is_enabled,
                "path": str(model_path),
            }
        )
    return {"models": models_info, "enabled": list(ENABLED_MODELS)}


@app.get("/loras")
def list_loras():
    """Expose la bibliothèque de LoRAs disponibles côté backend."""
    items = []
    for key, config in LORA_LIBRARY.items():
        items.append(
            {
                "key": key,
                "label": config["label"],
                "description": config.get("description", ""),
                "default_weight": config.get("default_weight", 0.5),
                "path": config.get("path"),
                "nsfw": config.get("nsfw", False),
            }
        )
    return {"loras": items}


@app.post("/chat", response_model=ChatResponseModel)
def chat_with_ollama(payload: ChatRequest):
    if not payload.messages:
        raise HTTPException(status_code=400, detail="Fournissez au moins un message.")

    request_payload: Dict[str, Any] = {
        "model": payload.model or OLLAMA_MODEL,
        "messages": [{"role": msg.role, "content": msg.content} for msg in payload.messages],
        "stream": False,
        "options": {
            "temperature": payload.temperature,
            "top_p": payload.top_p,
        },
    }
    if payload.max_tokens:
        request_payload["options"]["num_predict"] = payload.max_tokens

    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json=request_payload,
            timeout=120,
        )
    except requests.RequestException as err:
        raise HTTPException(status_code=502, detail=f"Ollama injoignable: {err}") from err

    if response.status_code != 200:
        raise HTTPException(
            status_code=response.status_code,
            detail=f"Erreur Ollama: {response.text}",
        )

    resp_json = response.json()
    message_data = resp_json.get("message")
    if not message_data:
        raise HTTPException(status_code=500, detail="Réponse Ollama invalide.")

    chat_message = ChatMessage(
        role=message_data.get("role", "assistant"),
        content=message_data.get("content", ""),
    )
    return ChatResponseModel(message=chat_message, raw=resp_json)


@app.post("/generate")
def generate(payload: GenerateRequest):
    try:
        _ensure_model_enabled(payload.model)
    except RuntimeError as err:
        raise HTTPException(status_code=400, detail=str(err)) from err

    job = job_manager.create_job(
        "image",
        payload.dict(),
        metadata={"prompt_preview": payload.prompt[:120]},
        total_steps=max(payload.steps, 1),
    )
    return {"job_id": job["id"], "status": job["status"], "type": job["type"]}


@app.post("/generate-video")
def generate_video(payload: GenerateVideoRequest):
    if not payload.init_image_base64:
        if payload.mode == "img2vid":
            raise HTTPException(
                status_code=400,
                detail="init_image_base64 est requis pour le mode image→vidéo.",
            )

    prep_steps = 0
    if payload.mode == "text2vid":
        prep_steps = payload.image_settings.steps if payload.image_settings else 30

    job = job_manager.create_job(
        "video",
        payload.dict(),
        metadata={"prompt_preview": payload.prompt[:120]},
        total_steps=payload.num_frames + prep_steps,
    )
    return {"job_id": job["id"], "status": job["status"], "type": job["type"]}


@app.get("/jobs")
def list_jobs():
    return {"jobs": job_manager.list_jobs()}


@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job introuvable.")
    return job


@app.post("/jobs/{job_id}/pause")
def pause_job(job_id: str):
    if not job_manager.request_pause(job_id):
        raise HTTPException(status_code=400, detail="Impossible de mettre le job en pause.")
    return {"status": "paused", "job_id": job_id}


@app.post("/jobs/{job_id}/resume")
def resume_job(job_id: str):
    if not job_manager.resume_job(job_id, prioritize=False):
        raise HTTPException(status_code=400, detail="Impossible de reprendre ce job.")
    return {"status": "pending", "job_id": job_id}


@app.post("/jobs/{job_id}/start")
def start_job(job_id: str):
    if not job_manager.resume_job(job_id, prioritize=True):
        raise HTTPException(status_code=400, detail="Impossible de démarrer ce job.")
    return {"status": "pending", "job_id": job_id, "priority": "high"}


@app.post("/jobs/{job_id}/cancel")
def cancel_job(job_id: str):
    if not job_manager.cancel_job(job_id):
        raise HTTPException(status_code=400, detail="Impossible d'annuler ce job.")
    return {"status": "cancelled", "job_id": job_id}


@app.delete("/jobs/{job_id}")
def delete_job(job_id: str):
    if not job_manager.delete_job(job_id):
        raise HTTPException(status_code=400, detail="Impossible de supprimer ce job.")
    return {"status": "deleted", "job_id": job_id}


@app.post("/jobs/resume-all")
def resume_all_jobs():
    resumed = 0
    for job in job_manager.list_jobs():
        if job["status"] in ("paused", "failed"):
            if job_manager.resume_job(job["id"], prioritize=False):
                resumed += 1
    return {"resumed": resumed}


@app.post("/jobs/clear-completed")
def clear_completed_jobs():
    removed = job_manager.clear_completed()
    return {"removed": removed}


@app.get("/storage/images")
def get_stored_images():
    """Récupère toutes les images sauvegardées avec leurs métadonnées."""
    stored = _load_storage_json(STORAGE_IMAGES_JSON)
    # Charger les images en base64 depuis les fichiers
    for item in stored:
        file_path = STORAGE_IMAGES_DIR / f"{item['id']}.png"
        if file_path.exists():
            with open(file_path, "rb") as f:
                img_bytes = f.read()
                item["base64"] = base64.b64encode(img_bytes).decode("utf-8")
        else:
            item["base64"] = None
    return {"images": stored}


@app.get("/storage/videos")
def get_stored_videos():
    """Récupère toutes les vidéos sauvegardées avec leurs métadonnées."""
    stored = _load_storage_json(STORAGE_VIDEOS_JSON)
    # Charger les vidéos en base64 depuis les fichiers
    for item in stored:
        file_path = STORAGE_VIDEOS_DIR / f"{item['id']}.mp4"
        if file_path.exists():
            with open(file_path, "rb") as f:
                video_bytes = f.read()
                item["mp4_base64"] = base64.b64encode(video_bytes).decode("utf-8")
        else:
            item["mp4_base64"] = None
    return {"videos": stored}


@app.get("/storage/history")
def get_stored_history():
    """Récupère les entrées d'historique persistées."""
    stored = _load_storage_json(STORAGE_HISTORY_JSON)
    for item in stored:
        thumb_id = item.get("thumbnail_id")
        if thumb_id:
            file_path = STORAGE_IMAGES_DIR / f"{thumb_id}.png"
            if file_path.exists():
                with open(file_path, "rb") as f:
                    item["thumbnail_base64"] = base64.b64encode(f.read()).decode("utf-8")
            else:
                item["thumbnail_base64"] = None
        else:
            item["thumbnail_base64"] = None
    return {"history": stored}


@app.delete("/storage/history/{entry_id}")
def delete_history_entry(entry_id: str):
    """Supprime une entrée d'historique."""
    stored = _load_storage_json(STORAGE_HISTORY_JSON)
    updated = [entry for entry in stored if entry.get("id") != entry_id]
    _save_storage_json(STORAGE_HISTORY_JSON, updated)
    return {"success": True}


@app.delete("/storage/image/{image_id}")
def delete_stored_image(image_id: str):
    """Supprime une image et ses métadonnées."""
    # Supprimer le fichier
    file_path = STORAGE_IMAGES_DIR / f"{image_id}.png"
    if file_path.exists():
        file_path.unlink()
    
    # Supprimer des métadonnées
    stored = _load_storage_json(STORAGE_IMAGES_JSON)
    stored = [img for img in stored if img.get("id") != image_id]
    _save_storage_json(STORAGE_IMAGES_JSON, stored)
    
    return {"status": "deleted", "id": image_id}


@app.delete("/storage/video/{video_id}")
def delete_stored_video(video_id: str):
    """Supprime une vidéo et ses métadonnées."""
    # Supprimer le fichier
    file_path = STORAGE_VIDEOS_DIR / f"{video_id}.mp4"
    if file_path.exists():
        file_path.unlink()
    
    # Supprimer des métadonnées
    stored = _load_storage_json(STORAGE_VIDEOS_JSON)
    stored = [vid for vid in stored if vid.get("id") != video_id]
    _save_storage_json(STORAGE_VIDEOS_JSON, stored)
    
    return {"status": "deleted", "id": video_id}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

