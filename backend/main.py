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
MAX_HISTORY_ENTRIES = 500

# Créer les dossiers s'ils n'existent pas
STORAGE_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
STORAGE_VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
STORAGE_HISTORY_JSON.parent.mkdir(parents=True, exist_ok=True)

for json_path in (STORAGE_IMAGES_JSON, STORAGE_VIDEOS_JSON, STORAGE_HISTORY_JSON):
    if not json_path.exists():
        json_path.write_text("[]", encoding="utf-8")


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
    resolution: ResolutionKey | None = None
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
    weight: float = Field(default=0.5, ge=0.0, le=2.0)


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
    resolution: ResolutionKey = "512x512"
    seed: int = -1
    init_image_base64: str | None = None


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


PIPELINE_CACHE: Dict[ModelKey, StableDiffusionPipeline | StableDiffusionXLPipeline] = {}
VIDEO_PIPELINE: StableVideoDiffusionPipeline | None = None

ENABLED_MODELS = {
    key.strip()
    for key in os.getenv(
        "ENABLED_MODELS",
        "sdxl,cyberrealistic-pony,tsunade-il,wai-illustrious-sdxl,wan22-enhanced-nsfw-camera,hassaku-xl-illustrious-v32,duchaiten-pony-xl,lucentxl-pony,ponydiffusion-v6-xl,ishtars-gate-nsfw-sfw",
    ).split(",")
    if key.strip()
}


def load_pipeline(model_key: ModelKey):
    if model_key in PIPELINE_CACHE:
        return PIPELINE_CACHE[model_key]

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
                with torch.enable_grad():
                    pipe.load_lora_weights(lora_path, adapter_name=adapter_name)
                pipe.set_adapters([adapter_name])
                pipe._loaded_loras.add(adapter_name)  # type: ignore[attr-defined]
                pipe._default_lora_adapter = adapter_name  # type: ignore[attr-defined]
                pipe._default_lora_label = config.get("lora_label") or adapter_name  # type: ignore[attr-defined]
                pipe._default_lora_weight = float(config.get("lora_weight", 1.0))  # type: ignore[attr-defined]
            except Exception as err:  # pylint: disable=broad-except
                print(f"[WARN] Impossible de charger le LoRA '{lora_path}': {err}")
        else:
            print(f"[WARN] LoRA '{lora_path}' introuvable pour {model_key}.")
    else:
        if hasattr(pipe, "disable_lora"):
            pipe.disable_lora()

    if DEVICE == "cuda":
        pipe.enable_model_cpu_offload()
    else:
        pipe = pipe.to("cpu")

    PIPELINE_CACHE[model_key] = pipe
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
    loaded = getattr(pipe, "_loaded_loras", set())
    if adapter_name not in loaded:
        with torch.enable_grad():
            pipe.load_lora_weights(lora_path, adapter_name=adapter_name)
        loaded.add(adapter_name)
        pipe._loaded_loras = loaded  # type: ignore[attr-defined]


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
            continue
        try:
            lora_path = ensure_lora_file(lora_input.key)
        except FileNotFoundError as err:
            raise HTTPException(status_code=400, detail=str(err)) from err
        adapter_name = f"user-{lora_input.key}"
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


@app.get("/health")
def health():
    return {"status": "ok", "device": DEVICE}


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


@app.post("/generate", response_model=GenerateResponse)
def generate(payload: GenerateRequest):
    if payload.model not in ENABLED_MODELS:
        allowed = ", ".join(sorted(ENABLED_MODELS)) or "sdxl"
        raise HTTPException(
            status_code=400,
            detail=(
                f"Le modèle '{payload.model}' est désactivé. "
                f"Modèles disponibles actuellement: {allowed}. "
                "Définissez la variable d'environnement ENABLED_MODELS pour en activer d'autres."
            ),
        )

    try:
        pipe = load_pipeline(payload.model)
    except FileNotFoundError as err:
        raise HTTPException(status_code=400, detail=str(err)) from err
    except Exception as err:  # pylint: disable=broad-except
        raise HTTPException(status_code=500, detail=str(err)) from err

    apply_sampler(pipe, payload.sampler)
    
    # Utilise width/height personnalisés si fournis, sinon utilise la résolution standard
    if payload.width and payload.height:
        width = payload.width
        height = payload.height
        # Arrondit aux multiples de 64 (requis par SDXL)
        width = (width // 64) * 64
        height = (height // 64) * 64
    else:
        # Parse la résolution (peut être "1024x1024" ou "customWxH")
        if payload.resolution in RESOLUTIONS:
            width, height = RESOLUTIONS[payload.resolution]
        else:
            # Parse format "WxH"
            try:
                parts = payload.resolution.split("x")
                width = int(parts[0])
                height = int(parts[1])
                width = (width // 64) * 64
                height = (height // 64) * 64
            except (ValueError, IndexError):
                width, height = RESOLUTIONS["1024x1024"]  # Fallback

    seed = payload.seed if payload.seed >= 0 else random.randint(0, 2**32 - 1)
    generator = torch.Generator(device=DEVICE).manual_seed(seed)

    applied_loras_meta: list[AppliedLoRA] = []
    start = time.perf_counter()
    try:
        applied_loras_meta = _apply_request_loras(pipe, payload)
        with torch.inference_mode():
            result = pipe(
                prompt=payload.prompt,
                negative_prompt=payload.negative_prompt or "",
                guidance_scale=payload.cfg_scale,
                num_inference_steps=payload.steps,
                width=width,
                height=height,
                generator=generator,
                num_images_per_prompt=payload.image_count,
            )
    finally:
        _reset_pipeline_adapters(pipe)
    duration = time.perf_counter() - start

    images = []
    for idx, image in enumerate(result.images):
        image_id = f"{payload.model}-{seed + idx}-{int(time.time())}"
        base64_data = pil_to_base64(image)
        
        # Sauvegarder l'image en fichier
        _save_image_file(image_id, base64_data)
        
        # Sauvegarder les métadonnées
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
        }
        
        # Ajouter aux métadonnées stockées
        stored_images = _load_storage_json(STORAGE_IMAGES_JSON)
        stored_images.append(image_meta)
        # Garder seulement les 1000 dernières images
        stored_images = stored_images[-1000:]
        _save_storage_json(STORAGE_IMAGES_JSON, stored_images)
        
        images.append(
            GeneratedImage(
                id=image_id,
                seed=seed + idx,
                base64=base64_data,
                model=payload.model,
                sampler=payload.sampler,
                steps=payload.steps,
                cfg_scale=payload.cfg_scale,
                resolution=f"{width}x{height}",
                prompt=payload.prompt,
                negative_prompt=payload.negative_prompt or "",
                clip_skip=payload.clip_skip,
                loras=applied_loras_meta,
            )
        )

    if images:
        history_entry = {
            "id": str(uuid.uuid4()),
            "prompt": payload.prompt,
            "negative_prompt": payload.negative_prompt or "",
            "model": payload.model,
            "timestamp": datetime.now().isoformat(),
            "thumbnail_id": images[0].id,
            "settings": {
                "sampler": payload.sampler,
                "steps": payload.steps,
                "cfg_scale": payload.cfg_scale,
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
        _append_history_entry(history_entry)

    return GenerateResponse(
        images=images,
        model=payload.model,
        sampler=payload.sampler,
        steps=payload.steps,
        cfg_scale=payload.cfg_scale,
        resolution=f"{width}x{height}",
        duration_seconds=duration,
    )


@app.post("/generate-video", response_model=GenerateVideoResponse)
def generate_video(payload: GenerateVideoRequest):
    """Génère une petite vidéo à partir d'une image de départ + texte."""
    pipe = get_video_pipeline()
    
    # Résolution fallback (au cas où l'image initiale manque, mais init est requis)
    width, height = RESOLUTIONS[payload.resolution]
    max_video_size = 384

    # Seed
    seed = payload.seed if payload.seed >= 0 else random.randint(0, 2**32 - 1)
    generator = torch.Generator(device=DEVICE).manual_seed(seed)

    # Image de départ obligatoire pour le moment
    if not payload.init_image_base64:
        raise HTTPException(
            status_code=400,
            detail="init_image_base64 est requis pour la vidéo pour l'instant.",
        )

    try:
        img_bytes = base64.b64decode(payload.init_image_base64)
        init_image = Image.open(BytesIO(img_bytes)).convert("RGB")
        target_w, target_h = init_image.size
        max_dim = max(target_w, target_h)
        if max_dim > max_video_size:
            scale = max_video_size / max_dim
            target_w = int(target_w * scale)
            target_h = int(target_h * scale)
        # Arrondit aux multiples de 64 pour SVD, minimum 64
        target_w = max(64, (target_w // 64) * 64)
        target_h = max(64, (target_h // 64) * 64)
        width, height = target_w, target_h
        init_image = init_image.resize((width, height))
    except Exception as err:  # pylint: disable=broad-except
        raise HTTPException(status_code=400, detail=f"Image invalide: {err}") from err

    start = time.perf_counter()
    try:
        with torch.inference_mode():
            result = pipe(
                image=init_image,
                num_frames=payload.num_frames,
                generator=generator,
                decode_chunk_size=1,  # Décode frame par frame pour économiser VRAM
            )
    except RuntimeError as err:
        if "out of memory" in str(err).lower() or "CUDA" in str(err):
            raise HTTPException(
                status_code=507,
                detail=(
                    "VRAM insuffisante (OOM). Réduisez num_frames (8 max), "
                    "résolution (384x384 max), ou fermez d'autres applications GPU."
                ),
            ) from err
        raise HTTPException(status_code=500, detail=f"Erreur génération vidéo: {err}") from err
    duration = time.perf_counter() - start

    frames = result.frames[0]

    # Encodage en mp4 via fichier temporaire (FFMPEG ne supporte pas BytesIO directement)
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_file:
        tmp_path = tmp_file.name

    try:
        writer = imageio.get_writer(tmp_path, format="ffmpeg", fps=payload.fps, codec="libx264", pixelformat="yuv420p")
        for frame in frames:
            # Convertir PIL Image en numpy array (RGB uint8)
            if hasattr(frame, "convert"):  # C'est une PIL Image
                frame_array = np.array(frame.convert("RGB"))
            else:
                frame_array = np.array(frame)
            writer.append_data(frame_array)
        writer.close()

        # Lire le fichier temporaire et l'encoder en base64
        with open(tmp_path, "rb") as f:
            mp4_bytes = f.read()
        mp4_b64 = base64.b64encode(mp4_bytes).decode("utf-8")
        
        # Sauvegarder la vidéo
        video_id = f"video-{seed}-{int(time.time())}"
        _save_video_file(video_id, mp4_b64)
        
        # Sauvegarder les métadonnées
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
        }
        
        stored_videos = _load_storage_json(STORAGE_VIDEOS_JSON)
        stored_videos.append(video_meta)
        stored_videos = stored_videos[-500:]  # Garder les 500 dernières vidéos
        _save_storage_json(STORAGE_VIDEOS_JSON, stored_videos)
    finally:
        # Nettoyer le fichier temporaire
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

    return GenerateVideoResponse(
        video=GeneratedVideo(mp4_base64=mp4_b64, id=video_id),
        duration_seconds=duration,
    )


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

