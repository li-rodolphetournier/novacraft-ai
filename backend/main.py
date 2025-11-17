from __future__ import annotations

import base64
import os
import random
import tempfile
import time
from io import BytesIO
from pathlib import Path
from typing import Dict, Literal, Tuple

import imageio
import numpy as np
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
        "sdxl,cyberrealistic-pony,tsunade-il,wai-illustrious-sdxl,wan22-enhanced-nsfw-camera,hassaku-xl-illustrious-v32,duchaiten-pony-xl",
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
            }
        )
    return {"loras": items}


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

    images = [
        GeneratedImage(
            id=f"{payload.model}-{idx}",
            seed=seed + idx,
            base64=pil_to_base64(image),
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
        for idx, image in enumerate(result.images)
    ]

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
    finally:
        # Nettoyer le fichier temporaire
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

    return GenerateVideoResponse(
        video=GeneratedVideo(mp4_base64=mp4_b64),
        duration_seconds=duration,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

