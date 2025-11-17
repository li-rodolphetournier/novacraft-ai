"use client";

import { useEffect, useMemo, useState } from "react";

type Sampler = "euler" | "dpmpp_2s" | "unipc" | "ddim";
type Resolution = "512x512" | "768x768" | "1024x1024" | "1536x1536";
type ModelKey =
  | "realistic-vision"
  | "dreamshaper"
  | "meinamik"
  | "sdxl"
  | "cyberrealistic-pony"
  | "tsunade-il"
  | "wai-illustrious-sdxl"
  | "wan22-enhanced-nsfw-camera"
  | "hassaku-xl-illustrious-v32"
  | "duchaiten-pony-xl";

type SelectedLora = {
  key: string;
  weight: number;
};

type LoraOption = {
  key: string;
  label: string;
  description?: string;
  defaultWeight: number;
};

type AppliedLora = {
  key: string;
  label: string;
  weight: number;
  type?: string;
};

type HistoryEntry = {
  id: string;
  prompt: string;
  negativePrompt: string;
  model: ModelKey;
  timestamp: number;
  thumbnail?: string;
  settings?: {
    sampler: Sampler;
    steps: number;
    cfgScale: number;
    resolution: Resolution;
    seed: number;
    useAspectRatio?: boolean;
    aspectRatio?: string;
    customWidth?: number;
    customHeight?: number;
    loras?: SelectedLora[];
  };
};

type GeneratedImage = {
  id: string;
  seed: number;
  base64: string;
  model?: string;
  sampler?: string;
  steps?: number;
  cfg_scale?: number;
  resolution?: string;
  prompt?: string;
  negative_prompt?: string;
  clip_skip?: number;
  loras?: AppliedLora[];
};

type GenerateResponse = {
  images: GeneratedImage[];
  duration_seconds?: number;
};

type GeneratedVideo = {
  id: string;
  mp4Base64: string;
  durationSeconds?: number;
};

type PromptPreset = {
  name: string;
  prompt: string;
  negativePrompt: string;
  description: string;
};

const samplers: { label: string; value: Sampler }[] = [
  { label: "Euler A", value: "euler" },
  { label: "DPM++ 2S", value: "dpmpp_2s" },
  { label: "UniPC", value: "unipc" },
  { label: "DDIM", value: "ddim" },
];

const resolutions: { label: string; value: Resolution; hint: string }[] = [
  { label: "512 × 512 · SD1.5 rapide", value: "512x512", hint: "SD1.5 rapide" },
  {
    label: "768 × 768 · SDXL léger",
    value: "768x768",
    hint: "SDXL faible VRAM",
  },
  {
    label: "1024 × 1024 · SDXL conseillé",
    value: "1024x1024",
    hint: "GPU 8–12 Go",
  },
  {
    label: "1536 × 1536 · SDXL haute résolution",
    value: "1536x1536",
    hint: "GPU 12 Go+ recommandé",
  },
];

const aspectRatios: { label: string; value: string; width: number; height: number }[] = [
  { label: "1:1 (Carré)", value: "1:1", width: 1024, height: 1024 },
  { label: "16:9 (Paysage)", value: "16:9", width: 1344, height: 768 },
  { label: "9:16 (Portrait)", value: "9:16", width: 768, height: 1344 },
  { label: "4:3 (Classique)", value: "4:3", width: 1152, height: 896 },
  { label: "3:4 (Portrait)", value: "3:4", width: 896, height: 1152 },
  { label: "21:9 (Ultra large)", value: "21:9", width: 1536, height: 640 },
  { label: "2:3 (Portrait)", value: "2:3", width: 832, height: 1216 },
  { label: "3:2 (Paysage)", value: "3:2", width: 1216, height: 832 },
];

const promptPresets: PromptPreset[] = [
  {
    name: "Portrait réaliste",
    prompt: "portrait of a person, professional photography, high quality, detailed face, natural lighting",
    negativePrompt: "blurry, bad anatomy, low quality, distorted hands, watermark, cartoon, anime",
    description: "Portrait photographique de qualité",
  },
  {
    name: "Paysage fantastique",
    prompt: "epic fantasy landscape, mountains, magical atmosphere, cinematic lighting, highly detailed, 4k",
    negativePrompt: "blurry, low quality, distorted, watermark, people",
    description: "Paysage épique et cinématique",
  },
  {
    name: "Architecture moderne",
    prompt: "modern architecture, futuristic building, clean lines, minimalist design, professional photography",
    negativePrompt: "blurry, low quality, old, vintage, distorted",
    description: "Architecture contemporaine",
  },
  {
    name: "Art conceptuel",
    prompt: "concept art, digital painting, vibrant colors, detailed, fantasy, artistic style",
    negativePrompt: "blurry, low quality, watermark, realistic photo",
    description: "Style artistique et coloré",
  },{
    name: "Nfsw",
    prompt: "anime style, manga style, A woman is at the beach, she has semen on her mouth that is dripping onto the man's penis. The woman uses her breasts to perform fellatio on the man. She squeezes her breasts together with her fists",
    negativePrompt: "score_5, score_4, 3d, render, simple background, zPDXL2, pointy chin, flat chested, cross eyed, sleeves, long hair, blush, fewer digits, lesser digits, missing fingers, extra hands, extra fingers, interracial,",
    description: "Style artistique et coloré",
  },
  {
    name: "Nfsw2",
    prompt: "score_9, score_8_up, score_7_up, score_6_up, Fubuki \(One-Punch Man\), black leotard, jewelry, natural breasts, anime, hips, cinematic angle, cinematic lighting, volumetric lighting, solo focus, erect nipples, medium breasts, saggy breasts, teardrop breasts, face focus, sweat, sleeveless, mature woman, topless, sexy, seductive, parted lips, determined, looking at viewer, (leaning forward), arched back, 1boy, looking at viewer, (pov:1.5), crotch, (paizuri:1.2),penis between breasts, upper body (close up:1.2), low angle, cum on breasts, cum on face, ejaculation, (mouth open, tongue out), tongue, tongue out, rolling eyes, sunglasses on head, beach, sky, palms, <lora:incase-ilff-v3-4:0.5> <lora:Expressive_H:0.45>",
    negativePrompt: "score_5, score_4, 3d, render, simple background, zPDXL2, pointy chin, flat chested, cross eyed, sleeves, long hair, blush, fewer digits, lesser digits, missing fingers, extra hands, extra fingers, interracial,",
    description: "Style artistique et coloré",
  },
];

const apiBase = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

const HISTORY_STORAGE_KEY = "sd_generator_history";
const MAX_HISTORY_ENTRIES = 50;
const GALLERY_STORAGE_KEY = "sd_generator_gallery";
const MAX_GALLERY_IMAGES = 100;

export default function Home() {
  const [prompt, setPrompt] = useState("");
  const [negativePrompt, setNegativePrompt] = useState(
    "blurry, bad anatomy, low quality, distorted hands, watermark",
  );
  const [sampler, setSampler] = useState<Sampler>("euler");
  const [resolution, setResolution] = useState<Resolution>("1024x1024");
  const [useAspectRatio, setUseAspectRatio] = useState(false);
  const [aspectRatio, setAspectRatio] = useState("1:1");
  const [customWidth, setCustomWidth] = useState(1024);
  const [customHeight, setCustomHeight] = useState(1024);
  const [steps, setSteps] = useState(30);
  const [cfgScale, setCfgScale] = useState(7);
  const [seed, setSeed] = useState("3884499817");
  const [clipSkip, setClipSkip] = useState(2);
  const [imageCount, setImageCount] = useState(1);
  const [model, setModel] = useState<ModelKey>("sdxl");
  const [images, setImages] = useState<GeneratedImage[]>([]);
  const [history, setHistory] = useState<HistoryEntry[]>([]);
  const [isGenerating, setIsGenerating] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [generationProgress, setGenerationProgress] = useState<number | null>(null);
  const [lastDuration, setLastDuration] = useState<number | null>(null);
  const [availableModels, setAvailableModels] = useState<Set<string>>(new Set(["sdxl"]));
  const [mode, setMode] = useState<"image" | "video">("image");
  const [videos, setVideos] = useState<GeneratedVideo[]>([]);
  const [initImageBase64, setInitImageBase64] = useState<string | null>(null);
  const [numFrames, setNumFrames] = useState(8);
  const [fps, setFps] = useState(6);
  const [videoDuration, setVideoDuration] = useState(1.33); // 8 frames / 6 fps ≈ 1.33s
  const [availableLoras, setAvailableLoras] = useState<LoraOption[]>([]);
  const [selectedLoras, setSelectedLoras] = useState<SelectedLora[]>([]);

  // Charger l'historique depuis localStorage au montage
  useEffect(() => {
    try {
      const stored = localStorage.getItem(HISTORY_STORAGE_KEY);
      if (stored) {
        const parsed = JSON.parse(stored) as HistoryEntry[];
        setHistory(parsed.slice(0, MAX_HISTORY_ENTRIES));
      }
    } catch {
      // Ignorer les erreurs de parsing
    }
  }, []);

  // Charger les images depuis localStorage au montage
  useEffect(() => {
    try {
      const stored = localStorage.getItem(GALLERY_STORAGE_KEY);
      if (stored) {
        const parsed = JSON.parse(stored) as GeneratedImage[];
        setImages(parsed.slice(0, MAX_GALLERY_IMAGES));
      }
    } catch {
      // Ignorer les erreurs de parsing
    }
  }, []);

  // Vérifier les modèles disponibles au montage
  useEffect(() => {
    fetch(`${apiBase}/models`)
      .then((res) => res.json())
      .then((data: { enabled?: string[] }) => {
        const enabled = new Set<string>(data.enabled || []);
        setAvailableModels(enabled);
      })
      .catch(() => {
        // Ignorer les erreurs, garder les valeurs par défaut
      });
  }, []);

  useEffect(() => {
    fetch(`${apiBase}/loras`)
      .then((res) => res.json())
      .then((data: { loras?: LoraOption[] }) => {
        setAvailableLoras(data.loras || []);
      })
      .catch(() => {
        setAvailableLoras([]);
      });
  }, []);

  // Sauvegarder l'historique dans localStorage
  const saveHistory = (newHistory: HistoryEntry[]) => {
    try {
      localStorage.setItem(HISTORY_STORAGE_KEY, JSON.stringify(newHistory.slice(0, MAX_HISTORY_ENTRIES)));
    } catch {
      // Ignorer les erreurs de stockage
    }
  };

  // Sauvegarder les images dans localStorage
  const saveImages = (newImages: GeneratedImage[]) => {
    try {
      localStorage.setItem(GALLERY_STORAGE_KEY, JSON.stringify(newImages.slice(0, MAX_GALLERY_IMAGES)));
    } catch {
      // Ignorer les erreurs de stockage
    }
  };

  const cfgHint = useMemo(() => {
    if (cfgScale <= 7) return "5–7 → réaliste";
    if (cfgScale <= 11) return "7–11 → artistique";
    return "12+ = surprompting";
  }, [cfgScale]);

  const stepsHint = useMemo(() => {
    if (steps <= 20) return "Rapide / brouillon";
    if (steps <= 35) return "Qualité standard";
    return "Haute fidélité (>35)";
  }, [steps]);

  const loraOptionMap = useMemo(() => {
    return new Map(availableLoras.map((lora) => [lora.key, lora]));
  }, [availableLoras]);

  const handleToggleLora = (key: string) => {
    setSelectedLoras((prev) => {
      const exists = prev.find((item) => item.key === key);
      if (exists) {
        return prev.filter((item) => item.key !== key);
      }
      const defaultWeight =
        loraOptionMap.get(key)?.defaultWeight ?? 0.5;
      return [...prev, { key, weight: defaultWeight }];
    });
  };

  const handleLoraWeightChange = (key: string, weight: number) => {
    setSelectedLoras((prev) =>
      prev.map((item) => (item.key === key ? { ...item, weight } : item)),
    );
  };

  const handleClearLoras = () => {
    setSelectedLoras([]);
  };

  const models = useMemo(() => {
    return [
      {
        label: "SDXL Base 1.0",
        value: "sdxl" as ModelKey,
        note: "Officiel · top qualité",
        available: availableModels.has("sdxl"),
        highlight: "Choix recommandé",
      },
      {
        label: "CyberRealistic Pony",
        value: "cyberrealistic-pony" as ModelKey,
        note: "CyberRealistic / style pony",
        available: availableModels.has("cyberrealistic-pony"),
      },
      {
        label: "Tsunade iL",
        value: "tsunade-il" as ModelKey,
        note: "Style Tsunade iL",
        available: availableModels.has("tsunade-il"),
      },
      {
        label: "Wai Illustrious SDXL v1.4",
        value: "wai-illustrious-sdxl" as ModelKey,
        note: "Base SDXL illustrée",
        available: availableModels.has("wai-illustrious-sdxl"),
      },
      {
        label: "wan22 Enhanced NSFW Camera",
        value: "wan22-enhanced-nsfw-camera" as ModelKey,
        note: "wan22Enhanced NSFW Camera Prompt",
        available: availableModels.has("wan22-enhanced-nsfw-camera"),
      },
      {
        label: "Hassaku XL Illustrious v3.2",
        value: "hassaku-xl-illustrious-v32" as ModelKey,
        note: "Hassaku XL Illustrious v3.2",
        available: availableModels.has("hassaku-xl-illustrious-v32"),
      },
      {
        label: "DucHaiten Pony XL (no-score)",
        value: "duchaiten-pony-xl" as ModelKey,
        note: "Checkpoint pony-no-score v4.0",
        available: availableModels.has("duchaiten-pony-xl"),
      },
    ];
  }, [availableModels]);

  const handleGenerate = async () => {
    if (!prompt.trim()) {
      setError("Ajoutez un prompt pour lancer la génération.");
      return;
    }
    if (mode === "video" && !initImageBase64) {
      setError("Veuillez fournir une image de départ pour la vidéo.");
      return;
    }
    setIsGenerating(true);
    setError(null);
    setGenerationProgress(0);
    setLastDuration(null);

    // Progression estimée par step côté client (affichage uniquement)
    const totalSteps = steps;
    const stepIncrement = totalSteps > 0 ? 100 / totalSteps : 10;
    const intervalId = window.setInterval(() => {
      setGenerationProgress((prev) => {
        const current = prev ?? 0;
        // On évite d'aller à 100% avant la réponse réelle
        const next = Math.min(current + stepIncrement, 95);
        return next;
      });
    }, 400);

    try {
      if (mode === "image") {
        const response = await fetch(`${apiBase}/generate`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            prompt,
            negative_prompt: negativePrompt,
            sampler,
            steps,
            cfg_scale: cfgScale,
            resolution: useAspectRatio ? `${customWidth}x${customHeight}` : resolution,
            seed: Number(seed) || -1,
            clip_skip: clipSkip,
            image_count: imageCount,
            model,
            ...(useAspectRatio && { width: customWidth, height: customHeight }),
            additional_loras: selectedLoras,
          }),
        });

        const payload = await response.json();
        if (!response.ok) {
          throw new Error((payload as { detail?: string }).detail ?? "Erreur inconnue côté backend.");
        }
        const typedPayload: GenerateResponse = payload;

        setGenerationProgress(100);
        const generated: GeneratedImage[] = typedPayload.images ?? [];
        setImages((prev) => {
          const updated = [...prev, ...generated];
          saveImages(updated);
          return updated;
        });
        if (typeof typedPayload.duration_seconds === "number") {
          setLastDuration(typedPayload.duration_seconds);
        }

        const thumb = generated[0]?.base64;
        const newEntry: HistoryEntry = {
          id: crypto.randomUUID(),
          prompt,
          negativePrompt,
          model,
          thumbnail: thumb ? `data:image/png;base64,${thumb}` : undefined,
          timestamp: Date.now(),
          settings: {
            sampler,
            steps,
            cfgScale,
            resolution,
            seed: Number(seed) || -1,
            useAspectRatio,
            aspectRatio,
            customWidth,
            customHeight,
            loras: selectedLoras,
          },
        };

        const updatedHistory = [newEntry, ...history];
        setHistory(updatedHistory);
        saveHistory(updatedHistory);
      } else {
        const response = await fetch(`${apiBase}/generate-video`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            prompt,
            negative_prompt: negativePrompt,
            num_frames: numFrames,
            fps: fps,
            resolution,
            seed: Number(seed) || -1,
            init_image_base64: initImageBase64,
          }),
        });

        const payload = await response.json();
        if (!response.ok) {
          throw new Error(payload.detail ?? "Erreur inconnue côté backend (vidéo).");
        }

        setGenerationProgress(100);
        const newVideo: GeneratedVideo = {
          id: crypto.randomUUID(),
          mp4Base64: payload.video?.mp4_base64,
          durationSeconds: payload.duration_seconds,
        };
        setVideos((prev) => [newVideo, ...prev]);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Impossible de générer.");
    } finally {
      if (intervalId !== undefined) {
        window.clearInterval(intervalId);
      }
      setIsGenerating(false);
      setGenerationProgress(null);
    }
  };

  const handleLoadPreset = (preset: PromptPreset) => {
    setPrompt(preset.prompt);
    setNegativePrompt(preset.negativePrompt);
  };

  const handleLoadHistory = (entry: HistoryEntry) => {
    setPrompt(entry.prompt);
    setNegativePrompt(entry.negativePrompt);
    setModel(entry.model);
    if (entry.settings) {
      setSampler(entry.settings.sampler);
      setSteps(entry.settings.steps);
      setCfgScale(entry.settings.cfgScale);
      setResolution(entry.settings.resolution);
      setSeed(entry.settings.seed.toString());
      if (entry.settings.useAspectRatio) {
        setUseAspectRatio(true);
        setAspectRatio(entry.settings.aspectRatio ?? "1:1");
        setCustomWidth(entry.settings.customWidth ?? customWidth);
        setCustomHeight(entry.settings.customHeight ?? customHeight);
      } else {
        setUseAspectRatio(false);
      }
      setSelectedLoras(entry.settings.loras ?? []);
    } else {
      setSelectedLoras([]);
      setUseAspectRatio(false);
    }
  };

  const handleDeleteHistory = (id: string) => {
    const updated = history.filter((entry) => entry.id !== id);
    setHistory(updated);
    saveHistory(updated);
  };

  const handleExportImage = (image: GeneratedImage) => {
    const byteCharacters = atob(image.base64);
    const byteNumbers = new Array(byteCharacters.length);
    for (let i = 0; i < byteCharacters.length; i++) {
      byteNumbers[i] = byteCharacters.charCodeAt(i);
    }
    const byteArray = new Uint8Array(byteNumbers);
    const blob = new Blob([byteArray], { type: "image/png" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `sd_generated_${image.seed}_${Date.now()}.png`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const handleCopyBase64 = async (image: GeneratedImage) => {
    try {
      await navigator.clipboard.writeText(`data:image/png;base64,${image.base64}`);
      // Optionnel: afficher un toast de confirmation
    } catch {
      // Fallback si clipboard API n'est pas disponible
    }
  };

  const handleDeleteImage = (id: string) => {
    setImages((prev) => {
      const updated = prev.filter((img) => img.id !== id);
      saveImages(updated);
      return updated;
    });
  };

  // Composant pour afficher une image avec ses métadonnées (Resources used)
  const ImageCard = ({
    image,
    onDelete,
    onCopy,
    onExport,
  }: {
    image: GeneratedImage;
    onDelete: (id: string) => void;
    onCopy: (image: GeneratedImage) => void;
    onExport: (image: GeneratedImage) => void;
  }) => {
    const [showMetadata, setShowMetadata] = useState(false);

    const getSamplerLabel = (sampler?: string) => {
      const found = samplers.find((s) => s.value === sampler);
      return found?.label || sampler || "N/A";
    };

    const getModelLabel = (model?: string) => {
      const found = models.find((m) => m.value === model);
      return found?.label || model || "N/A";
    };

    return (
      <div className="group relative rounded-3xl border border-white/10 bg-black/30 p-2 transition hover:border-indigo-400">
        <button
          type="button"
          onClick={() => onDelete(image.id)}
          className="absolute right-3 top-3 z-10 flex h-6 w-6 items-center justify-center rounded-full border border-white/15 bg-black/80 text-xs text-slate-300 opacity-0 transition group-hover:opacity-100 hover:border-rose-400 hover:text-rose-200"
          title="Retirer cette image de la galerie"
        >
          ×
        </button>
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img
          src={`data:image/png;base64,${image.base64}`}
          alt={`seed ${image.seed}`}
          className="h-64 w-full rounded-2xl object-cover"
        />
        <div className="mt-2 space-y-2">
          <div className="flex items-center justify-between text-xs text-slate-400">
            <span>Seed: {image.seed}</span>
            <div className="flex gap-2">
              <button
                type="button"
                onClick={() => onCopy(image)}
                className="text-indigo-300 transition hover:text-white"
                title="Copier base64"
              >
                Copier
              </button>
              <button
                type="button"
                onClick={() => onExport(image)}
                className="text-green-300 transition hover:text-white"
                title="Télécharger PNG"
              >
                Télécharger
              </button>
            </div>
          </div>
          <button
            type="button"
            onClick={() => setShowMetadata(!showMetadata)}
            className="w-full text-left text-[10px] uppercase tracking-wider text-slate-500 transition hover:text-slate-300"
          >
            {showMetadata ? "▼ Masquer" : "▶ Afficher"} les paramètres
          </button>
          {showMetadata && (
            <div className="rounded-xl border border-white/10 bg-slate-900/60 p-3 text-[11px] text-slate-300">
              <div className="mb-2 text-xs font-semibold uppercase tracking-wider text-slate-400">
                Resources Used
              </div>
              <div className="space-y-1.5">
                {image.model && (
                  <div className="flex justify-between">
                    <span className="text-slate-400">Modèle:</span>
                    <span className="font-semibold">{getModelLabel(image.model)}</span>
                  </div>
                )}
                {image.sampler && (
                  <div className="flex justify-between">
                    <span className="text-slate-400">Sampler:</span>
                    <span>{getSamplerLabel(image.sampler)}</span>
                  </div>
                )}
                {image.steps && (
                  <div className="flex justify-between">
                    <span className="text-slate-400">Steps:</span>
                    <span>{image.steps}</span>
                  </div>
                )}
                {image.cfg_scale && (
                  <div className="flex justify-between">
                    <span className="text-slate-400">CFG Scale:</span>
                    <span>{image.cfg_scale.toFixed(1)}</span>
                  </div>
                )}
                {image.resolution && (
                  <div className="flex justify-between">
                    <span className="text-slate-400">Résolution:</span>
                    <span>{image.resolution}</span>
                  </div>
                )}
                {image.clip_skip && (
                  <div className="flex justify-between">
                    <span className="text-slate-400">Clip Skip:</span>
                    <span>{image.clip_skip}</span>
                  </div>
                )}
                {image.prompt && (
                  <div className="mt-2 border-t border-white/10 pt-2">
                    <div className="mb-1 text-slate-400">Prompt:</div>
                    <div className="max-h-20 overflow-y-auto text-slate-200">{image.prompt}</div>
                  </div>
                )}
                {image.negative_prompt && (
                  <div className="mt-2 border-t border-white/10 pt-2">
                    <div className="mb-1 text-slate-400">Negative Prompt:</div>
                    <div className="max-h-20 overflow-y-auto text-slate-200">{image.negative_prompt}</div>
                  </div>
                )}
                {image.loras && image.loras.length > 0 && (
                  <div className="mt-2 border-t border-white/10 pt-2 space-y-1.5">
                    <div className="text-[10px] uppercase tracking-wider text-slate-500">
                      LoRA appliqués
                    </div>
                    {image.loras.map((lora) => (
                      <div
                        key={`${image.id}-${lora.key}-${lora.label}`}
                        className="rounded-xl border border-white/10 bg-slate-950/40 px-3 py-2 text-xs text-slate-200"
                      >
                        <div className="flex items-center justify-between">
                          <span>{lora.label}</span>
                          <span className="text-slate-300">poids {lora.weight.toFixed(2)}</span>
                        </div>
                        <p className="text-[10px] uppercase tracking-wider text-slate-500">
                          {lora.type ?? "LoRA"}
                        </p>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      </div>
    );
  };

  return (
    <div className="min-h-screen bg-slate-950 bg-[radial-gradient(circle_at_top,_#1e293b,_#020617)] text-slate-100">
      <main className="mx-auto flex max-w-7xl gap-6 px-4 py-10 lg:px-8">
        <aside className="hidden w-72 rounded-3xl border border-white/10 bg-white/5 p-5 shadow-2xl backdrop-blur md:block">
          <div>
            <h2 className="text-sm font-semibold uppercase tracking-[0.2em] text-slate-400">
              Historique
            </h2>
            <p className="mb-4 mt-1 text-sm text-slate-400">
              Vos derniers prompts restent hors-ligne.
            </p>
            <div className="space-y-3 overflow-y-auto pr-1 max-h-[75vh]">
              {history.length === 0 && (
                <p className="text-sm text-slate-500">
                  Aucune génération pour l&apos;instant.
                </p>
              )}
              {history.map((entry) => (
                <div
                  key={entry.id}
                  className="group relative flex items-start gap-2 rounded-2xl border border-white/10 bg-slate-900/60 p-3 transition hover:border-white/30"
                >
                  <button
                    type="button"
                    onClick={() => handleLoadHistory(entry)}
                    className="flex flex-1 items-center gap-3 text-left"
                  >
                    {entry.thumbnail ? (
                      // eslint-disable-next-line @next/next/no-img-element
                      <img
                        src={entry.thumbnail}
                        alt="miniature"
                        className="h-14 w-14 rounded-xl object-cover"
                      />
                    ) : (
                      <div className="flex h-14 w-14 items-center justify-center rounded-xl bg-slate-800 text-xs text-slate-400">
                        {entry.model}
                      </div>
                    )}
                    <div className="flex-1 min-w-0">
                      <p className="line-clamp-2 text-sm font-medium text-white">
                        {entry.prompt}
                      </p>
                      <p className="text-xs text-slate-500">
                        {new Date(entry.timestamp).toLocaleTimeString("fr-FR", {
                          hour: "2-digit",
                          minute: "2-digit",
                        })}
                      </p>
                    </div>
                  </button>
                  <button
                    type="button"
                    onClick={() => handleDeleteHistory(entry.id)}
                    className="absolute right-2 top-2 flex h-5 w-5 items-center justify-center rounded-full border border-white/15 bg-slate-900/80 text-[11px] text-slate-400 opacity-0 transition group-hover:opacity-100 hover:border-rose-400 hover:text-rose-200"
                    title="Supprimer cet historique"
                  >
                    ×
                  </button>
                </div>
              ))}
            </div>
          </div>
        </aside>

        <section className="flex-1 space-y-7">
          <header className="rounded-3xl border border-white/10 bg-gradient-to-r from-slate-900/80 to-slate-900/40 p-6 shadow-2xl">
            <div className="flex items-start justify-between gap-4">
              <div>
                <p className="text-xs uppercase tracking-[0.3em] text-indigo-300">
                  Stable Diffusion Local
                </p>
                <h1 className="mt-2 text-3xl font-semibold text-white">
                  Atelier SDXL / SD1.5
          </h1>
                <p className="mt-2 text-sm text-slate-300">
                  Prompt + paramètres essentiels, inspirés de CivitAI et ComfyUI,
                  fonctionnement 100% offline.
                </p>
              </div>
              <div className="flex items-center gap-2 text-xs">
                <button
                  type="button"
                  onClick={() => setMode("image")}
                  className={`rounded-full px-3 py-1 border ${
                    mode === "image"
                      ? "bg-white text-slate-900 border-white"
                      : "bg-slate-900/60 text-slate-300 border-white/20"
                  }`}
                >
                  Image
                </button>
                <button
                  type="button"
                  onClick={() => setMode("video")}
                  className={`rounded-full px-3 py-1 border ${
                    mode === "video"
                      ? "bg-white text-slate-900 border-white"
                      : "bg-slate-900/60 text-slate-300 border-white/20"
                  }`}
                >
                  Vidéo (beta)
                </button>
              </div>
            </div>
          </header>

          <div className="grid gap-6 lg:grid-cols-[minmax(0,1.2fr)_minmax(0,0.8fr)]">
            <div className="space-y-6 rounded-3xl border border-white/10 bg-white/5 p-6 shadow-2xl">
              <div>
                <div className="flex items-center justify-between mb-2">
                  <label className="text-xs font-semibold uppercase tracking-[0.2em] text-slate-400">
                    Prompt
                  </label>
                  <div className="flex gap-1">
                    {promptPresets.map((preset) => (
                      <button
                        key={preset.name}
                        onClick={() => handleLoadPreset(preset)}
                        className="rounded-lg border border-white/10 bg-slate-900/60 px-2 py-1 text-[10px] uppercase tracking-wide text-slate-300 transition hover:bg-slate-800 hover:text-white"
                        title={preset.description}
                      >
                        {preset.name}
                      </button>
                    ))}
                  </div>
                </div>
                <textarea
                  className="min-h-[120px] w-full rounded-2xl border border-white/10 bg-slate-900/60 p-4 text-sm text-white outline-none transition focus:border-indigo-400"
                  placeholder="Texte principal qui décrit la scène..."
                  value={prompt}
                  onChange={(event) => setPrompt(event.target.value)}
                />
              </div>

              <div>
                <label className="text-xs font-semibold uppercase tracking-[0.2em] text-slate-400">
                  Negative prompt
                </label>
                <textarea
                  className="mt-2 min-h-[80px] w-full rounded-2xl border border-white/10 bg-slate-900/60 p-4 text-sm text-white outline-none transition focus:border-rose-400"
                  placeholder="Ce que vous ne voulez pas voir…"
                  value={negativePrompt}
                  onChange={(event) => setNegativePrompt(event.target.value)}
                />
              </div>

              <div className="grid gap-4 md:grid-cols-2">
                <div>
                  <p className="text-xs font-semibold uppercase tracking-[0.2em] text-slate-400">
                    Sampler
                  </p>
                  <select
                    className="mt-2 w-full rounded-2xl border border-white/10 bg-slate-900/60 p-3 text-sm text-white"
                    value={sampler}
                    onChange={(event) => setSampler(event.target.value as Sampler)}
                  >
                    {samplers.map((option) => (
                      <option key={option.value} value={option.value}>
                        {option.label}
                      </option>
                    ))}
                  </select>
                  <p className="mt-2 text-xs text-slate-500">
                    Euler A & DPM++ conseillés pour un usage général.
                  </p>
                </div>

                <div>
                  <div className="flex items-center justify-between mb-2">
                    <p className="text-xs font-semibold uppercase tracking-[0.2em] text-slate-400">
                      Résolution
                    </p>
                    <label className="flex items-center gap-2 text-[10px] text-slate-400">
                      <input
                        type="checkbox"
                        checked={useAspectRatio}
                        onChange={(e) => {
                          setUseAspectRatio(e.target.checked);
                          if (e.target.checked) {
                            const selected = aspectRatios.find((r) => r.value === aspectRatio);
                            if (selected) {
                              setCustomWidth(selected.width);
                              setCustomHeight(selected.height);
                            }
                          }
                        }}
                        className="rounded"
                      />
                      Ratio personnalisé
                    </label>
                  </div>
                  {!useAspectRatio ? (
                    <>
                      <select
                        className="mt-2 w-full rounded-2xl border border-white/10 bg-slate-900/60 p-3 text-sm text-white"
                        value={resolution}
                        onChange={(event) =>
                          setResolution(event.target.value as Resolution)
                        }
                      >
                        {resolutions.map((option) => (
                          <option key={option.value} value={option.value}>
                            {option.label}
                          </option>
                        ))}
                      </select>
                      <p className="mt-2 text-xs text-slate-500">
                        {resolutions.find((item) => item.value === resolution)?.hint}
                      </p>
                    </>
                  ) : (
                    <div className="mt-2 space-y-3">
                      <div>
                        <label className="text-[10px] uppercase tracking-wider text-slate-400">
                          Ratio d&apos;aspect
                        </label>
                        <select
                          className="mt-1 w-full rounded-xl border border-white/10 bg-slate-900/60 p-2 text-sm text-white"
                          value={aspectRatio}
                          onChange={(e) => {
                            setAspectRatio(e.target.value);
                            const selected = aspectRatios.find((r) => r.value === e.target.value);
                            if (selected) {
                              setCustomWidth(selected.width);
                              setCustomHeight(selected.height);
                            }
                          }}
                        >
                          {aspectRatios.map((ratio) => (
                            <option key={ratio.value} value={ratio.value}>
                              {ratio.label} ({ratio.width}×{ratio.height})
                            </option>
                          ))}
                        </select>
                      </div>
                      <div className="grid grid-cols-2 gap-2">
                        <div>
                          <label className="text-[10px] uppercase tracking-wider text-slate-400">
                            Largeur
                          </label>
                          <input
                            type="number"
                            min={256}
                            max={2048}
                            step={64}
                            value={customWidth}
                            onChange={(e) => {
                              const w = Number(e.target.value);
                              setCustomWidth(w);
                              // Maintient le ratio si un ratio est sélectionné
                              const selected = aspectRatios.find((r) => r.value === aspectRatio);
                              if (selected) {
                                const ratio = selected.height / selected.width;
                                setCustomHeight(Math.round(w * ratio));
                              }
                            }}
                            className="mt-1 w-full rounded-xl border border-white/10 bg-slate-900/60 px-3 py-2 text-sm text-white"
                          />
                        </div>
                        <div>
                          <label className="text-[10px] uppercase tracking-wider text-slate-400">
                            Hauteur
                          </label>
                          <input
                            type="number"
                            min={256}
                            max={2048}
                            step={64}
                            value={customHeight}
                            onChange={(e) => {
                              const h = Number(e.target.value);
                              setCustomHeight(h);
                              // Maintient le ratio si un ratio est sélectionné
                              const selected = aspectRatios.find((r) => r.value === aspectRatio);
                              if (selected) {
                                const ratio = selected.width / selected.height;
                                setCustomWidth(Math.round(h * ratio));
                              }
                            }}
                            className="mt-1 w-full rounded-xl border border-white/10 bg-slate-900/60 px-3 py-2 text-sm text-white"
                          />
                        </div>
                      </div>
                      <p className="text-[10px] text-slate-500">
                        Dimensions personnalisées (multiples de 64 recommandés)
                      </p>
                    </div>
                  )}
                </div>
              </div>

              <div className="grid gap-6 md:grid-cols-2">
                <div>
                  <div className="flex items-center justify-between text-xs uppercase tracking-[0.2em] text-slate-400">
                    <span>Steps</span>
                    <span className="text-[11px] text-slate-300">{stepsHint}</span>
                  </div>
                  <input
                    type="range"
                    min={10}
                    max={60}
                    value={steps}
                    onChange={(event) => setSteps(Number(event.target.value))}
                    className="mt-3 w-full accent-indigo-400"
                  />
                  <p className="mt-2 text-sm font-semibold text-white">{steps} itérations</p>
                </div>

                <div>
                  <div className="flex items-center justify-between text-xs uppercase tracking-[0.2em] text-slate-400">
                    <span>CFG Scale</span>
                    <span className="text-[11px] text-slate-300">{cfgHint}</span>
                  </div>
                  <input
                    type="range"
                    min={1}
                    max={14}
                    step={0.5}
                    value={cfgScale}
                    onChange={(event) => setCfgScale(Number(event.target.value))}
                    className="mt-3 w-full accent-indigo-400"
                  />
                  <p className="mt-2 text-sm font-semibold text-white">
                    Intensité : {cfgScale.toFixed(1)}
          </p>
        </div>
              </div>
            </div>

            <div className="space-y-6 rounded-3xl border border-white/10 bg-white/5 p-6 shadow-2xl">
              <div>
                <p className="text-xs font-semibold uppercase tracking-[0.2em] text-slate-400">
                  Choisir le modèle (CivitAI-like)
                </p>
                <div className="mt-3 space-y-2">
                  {models.map((item) => (
                    <label
                      key={item.value}
                      className={`flex cursor-pointer items-center justify-between rounded-2xl border p-3 text-sm ${
                        model === item.value
                          ? "border-indigo-400 bg-indigo-400/10 text-white"
                          : "border-white/10 text-slate-300"
                      } ${!item.available ? "cursor-not-allowed opacity-60" : "hover:border-white/30"}`}
                    >
                      <div>
                        <div className="flex items-center gap-2">
                          <p className="font-semibold">{item.label}</p>
                          {item.highlight && (
                            <span className="rounded-full bg-indigo-500/20 px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wide text-indigo-200">
                              {item.highlight}
                            </span>
                          )}
                        </div>
                        <p className="text-xs text-slate-400">{item.note}</p>
                        {!item.available && (
                          <p className="text-[10px] text-amber-200">
                            Disponible prochainement – utilisez SDXL pour la meilleure qualité.
                          </p>
                        )}
                      </div>
                      <input
                        type="radio"
                        name="model"
                        value={item.value}
                        checked={model === item.value}
                        onChange={() => item.available && setModel(item.value)}
                        className="accent-indigo-400"
                        disabled={!item.available}
                      />
                    </label>
                  ))}
                </div>
              </div>

              <div className="grid gap-4 md:grid-cols-2">
                <div>
                  <p className="text-xs font-semibold uppercase tracking-[0.2em] text-slate-400">
                    Seed
                  </p>
                  <input
                    className="mt-2 w-full rounded-2xl border border-white/10 bg-slate-900/60 p-3 text-sm text-white"
                    type="number"
                    value={seed}
                    onChange={(event) => setSeed(event.target.value)}
                    placeholder="-1 pour aléatoire"
                  />
                  <p className="mt-2 text-xs text-slate-500">-1 = seed aléatoire</p>
                </div>

                <div>
                  <p className="text-xs font-semibold uppercase tracking-[0.2em] text-slate-400">
                    Clip skip
                  </p>
                  <input
                    className="mt-2 w-full rounded-2xl border border-white/10 bg-slate-900/60 p-3 text-sm text-white"
                    type="number"
                    min={1}
                    max={12}
                    value={clipSkip}
                    onChange={(event) => setClipSkip(Number(event.target.value))}
                  />
                  <p className="mt-2 text-xs text-slate-500">2 recommandé pour certains modèles SDXL.</p>
                </div>
              </div>

              <div className="border-t border-white/5 pt-5">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-xs font-semibold uppercase tracking-[0.2em] text-slate-400">
                      Styles / LoRA
                    </p>
                    <p className="text-[11px] text-slate-500">
                      Combinez plusieurs LoRA comme sur Leonardo.ai
                    </p>
                  </div>
                  {selectedLoras.length > 0 && (
                    <button
                      className="text-[10px] uppercase tracking-[0.2em] text-rose-200 transition hover:text-white"
                      onClick={handleClearLoras}
                    >
                      Effacer
                    </button>
                  )}
                </div>
                {availableLoras.length === 0 ? (
                  <p className="mt-3 rounded-2xl border border-dashed border-white/10 p-3 text-xs text-slate-400">
                    Ajoutez vos fichiers LoRA dans <code>backend/models/lora</code> puis rechargez la page.
                  </p>
                ) : (
                  <div className="mt-4 space-y-3">
                    {availableLoras.map((option) => {
                      const active = selectedLoras.find((item) => item.key === option.key);
                      return (
                        <div
                          key={option.key}
                          className={`rounded-2xl border p-3 ${
                            active ? "border-indigo-400 bg-indigo-500/10" : "border-white/10 bg-slate-900/40"
                          }`}
                        >
                          <label className="flex cursor-pointer items-center justify-between gap-3">
                            <div>
                              <p className="text-sm font-semibold text-white">{option.label}</p>
                              {option.description && (
                                <p className="text-[11px] text-slate-400">{option.description}</p>
                              )}
                            </div>
                            <input
                              type="checkbox"
                              checked={!!active}
                              onChange={() => handleToggleLora(option.key)}
                              className="h-4 w-4 accent-indigo-400"
                            />
                          </label>
                          {active && (
                            <div className="mt-3 space-y-2">
                              <input
                                type="range"
                                min={0}
                                max={1.5}
                                step={0.05}
                                value={active.weight}
                                onChange={(event) =>
                                  handleLoraWeightChange(option.key, Number(event.target.value))
                                }
                                className="w-full accent-indigo-400"
                              />
                              <div className="flex items-center justify-between text-xs text-slate-300">
                                <span>Poids</span>
                                <span className="font-semibold">{active.weight.toFixed(2)}</span>
                              </div>
                            </div>
                          )}
                        </div>
                      );
                    })}
                  </div>
                )}
              </div>

              {mode === "image" && (
                <div className="grid gap-4 md:grid-cols-2 mt-2">
                  <div>
                    <p className="text-xs font-semibold uppercase tracking-[0.2em] text-slate-400">
                      Image count
                    </p>
                    <input
                      className="mt-2 w-full rounded-2xl border border-white/10 bg-slate-900/60 p-3 text-sm text-white"
                      type="number"
                      min={1}
                      max={4}
                      value={imageCount}
                      onChange={(event) =>
                        setImageCount(Math.min(4, Math.max(1, Number(event.target.value))))
                      }
                    />
                    <p className="mt-2 text-xs text-slate-500">Jusqu&apos;à 4 images.</p>
                  </div>
                </div>
              )}

              {mode === "video" && (
                <>
                  <div className="mt-4">
                    <p className="text-xs font-semibold uppercase tracking-[0.2em] text-slate-400">
                      Image de départ (PNG/JPEG)
                    </p>
                    <input
                      type="file"
                      accept="image/*"
                      className="mt-2 text-xs text-slate-300"
                      onChange={async (event) => {
                        const file = event.target.files?.[0];
                        if (!file) {
                          setInitImageBase64(null);
                          return;
                        }
                        const reader = new FileReader();
                        reader.onload = () => {
                          const result = reader.result;
                          if (typeof result === "string") {
                            const base64 = result.split(",")[1] ?? result;
                            setInitImageBase64(base64);
                          }
                        };
                        reader.readAsDataURL(file);
                      }}
                    />
                    <p className="mt-2 text-[11px] text-slate-500">
                      Obligatoire pour la vidéo pour le moment (l&apos;animation part de cette image).
                    </p>
                  </div>

                  <div className="mt-4">
                    <label className="text-xs font-semibold uppercase tracking-[0.2em] text-slate-400">
                      Durée de la vidéo (secondes)
                    </label>
                    <input
                      type="number"
                      min={0.5}
                      max={5}
                      step={0.1}
                      value={videoDuration}
                      onChange={(e) => {
                        const duration = Number(e.target.value);
                        setVideoDuration(duration);
                        // Calcule le nombre de frames nécessaire
                        const calculatedFrames = Math.round(duration * fps);
                        // Limite entre 6 et 16 frames
                        const clampedFrames = Math.max(6, Math.min(16, calculatedFrames));
                        setNumFrames(clampedFrames);
                      }}
                      className="mt-2 w-full rounded-xl border border-white/10 bg-white/5 px-3 py-2 text-sm text-slate-100 placeholder:text-slate-500 focus:border-indigo-500 focus:outline-none"
                    />
                    <p className="mt-1 text-[11px] text-slate-500">
                      Durée cible (0.5-5s). Frames calculés: {numFrames} ({(numFrames / fps).toFixed(2)}s réelle)
                    </p>
                  </div>

                  <div className="mt-4">
                    <label className="text-xs font-semibold uppercase tracking-[0.2em] text-slate-400">
                      FPS (Images par seconde)
                    </label>
                    <input
                      type="number"
                      min={3}
                      max={30}
                      value={fps}
                      onChange={(e) => {
                        const newFps = Number(e.target.value);
                        setFps(newFps);
                        // Recalcule les frames pour maintenir la durée
                        const calculatedFrames = Math.round(videoDuration * newFps);
                        const clampedFrames = Math.max(6, Math.min(16, calculatedFrames));
                        setNumFrames(clampedFrames);
                      }}
                      className="mt-2 w-full rounded-xl border border-white/10 bg-white/5 px-3 py-2 text-sm text-slate-100 placeholder:text-slate-500 focus:border-indigo-500 focus:outline-none"
                    />
                    <p className="mt-1 text-[11px] text-slate-500">
                      Vitesse de lecture (3-30, recommandé: 6-8)
                    </p>
                  </div>

                  <div className="mt-4">
                    <label className="text-xs font-semibold uppercase tracking-[0.2em] text-slate-400">
                      Nombre de frames (avancé)
                    </label>
                    <input
                      type="number"
                      min={6}
                      max={16}
                      value={numFrames}
                      onChange={(e) => {
                        const frames = Number(e.target.value);
                        setNumFrames(frames);
                        // Met à jour la durée estimée
                        setVideoDuration(frames / fps);
                      }}
                      className="mt-2 w-full rounded-xl border border-white/10 bg-white/5 px-3 py-2 text-sm text-slate-100 placeholder:text-slate-500 focus:border-indigo-500 focus:outline-none"
                    />
                    <p className="mt-1 text-[11px] text-slate-500">
                      Nombre d&apos;images (6-16, recommandé: 8 pour 8 Go VRAM). Durée: {(numFrames / fps).toFixed(2)}s
                    </p>
                  </div>
                </>
              )}

              <button
                onClick={handleGenerate}
                disabled={isGenerating}
                className="relative flex w-full items-center justify-center rounded-2xl bg-gradient-to-r from-indigo-500 via-purple-500 to-rose-500 py-3 text-sm font-semibold text-white shadow-lg shadow-indigo-500/30 transition hover:scale-[1.01] disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {isGenerating ? (
                  <>
                    <span className="mr-2">Génération en cours…</span>
                    {generationProgress !== null && generationProgress < 100 && (
                      <span className="text-xs opacity-75">
                        {Math.round(generationProgress)}%
                      </span>
                    )}
                  </>
                ) : (
                  "Generate"
                )}
              </button>

              {generationProgress !== null && generationProgress < 100 && (
                <div className="h-2 w-full overflow-hidden rounded-full bg-slate-800">
                  <div
                    className="h-full bg-gradient-to-r from-indigo-500 to-purple-500 transition-all duration-300"
                    style={{ width: `${generationProgress}%` }}
                  />
                </div>
              )}

              {lastDuration !== null && (
                <p className="mt-2 text-xs text-slate-400">
                  Temps de génération&nbsp;:{" "}
                  <span className="font-semibold text-slate-100">
                    {lastDuration.toFixed(1)}&nbsp;s
                  </span>
                </p>
              )}

              {error && (
                <p className="rounded-2xl border border-rose-400/40 bg-rose-500/10 p-3 text-sm text-rose-200">
                  {error}
                </p>
              )}

              <div className="rounded-2xl border border-white/10 bg-slate-900/40 p-4 text-xs text-slate-400">
                Backend attendu sur{" "}
                <span className="font-semibold text-white">{apiBase}/generate</span>
                . Configurez la variable <code>NEXT_PUBLIC_API_BASE_URL</code> si besoin.
              </div>
            </div>
          </div>

          <div className="rounded-3xl border border-white/10 bg-white/5 p-6 shadow-2xl">
            <div className="flex flex-wrap items-center justify-between gap-3">
              <div>
                <p className="text-xs uppercase tracking-[0.2em] text-slate-400">
                  Galerie
                </p>
                <h2 className="text-2xl font-semibold text-white">Dernières images</h2>
              </div>
              {images.length > 0 && (
                <button
                  onClick={() => {
                    setImages([]);
                    saveImages([]);
                  }}
                  className="text-xs uppercase tracking-[0.2em] text-slate-400 transition hover:text-white"
                >
                  Effacer
                </button>
              )}
            </div>

            <div className="mt-6 grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
              {images.length === 0 && (
                <div className="col-span-full rounded-3xl border border-dashed border-white/20 p-10 text-center text-sm text-slate-400">
                  Les rendus apparaîtront ici.
                </div>
              )}
              {images.map((image) => (
                <ImageCard
                  key={image.id}
                  image={image}
                  onDelete={handleDeleteImage}
                  onCopy={handleCopyBase64}
                  onExport={handleExportImage}
                />
              ))}
            </div>

            {videos.length > 0 && (
              <div className="mt-10">
                <div className="flex items-center justify-between">
                  <h3 className="text-sm font-semibold text-slate-200 uppercase tracking-[0.2em]">
                    Vidéos
                  </h3>
                </div>
                <div className="mt-4 grid gap-4 sm:grid-cols-2">
                  {videos.map((video) => (
                    <div
                      key={video.id}
                      className="rounded-3xl border border-white/10 bg-black/40 p-2"
                    >
                      <video
                        controls
                        className="w-full rounded-2xl"
                        src={`data:video/mp4;base64,${video.mp4Base64}`}
                      />
                      {video.durationSeconds != null && (
                        <p className="mt-1 text-xs text-slate-400">
                          Durée&nbsp;:{" "}
                          <span className="font-semibold text-slate-100">
                            {video.durationSeconds.toFixed(1)}&nbsp;s
                          </span>
                        </p>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            )}
        </div>
        </section>
      </main>
    </div>
  );
}
