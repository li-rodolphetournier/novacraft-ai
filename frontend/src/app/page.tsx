"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { toast } from "react-toastify";

import { aspectRatios, promptPresets, resolutions, samplers } from "@/config/generator";
import {
  GeneratedImage,
  GeneratedVideo,
  HistoryEntry,
  Job,
  LoraOption,
  ModelKey,
  Mode,
  PromptPreset,
  Resolution,
  Sampler,
  SelectedLora,
  VideoMode,
} from "@/types/generator";
import { HistorySidebar } from "@/components/generator/HistorySidebar";
import { GeneratorHeader } from "@/components/generator/GeneratorHeader";
import { PromptSettingsPanel } from "@/components/generator/PromptSettingsPanel";
import { ModelSettingsPanel } from "@/components/generator/ModelSettingsPanel";
import { GallerySection } from "@/components/generator/GallerySection";
import { JobQueuePanel } from "@/components/generator/JobQueuePanel";
import { estimateVramUsage, type VramEstimate } from "@/utils/vramEstimator";
import { Modal, PromptModal } from "@/components/common/Modal";

const apiBase = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

const HISTORY_STORAGE_KEY = "sd_generator_history";
const MAX_HISTORY_ENTRIES = 50;
const GALLERY_STORAGE_KEY = "sd_generator_gallery";
const MAX_GALLERY_IMAGES = 100;
const CUSTOM_PRESET_STORAGE_KEY = "sd_generator_custom_presets";

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
  const [availableModels, setAvailableModels] = useState<Set<string>>(new Set(["sdxl"]));
  const [mode, setMode] = useState<Mode>("image");
  type ChatEntry = { role: "user" | "assistant"; content: string; images?: string[] };
  const [chatMessages, setChatMessages] = useState<ChatEntry[]>([]);
  const [chatInput, setChatInput] = useState("");
  const [isChatting, setIsChatting] = useState(false);
  const [chatAttachment, setChatAttachment] = useState<string | null>(null);
  const [chatAttachmentName, setChatAttachmentName] = useState<string | null>(null);
  const [videos, setVideos] = useState<GeneratedVideo[]>([]);
  const [initImageBase64, setInitImageBase64] = useState<string | null>(null); // vidéo
  const [imageInitBase64, setImageInitBase64] = useState<string | null>(null); // image mode
  const [imageInitName, setImageInitName] = useState<string | null>(null);
  const [imageInitStrength, setImageInitStrength] = useState(0.5);
  const [numFrames, setNumFrames] = useState(8);
  const [fps, setFps] = useState(6);
  const [videoDuration, setVideoDuration] = useState(1.33); // 8 frames / 6 fps ≈ 1.33s
  const [videoMode, setVideoMode] = useState<VideoMode>("img2vid");
  const [availableLoras, setAvailableLoras] = useState<LoraOption[]>([]);
  const [selectedLoras, setSelectedLoras] = useState<SelectedLora[]>([]);
  const [showNSFW, setShowNSFW] = useState(false);
  const [customPresets, setCustomPresets] = useState<PromptPreset[]>([]);
  const [activePresetId, setActivePresetId] = useState<string | null>(null);
  const [jobs, setJobs] = useState<Job[]>([]);
  const [selectedJobId, setSelectedJobId] = useState<string | null>(null);
  const [jobsError, setJobsError] = useState<string | null>(null);
  const [showNamePrompt, setShowNamePrompt] = useState(false);
  const [showDescPrompt, setShowDescPrompt] = useState(false);
  const [showDeleteConfirm, setShowDeleteConfirm] = useState(false);
  const [presetToDelete, setPresetToDelete] = useState<string | null>(null);
  const [newPresetName, setNewPresetName] = useState("");
  const [newPresetDesc, setNewPresetDesc] = useState("");

  const processedJobIdsRef = useRef<Set<string>>(new Set());

  // Charger les images depuis le backend au montage
  useEffect(() => {
    const loadImages = async () => {
      try {
        // D'abord charger depuis le backend
        const response = await fetch(`${apiBase}/storage/images`);
        if (response.ok) {
          const data = await response.json();
          const backendImages: GeneratedImage[] = (data.images || []).map((img: any) => ({
            id: img.id,
            seed: img.seed,
            base64: img.base64 || "",
            model: img.model,
            sampler: img.sampler,
            steps: img.steps,
            cfg_scale: img.cfg_scale,
            resolution: img.resolution,
            prompt: img.prompt,
            negative_prompt: img.negative_prompt,
            clip_skip: img.clip_skip,
            loras: img.loras || [],
          })).filter((img: GeneratedImage) => img.base64); // Filtrer les images sans base64
        
          if (backendImages.length > 0) {
            setImages(backendImages.slice(0, MAX_GALLERY_IMAGES));
            saveImages(backendImages.slice(0, MAX_GALLERY_IMAGES));
            return;
          }
        }
      } catch (err) {
        console.warn("Impossible de charger depuis le backend, utilisation du localStorage", err);
      }
      
      // Fallback sur localStorage
      try {
        const stored = localStorage.getItem(GALLERY_STORAGE_KEY);
        if (stored) {
          const parsed = JSON.parse(stored) as GeneratedImage[];
          setImages(parsed.slice(0, MAX_GALLERY_IMAGES));
        }
      } catch {
        // Ignorer les erreurs de parsing
      }
    };
    
    loadImages();
  }, []);

  // Charger les vidéos depuis le backend au montage
  useEffect(() => {
    const loadVideos = async () => {
      try {
        const response = await fetch(`${apiBase}/storage/videos`);
        if (response.ok) {
          const data = await response.json();
          const backendVideos: GeneratedVideo[] = (data.videos || []).map((vid: any) => ({
            id: vid.id,
            mp4Base64: vid.mp4_base64 || vid.mp4Base64 || "",
            durationSeconds: vid.duration_seconds || vid.durationSeconds,
          })).filter((vid: GeneratedVideo) => vid.mp4Base64);
          
          if (backendVideos.length > 0) {
            setVideos(backendVideos);
          }
        }
      } catch (err) {
        console.warn("Impossible de charger les vidéos depuis le backend", err);
      }
    };
    
    loadVideos();
  }, []);

  // Vérifier les modèles disponibles au montage
  useEffect(() => {
    fetch(`${apiBase}/models`)
      .then((res) => res.json())
      .then((data: { enabled?: string[]; models?: Array<{ key: string; installed: boolean; enabled: boolean }> }) => {
        // Utilise les modèles installés ET activés
        const available = new Set<string>();
        if (data.models) {
          // Filtre les modèles installés et activés
          data.models
            .filter((m) => m.installed && m.enabled)
            .forEach((m) => available.add(m.key));
        } else if (data.enabled) {
          // Fallback sur enabled si models n'est pas disponible
          data.enabled.forEach((key) => available.add(key));
        }
        setAvailableModels(available);
      })
      .catch(() => {
        // Ignorer les erreurs, garder les valeurs par défaut
      });
  }, []);

  useEffect(() => {
    fetch(`${apiBase}/loras`)
      .then((res) => res.json())
      .then((data: { loras?: Array<Record<string, any>> }) => {
        const mapped =
          data.loras?.map((lora) => ({
            key: lora.key,
            label: lora.label,
            description: lora.description,
            defaultWeight: lora.default_weight ?? 0.5,
            nsfw: Boolean(lora.nsfw),
          })) ?? [];
        setAvailableLoras(mapped);
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

  const fetchHistory = useCallback(async () => {
    try {
      const response = await fetch(`${apiBase}/storage/history`);
      if (response.ok) {
        const data = await response.json();
        const backendHistory: HistoryEntry[] = (data.history || []).map((entry: any) => ({
          id: entry.id,
          prompt: entry.prompt,
          negativePrompt: entry.negative_prompt || "",
          model: entry.model,
          thumbnail: entry.thumbnail_base64
            ? `data:image/png;base64,${entry.thumbnail_base64}`
            : undefined,
          timestamp: entry.timestamp ?? Date.now(),
          settings: entry.settings
            ? {
                sampler: entry.settings.sampler,
                steps: entry.settings.steps,
                cfgScale: entry.settings.cfg_scale,
                resolution: entry.settings.resolution,
                seed: entry.settings.seed,
                useAspectRatio: entry.settings.use_aspect_ratio,
                aspectRatio: entry.settings.aspect_ratio,
                customWidth: entry.settings.custom_width,
                customHeight: entry.settings.custom_height,
                loras: entry.settings.loras,
              }
            : undefined,
        }));
        setHistory(backendHistory);
        saveHistory(backendHistory);
        return true;
      }
    } catch (err) {
      console.warn("Impossible de charger l'historique depuis le backend", err);
    }

    try {
      const stored = localStorage.getItem(HISTORY_STORAGE_KEY);
      if (stored) {
        const parsed = JSON.parse(stored) as HistoryEntry[];
        setHistory(parsed.slice(0, MAX_HISTORY_ENTRIES));
      }
    } catch {
      // Ignorer les erreurs de parsing
    }

    return false;
  }, [apiBase]);

  // Sauvegarder les images dans localStorage
  const saveImages = useCallback((newImages: GeneratedImage[]) => {
    try {
      localStorage.setItem(GALLERY_STORAGE_KEY, JSON.stringify(newImages.slice(0, MAX_GALLERY_IMAGES)));
    } catch {
      // Ignorer les erreurs de stockage
    }
  }, []);

  useEffect(() => {
    fetchHistory();
  }, [fetchHistory]);

  const refreshJobs = useCallback(async () => {
    try {
      const response = await fetch(`${apiBase}/jobs`);
      if (!response.ok) {
        throw new Error("Impossible de charger la file de jobs.");
      }
      const data = await response.json();
      setJobs(Array.isArray(data.jobs) ? data.jobs : []);
      setJobsError(null);
    } catch (err) {
      setJobsError(err instanceof Error ? err.message : "Erreur inconnue (jobs).");
    }
  }, [apiBase]);

  useEffect(() => {
    refreshJobs();
    const interval = window.setInterval(refreshJobs, 4000);
    return () => {
      window.clearInterval(interval);
    };
  }, [refreshJobs]);

  useEffect(() => {
    jobs.forEach((job) => {
      if (job.status === "completed" && job.result && !processedJobIdsRef.current.has(job.id)) {
        if (job.type === "image") {
          const generatedImages: GeneratedImage[] = (job.result.images || []).map((img: any) => ({
            id: img.id,
            seed: img.seed,
            base64: img.base64 || "",
            model: img.model,
            sampler: img.sampler,
            steps: img.steps,
            cfg_scale: img.cfg_scale ?? img.cfgScale,
            resolution: img.resolution,
            prompt: img.prompt,
            negative_prompt: img.negative_prompt ?? img.negativePrompt,
            clip_skip: img.clip_skip ?? img.clipSkip,
            loras: img.loras || [],
          }));
          if (generatedImages.length > 0) {
            setImages((prev) => {
              const updated = [...generatedImages, ...prev];
              saveImages(updated);
              return updated;
            });
          }
          fetchHistory();
        } else if (job.type === "video") {
          const videoPayload = job.result.video || {};
          const newVideo: GeneratedVideo = {
            id: videoPayload.id || job.id,
            mp4Base64: videoPayload.mp4_base64 || videoPayload.mp4Base64 || "",
            durationSeconds: job.result.duration_seconds ?? job.result.durationSeconds,
          };
          if (newVideo.mp4Base64) {
            setVideos((prev) => [newVideo, ...prev]);
          }
        }
        processedJobIdsRef.current.add(job.id);
      }
    });
  }, [jobs, fetchHistory, saveImages]);

  const sendJobCommand = useCallback(
    async (jobId: string, action: "pause" | "resume" | "start" | "cancel") => {
      try {
        const response = await fetch(`${apiBase}/jobs/${jobId}/${action}`, {
          method: "POST",
        });
        const payload = await response.json().catch(() => ({}));
        if (!response.ok) {
          throw new Error(payload.detail ?? "Action impossible sur ce job.");
        }
        await refreshJobs();
        toast.success(`Job ${action === "pause" ? "mis en pause" : action === "resume" ? "repris" : action === "start" ? "démarré" : "annulé"} avec succès`);
      } catch (err) {
        toast.error(err instanceof Error ? err.message : "Commande job impossible.");
      }
    },
    [apiBase, refreshJobs],
  );

  const handlePauseJob = useCallback((jobId: string) => sendJobCommand(jobId, "pause"), [sendJobCommand]);
  const handleResumeJob = useCallback((jobId: string) => sendJobCommand(jobId, "resume"), [sendJobCommand]);
  const handleStartJob = useCallback((jobId: string) => sendJobCommand(jobId, "start"), [sendJobCommand]);
  const handleCancelJob = useCallback((jobId: string) => sendJobCommand(jobId, "cancel"), [sendJobCommand]);
  const handleDeleteJob = useCallback(
    async (jobId: string) => {
      try {
        const response = await fetch(`${apiBase}/jobs/${jobId}`, { method: "DELETE" });
        const payload = await response.json().catch(() => ({}));
        if (!response.ok) {
          throw new Error(payload.detail ?? "Impossible de supprimer ce job.");
        }
        await refreshJobs();
        toast.success("Job supprimé avec succès");
      } catch (err) {
        toast.error(err instanceof Error ? err.message : "Suppression de job impossible.");
      }
    },
    [apiBase, refreshJobs],
  );

  const handleClearCompleted = useCallback(
    async () => {
      try {
        const response = await fetch(`${apiBase}/jobs/clear-completed`, { method: "POST" });
        const payload = await response.json().catch(() => ({}));
        if (!response.ok) {
          throw new Error(payload.detail ?? "Impossible de nettoyer les jobs complétés.");
        }
        const removed = payload.removed ?? 0;
        await refreshJobs();
        if (removed > 0) {
          toast.success(`${removed} job${removed > 1 ? "s" : ""} complété${removed > 1 ? "s" : ""} supprimé${removed > 1 ? "s" : ""}`);
        } else {
          toast.info("Aucun job complété à supprimer");
        }
      } catch (err) {
        toast.error(err instanceof Error ? err.message : "Impossible de nettoyer les jobs complétés.");
      }
    },
    [apiBase, refreshJobs],
  );

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

  const saveCustomPresetList = useCallback((presets: PromptPreset[]) => {
    setCustomPresets(presets);
    try {
      localStorage.setItem(CUSTOM_PRESET_STORAGE_KEY, JSON.stringify(presets));
    } catch (err) {
      console.warn("Impossible de sauvegarder les presets personnalisés", err);
    }
  }, []);

  useEffect(() => {
    try {
      const stored = localStorage.getItem(CUSTOM_PRESET_STORAGE_KEY);
      if (stored) {
        const parsed = JSON.parse(stored) as PromptPreset[];
        const normalized = parsed.map((preset) => ({
          ...preset,
          id: preset.id ?? `custom-${crypto.randomUUID()}`,
        }));
        setCustomPresets(normalized);
      }
    } catch (err) {
      console.warn("Impossible de charger les presets personnalisés", err);
    }
  }, []);

  const loraOptionMap = useMemo(() => {
    return new Map(availableLoras.map((lora) => [lora.key, lora]));
  }, [availableLoras]);

  const allPresets = useMemo(() => {
    return [...promptPresets, ...customPresets];
  }, [customPresets]);

  const filteredPromptPresets = useMemo(
    () => allPresets.filter((preset) => showNSFW || !preset.nsfw),
    [allPresets, showNSFW],
  );

  const filteredLoras = useMemo(
    () => availableLoras.filter((lora) => showNSFW || !lora.nsfw),
    [availableLoras, showNSFW],
  );

  const customPresetIds = useMemo(() => {
    return new Set(
      customPresets
        .map((preset) => preset.id)
        .filter((id): id is string => typeof id === "string"),
    );
  }, [customPresets]);

  const orderedJobs = useMemo(() => {
    const orderMap: Record<string, number> = {
      running: 0,
      pending: 1,
      paused: 2,
      failed: 3,
      completed: 4,
      cancelled: 5,
    };
    return [...jobs].sort((a, b) => {
      const orderDiff = (orderMap[a.status] ?? 9) - (orderMap[b.status] ?? 9);
      if (orderDiff !== 0) {
        return orderDiff;
      }
      return (b.created_at ?? 0) - (a.created_at ?? 0);
    });
  }, [jobs]);

  const hasRunningJob = useMemo(() => jobs.some((job) => job.status === "running"), [jobs]);
  const runningJob = useMemo(() => jobs.find((job) => job.status === "running") ?? null, [jobs]);
  const pendingJob = useMemo(() => jobs.find((job) => job.status === "pending") ?? null, [jobs]);
  const displayJob = runningJob ?? pendingJob ?? null;
  const jobStatusLabels: Record<string, string> = {
    running: "Génération en cours…",
    pending: "En attente dans la file…",
    paused: "Job en pause",
    failed: "Job en erreur",
    cancelled: "Job annulé",
    completed: "Job terminé",
  };
  const displayJobLabel = displayJob
    ? displayJob.metadata?.prompt_preview ?? `Job ${displayJob.id}`
    : null;
  const displayJobProgressPercent =
    displayJob?.progress && displayJob.progress.total
      ? Math.min(
          100,
          Math.round((displayJob.progress.current / displayJob.progress.total) * 100),
        )
      : null;
  const displayJobStatusText = displayJob
    ? displayJob.progress?.message ?? jobStatusLabels[displayJob.status] ?? ""
    : isGenerating
      ? "Création du job…"
      : null;

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

  const handleSelectJob = useCallback((jobId: string) => {
    setSelectedJobId(jobId);
  }, []);

  const models = useMemo(() => {
    return [
      {
        label: "SDXL Base 1.0",
        value: "sdxl" as ModelKey,
        note: "Officiel · top qualité",
        available: availableModels.has("sdxl"),
        highlight: "Choix recommandé",
        nsfw: false,
      },
      {
        label: "CyberRealistic Pony",
        value: "cyberrealistic-pony" as ModelKey,
        note: "CyberRealistic / style pony",
        available: availableModels.has("cyberrealistic-pony"),
        nsfw: false,
      },
      {
        label: "Tsunade iL",
        value: "tsunade-il" as ModelKey,
        note: "Style Tsunade iL",
        available: availableModels.has("tsunade-il"),
        nsfw: false,
      },
      {
        label: "Wai Illustrious SDXL v1.4",
        value: "wai-illustrious-sdxl" as ModelKey,
        note: "Base SDXL illustrée",
        available: availableModels.has("wai-illustrious-sdxl"),
        nsfw: false,
      },
      {
        label: "wan22 Enhanced NSFW Camera",
        value: "wan22-enhanced-nsfw-camera" as ModelKey,
        note: "wan22Enhanced NSFW Camera Prompt",
        available: availableModels.has("wan22-enhanced-nsfw-camera"),
        nsfw: true,
      },
      {
        label: "Hassaku XL Illustrious v3.2",
        value: "hassaku-xl-illustrious-v32" as ModelKey,
        note: "Hassaku XL Illustrious v3.2",
        available: availableModels.has("hassaku-xl-illustrious-v32"),
        nsfw: false,
      },
      {
        label: "DucHaiten Pony XL (no-score)",
        value: "duchaiten-pony-xl" as ModelKey,
        note: "Checkpoint pony-no-score v4.0",
        available: availableModels.has("duchaiten-pony-xl"),
        nsfw: true,
      },
      {
        label: "LucentXL Pony (Klaabu)",
        value: "lucentxl-pony" as ModelKey,
        note: "LucentXL pony stylisé",
        available: availableModels.has("lucentxl-pony"),
        nsfw: false,
      },
      {
        label: "Pony Diffusion V6 XL",
        value: "ponydiffusion-v6-xl" as ModelKey,
        note: "Pony Diffusion V6 XL Start",
        available: availableModels.has("ponydiffusion-v6-xl"),
        nsfw: false,
      },
      {
        label: "Ishtar's Gate (NSFW/SFW)",
        value: "ishtars-gate-nsfw-sfw" as ModelKey,
        note: "Ishtar's Gate mix NSFW/SFW",
        available: availableModels.has("ishtars-gate-nsfw-sfw"),
        nsfw: true,
      },
    ];
  }, [availableModels]);

  const filteredModels = useMemo(
    () => models.filter((modelOption) => showNSFW || !modelOption.nsfw),
    [models, showNSFW],
  );

  const selectedModelLabel = useMemo(() => {
    const current = models.find((item) => item.value === model);
    return current?.label ?? "modèle sélectionné";
  }, [models, model]);

  const galleryModels = useMemo(
    () => models.map(({ label, value }) => ({ label, value })),
    [models],
  );

  const targetResolution = useMemo(() => {
    if (useAspectRatio) {
      return { width: customWidth, height: customHeight };
    }
    const parts = resolution.split("x").map((part) => Number(part.trim()));
    const width = Number.isFinite(parts[0]) && parts[0] > 0 ? parts[0] : 512;
    const height = Number.isFinite(parts[1]) && parts[1] > 0 ? parts[1] : 512;
    return { width, height };
  }, [useAspectRatio, customWidth, customHeight, resolution]);

  const vramEstimate: VramEstimate | null = useMemo(() => {
    if (mode === "chat") {
      return null;
    }
    try {
      return estimateVramUsage({
        mode,
        resolution,
        useCustomResolution: useAspectRatio,
        width: targetResolution.width,
        height: targetResolution.height,
        steps,
        imageCount,
        model,
        videoMode,
        numFrames,
        fps,
        activeLoras: selectedLoras.length,
        hasImageInit: Boolean(imageInitBase64),
      });
    } catch {
      return null;
    }
  }, [
    mode,
    resolution,
    useAspectRatio,
    targetResolution.width,
    targetResolution.height,
    steps,
    imageCount,
    model,
    videoMode,
    numFrames,
    fps,
    selectedLoras.length,
  ]);

  useEffect(() => {
    if (!showNSFW) {
      const currentModel = models.find((item) => item.value === model);
      if (currentModel?.nsfw) {
        const fallback =
          models.find((item) => !item.nsfw && item.available) ?? models.find((item) => !item.nsfw);
        if (fallback) {
          setModel(fallback.value);
        }
      }

      setSelectedLoras((prev) =>
        prev.filter((selected) => {
          const meta = loraOptionMap.get(selected.key);
          return !meta?.nsfw;
        }),
      );
    }
  }, [showNSFW, models, model, loraOptionMap]);

  const handleGenerate = async () => {
    if (!prompt.trim()) {
      toast.error("Ajoutez un prompt pour lancer la génération.");
      return;
    }
    if (mode === "video" && videoMode === "img2vid" && !initImageBase64) {
      toast.error("Veuillez fournir une image de départ ou basculer en mode Texte → Vidéo.");
      return;
    }
    setIsGenerating(true);

    try {
      const endpoint = mode === "image" ? "generate" : "generate-video";
      const baseResolution = useAspectRatio ? `${customWidth}x${customHeight}` : resolution;
      const numericSeed = Number(seed) || -1;
      const payload =
        mode === "image"
          ? {
              prompt,
              negative_prompt: negativePrompt,
              sampler,
              steps,
              cfg_scale: cfgScale,
              resolution: baseResolution,
              seed: numericSeed,
              clip_skip: clipSkip,
              image_count: imageCount,
              model,
              ...(useAspectRatio && { width: customWidth, height: customHeight }),
              additional_loras: selectedLoras,
              init_image_base64: imageInitBase64,
              ...(imageInitBase64 ? { init_strength: imageInitStrength } : {}),
            }
          : {
              prompt,
              negative_prompt: negativePrompt,
              num_frames: numFrames,
              fps,
              resolution: baseResolution,
              seed: numericSeed,
              mode: videoMode,
              init_image_base64: videoMode === "img2vid" ? initImageBase64 : null,
              image_settings:
                videoMode === "text2vid"
                  ? {
                      prompt,
                      negative_prompt: negativePrompt,
                      sampler,
                      steps,
                      cfg_scale: cfgScale,
                      resolution: baseResolution,
                      seed: numericSeed,
                      clip_skip: clipSkip,
                      image_count: 1,
                      model,
                      ...(useAspectRatio && { width: customWidth, height: customHeight }),
                      additional_loras: selectedLoras,
                    }
                  : undefined,
            };

      const response = await fetch(`${apiBase}/${endpoint}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      const data = await response.json().catch(() => ({}));
      if (!response.ok) {
        throw new Error(data.detail ?? "Erreur inconnue côté backend.");
      }

      if (typeof data.job_id === "string") {
        setSelectedJobId(data.job_id);
        toast.success("Job créé avec succès !");
      }
      await refreshJobs();
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Impossible de créer le job.");
    } finally {
      setIsGenerating(false);
    }
  };

  const handleLoadPreset = (preset: PromptPreset) => {
    setActivePresetId(preset.id ?? null);
    setPrompt(preset.prompt);
    setNegativePrompt(preset.negativePrompt);

    if (preset.sampler) {
      setSampler(preset.sampler);
    }
    if (preset.resolution) {
      setUseAspectRatio(false);
      setResolution(preset.resolution);
    }
    if (typeof preset.steps === "number") {
      setSteps(preset.steps);
    }
    if (typeof preset.cfgScale === "number") {
      setCfgScale(preset.cfgScale);
    }
    if (typeof preset.clipSkip === "number") {
      setClipSkip(preset.clipSkip);
    }
    if (typeof preset.seed === "string") {
      setSeed(preset.seed);
    }

    if (preset.model) {
      const targetModel = models.find((item) => item.value === preset.model);
      if (targetModel && (showNSFW || !targetModel.nsfw) && targetModel.available) {
        setModel(targetModel.value);
      }
    }

    if (preset.loras) {
      const allowedLoras = preset.loras
        .map((lora) => {
          const meta = loraOptionMap.get(lora.key);
          if (!meta) {
            return null;
          }
          if (!showNSFW && meta.nsfw) {
            return null;
          }
          return {
            key: lora.key,
            weight: lora.weight ?? meta.defaultWeight ?? 0.5,
          };
        })
        .filter((item): item is SelectedLora => Boolean(item));
      setSelectedLoras(allowedLoras);
    } else {
      setSelectedLoras([]);
    }
  };

  const buildPresetFromState = (base?: PromptPreset): PromptPreset => ({
    id: base?.id ?? `custom-${crypto.randomUUID()}`,
    name: base?.name ?? "Preset personnalisé",
    description: base?.description ?? "Preset sauvegardé depuis l'interface",
    prompt,
    negativePrompt,
    sampler,
    resolution,
    steps,
    cfgScale,
    seed,
    clipSkip,
    nsfw: showNSFW,
    model,
    loras: selectedLoras.map((item) => ({ ...item })),
  });

  const handleAddPreset = () => {
    setNewPresetName("");
    setNewPresetDesc("");
    setShowNamePrompt(true);
  };

  const handleNameConfirm = (name: string) => {
    if (!name || !name.trim()) {
      toast.error("Le nom du preset ne peut pas être vide");
      return;
    }
    setNewPresetName(name.trim());
    setShowNamePrompt(false);
    setShowDescPrompt(true);
  };

  const handleDescConfirm = (description: string) => {
    const desc = description.trim() || "Preset sauvegardé depuis l'interface";
    const newPreset = {
      ...buildPresetFromState(),
      id: `custom-${crypto.randomUUID()}`,
      name: newPresetName,
      description: desc,
    };
    const updated = [...customPresets, newPreset];
    saveCustomPresetList(updated);
    setActivePresetId(newPreset.id ?? null);
    setShowDescPrompt(false);
    toast.success("Preset ajouté avec succès");
  };

  const handleSavePreset = () => {
    if (activePresetId) {
      const index = customPresets.findIndex((preset) => preset.id === activePresetId);
      if (index !== -1) {
        const updated = [...customPresets];
        updated[index] = {
          ...buildPresetFromState(customPresets[index]),
          id: activePresetId,
          name: customPresets[index].name,
          description: customPresets[index].description,
        };
        saveCustomPresetList(updated);
        toast.success("Preset enregistré avec succès");
        return;
      }
    }
    handleAddPreset();
  };

  const handleDeletePreset = (presetId: string) => {
    const preset = customPresets.find((p) => p.id === presetId);
    if (!preset) {
      return;
    }
    setPresetToDelete(presetId);
    setShowDeleteConfirm(true);
  };

  const confirmDeletePreset = () => {
    if (!presetToDelete) return;
    const updated = customPresets.filter((p) => p.id !== presetToDelete);
    saveCustomPresetList(updated);
    if (activePresetId === presetToDelete) {
      setActivePresetId(null);
    }
    setShowDeleteConfirm(false);
    setPresetToDelete(null);
    toast.success("Preset supprimé avec succès");
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

  const handleDeleteHistory = async (id: string) => {
    try {
      await fetch(`${apiBase}/storage/history/${id}`, { method: "DELETE" });
    } catch (err) {
      console.warn("Impossible de supprimer l'entrée d'historique côté backend", err);
    }
    setHistory((prev) => {
      const updated = prev.filter((entry) => entry.id !== id);
      saveHistory(updated);
      return updated;
    });
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

  const handleDeleteImage = async (id: string) => {
    // Supprimer côté backend
    try {
      await fetch(`${apiBase}/storage/image/${id}`, { method: "DELETE" });
    } catch (err) {
      console.warn("Impossible de supprimer l'image côté backend", err);
    }
    
    // Supprimer côté frontend
    setImages((prev) => {
      const updated = prev.filter((img) => img.id !== id);
      saveImages(updated);
      return updated;
    });
    toast.success("Image supprimée");
  };

  const handleDeleteVideo = async (id: string) => {
    // Supprimer côté backend
    try {
      await fetch(`${apiBase}/storage/video/${id}`, { method: "DELETE" });
    } catch (err) {
      console.warn("Impossible de supprimer la vidéo côté backend", err);
    }
    
    // Supprimer côté frontend
    setVideos((prev) => prev.filter((vid) => vid.id !== id));
    toast.success("Vidéo supprimée");
  };

  const handleCopyVideo = async (video: GeneratedVideo) => {
    try {
      await navigator.clipboard.writeText(video.mp4Base64);
      toast.success("Base64 de la vidéo copié dans le presse-papiers");
    } catch (err) {
      toast.error("Impossible de copier la vidéo");
    }
  };

  const handleExportVideo = (video: GeneratedVideo) => {
    try {
      const byteCharacters = atob(video.mp4Base64);
      const byteNumbers = new Array(byteCharacters.length);
      for (let i = 0; i < byteCharacters.length; i++) {
        byteNumbers[i] = byteCharacters.charCodeAt(i);
      }
      const byteArray = new Uint8Array(byteNumbers);
      const blob = new Blob([byteArray], { type: "video/mp4" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `video-${video.id}.mp4`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
      toast.success("Vidéo téléchargée");
    } catch (err) {
      toast.error("Impossible de télécharger la vidéo");
    }
  };

  const handleSendMessage = async () => {
    if (!chatInput.trim() || isChatting) return;

    const userEntry: ChatEntry = {
      role: "user",
      content: chatInput.trim(),
      ...(chatAttachment ? { images: [chatAttachment] } : {}),
    };
    const updatedMessages = [...chatMessages, userEntry];

    setChatInput("");
    setChatMessages(updatedMessages);
    setIsChatting(true);
    setChatAttachment(null);
    setChatAttachmentName(null);

    try {
      const response = await fetch(`${apiBase}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          messages: updatedMessages.map((msg) => ({
            role: msg.role,
            content: msg.content,
            ...(msg.images ? { images: msg.images } : {}),
          })),
        }),
      });

      if (!response.ok) {
        throw new Error("Erreur lors de la communication avec Ollama");
      }

      const data = await response.json();
      setChatMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: data.message.content,
          ...(data.message.images ? { images: data.message.images } : {}),
        },
      ]);
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Erreur de chat");
      setChatMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: "Désolé, une erreur s'est produite. Vérifiez que Ollama est démarré.",
        },
      ]);
    } finally {
      setIsChatting(false);
    }
  };

  const handleChatReset = () => {
    setChatMessages([]);
    setChatInput("");
    setChatAttachment(null);
    setChatAttachmentName(null);
  };

  const handleInitImageUpload = (file: File | null) => {
    if (!file) {
      setInitImageBase64(null);
      return;
    }

    return new Promise<void>((resolve) => {
      const reader = new FileReader();
      reader.onload = () => {
        const result = reader.result;
        if (typeof result === "string") {
          const base64 = result.split(",")[1] ?? result;
          setInitImageBase64(base64);
        }
        resolve();
      };
      reader.onerror = () => resolve();
      reader.readAsDataURL(file);
    });
  };

  const handleImageInitUpload = (file: File | null) => {
    if (!file) {
      setImageInitBase64(null);
      setImageInitName(null);
      return;
    }
    if (file.size > 4 * 1024 * 1024) {
      toast.error("Veuillez choisir une image inférieure à 4 Mo.");
      return;
    }
    return new Promise<void>((resolve) => {
      const reader = new FileReader();
      reader.onload = () => {
        const result = reader.result;
        if (typeof result === "string") {
          const base64 = result.includes(",") ? result.split(",")[1] ?? result : result;
          setImageInitBase64(base64);
          setImageInitName(file.name);
        }
        resolve();
      };
      reader.onerror = () => resolve();
      reader.readAsDataURL(file);
    });
  };

  const handleImageInitClear = () => {
    setImageInitBase64(null);
    setImageInitName(null);
  };

  const handleChatImageUpload = (file: File | null) => {
    if (!file) {
      setChatAttachment(null);
      setChatAttachmentName(null);
      return;
    }

    if (file.size > 4 * 1024 * 1024) {
      toast.error("Veuillez choisir une image inférieure à 4 Mo.");
      return;
    }

    return new Promise<void>((resolve) => {
      const reader = new FileReader();
      reader.onload = () => {
        const result = reader.result;
        if (typeof result === "string") {
          const base64 = result.includes(",") ? result.split(",")[1] ?? result : result;
          setChatAttachment(base64);
          setChatAttachmentName(file.name);
        }
        resolve();
      };
      reader.onerror = () => resolve();
      reader.readAsDataURL(file);
    });
  };

  const handleChatAttachmentClear = () => {
    setChatAttachment(null);
    setChatAttachmentName(null);
  };

  const handleVideoModeChange = useCallback((value: VideoMode) => {
    setVideoMode(value);
    if (value === "text2vid") {
      setInitImageBase64(null);
    }
  }, []);

  const handleUseAspectRatioChange = (enabled: boolean) => {
    setUseAspectRatio(enabled);
    if (enabled) {
      const selected = aspectRatios.find((r) => r.value === aspectRatio);
      if (selected) {
        setCustomWidth(selected.width);
        setCustomHeight(selected.height);
      }
    }
  };

  const handleAspectRatioChange = (value: string) => {
    setAspectRatio(value);
    const selected = aspectRatios.find((r) => r.value === value);
    if (selected) {
      setCustomWidth(selected.width);
      setCustomHeight(selected.height);
    }
  };

  const handleCustomWidthChange = (value: number) => {
    setCustomWidth(value);
    if (useAspectRatio) {
      const selected = aspectRatios.find((r) => r.value === aspectRatio);
      if (selected) {
        const ratio = selected.height / selected.width;
        setCustomHeight(Math.round(value * ratio));
      }
    }
  };

  const handleCustomHeightChange = (value: number) => {
    setCustomHeight(value);
    if (useAspectRatio) {
      const selected = aspectRatios.find((r) => r.value === aspectRatio);
      if (selected) {
        const ratio = selected.width / selected.height;
        setCustomWidth(Math.round(value * ratio));
      }
    }
  };

  const clampFrames = (frames: number) => Math.max(6, Math.min(16, frames));

  const handleVideoDurationChange = (duration: number) => {
    setVideoDuration(duration);
    setNumFrames(clampFrames(Math.round(duration * fps)));
  };

  const handleFpsChange = (value: number) => {
    setFps(value);
    setNumFrames(clampFrames(Math.round(videoDuration * value)));
  };

  const handleNumFramesChange = (value: number) => {
    setNumFrames(value);
    setVideoDuration(value / fps);
  };

  const handleImageCountChange = (value: number) => {
    setImageCount(Math.min(4, Math.max(1, value)));
  };

  const handleClearGallery = () => {
    setImages([]);
    saveImages([]);
  };


  return (
    <div className="min-h-screen bg-slate-950 bg-[radial-gradient(circle_at_top,_#1e293b,_#020617)] text-slate-100">
      <main className="flex w-full gap-6 px-4 py-10 lg:px-8">
        <HistorySidebar history={history} onSelect={handleLoadHistory} onDelete={handleDeleteHistory} />

        <section className="flex-1 space-y-7">
          <GeneratorHeader
            mode={mode}
            onModeChange={setMode}
            showNSFW={showNSFW}
            onToggleNSFW={setShowNSFW}
          />

          <div className="grid gap-6 lg:grid-cols-[minmax(0,1.2fr)_minmax(0,0.8fr)]">
            <PromptSettingsPanel
              mode={mode}
              apiBase={apiBase}
              prompt={prompt}
              negativePrompt={negativePrompt}
              sampler={sampler}
              resolution={resolution}
              useAspectRatio={useAspectRatio}
              aspectRatio={aspectRatio}
              customWidth={customWidth}
              customHeight={customHeight}
              steps={steps}
              cfgScale={cfgScale}
              stepsHint={stepsHint}
              cfgHint={cfgHint}
              videoDuration={videoDuration}
              fps={fps}
              numFrames={numFrames}
              chatMessages={chatMessages}
              chatInput={chatInput}
              isChatting={isChatting}
              chatAttachmentPreview={chatAttachment}
              chatAttachmentName={chatAttachmentName}
              imageInitPreview={
                imageInitBase64 ? `data:image/png;base64,${imageInitBase64}` : null
              }
              imageInitName={imageInitName}
              imageInitStrength={imageInitStrength}
              promptPresets={filteredPromptPresets}
              samplers={samplers}
              resolutions={resolutions}
              aspectRatios={aspectRatios}
              onPromptChange={setPrompt}
              onNegativePromptChange={setNegativePrompt}
              onSamplerChange={(value) => setSampler(value)}
              onResolutionChange={(value) => setResolution(value)}
              onUseAspectRatioChange={handleUseAspectRatioChange}
              onAspectRatioChange={handleAspectRatioChange}
              onCustomWidthChange={handleCustomWidthChange}
              onCustomHeightChange={handleCustomHeightChange}
              onStepsChange={setSteps}
              onCfgScaleChange={setCfgScale}
              onPresetApply={handleLoadPreset}
              onVideoDurationChange={handleVideoDurationChange}
              onFpsChange={handleFpsChange}
              onNumFramesChange={handleNumFramesChange}
              videoMode={videoMode}
              onVideoModeChange={handleVideoModeChange}
              selectedModelLabel={selectedModelLabel}
              onInitImageUpload={handleInitImageUpload}
              onChatInputChange={setChatInput}
              onSendMessage={handleSendMessage}
              onChatReset={handleChatReset}
              onChatImageUpload={handleChatImageUpload}
              onChatAttachmentClear={handleChatAttachmentClear}
              onImageInitUpload={handleImageInitUpload}
              onImageInitClear={handleImageInitClear}
              onImageInitStrengthChange={setImageInitStrength}
              onAddPreset={handleAddPreset}
              onSavePreset={handleSavePreset}
              onDeletePreset={handleDeletePreset}
              activePresetId={activePresetId}
              customPresetIds={customPresetIds}
            />

            <ModelSettingsPanel
              mode={mode}
              models={filteredModels}
              model={model}
              seed={seed}
              clipSkip={clipSkip}
              selectedLoras={selectedLoras}
              availableLoras={filteredLoras}
              isSubmitting={isGenerating}
              jobLabel={displayJobLabel}
              jobStatusText={displayJobStatusText}
              jobProgressPercent={displayJobProgressPercent}
              imageCount={imageCount}
            vramEstimate={vramEstimate}
              onModelChange={setModel}
              onSeedChange={setSeed}
              onClipSkipChange={setClipSkip}
              onToggleLora={handleToggleLora}
              onLoraWeightChange={handleLoraWeightChange}
              onClearLoras={handleClearLoras}
              onGenerate={handleGenerate}
              onImageCountChange={handleImageCountChange}
            />
          </div>

        <JobQueuePanel
          jobs={orderedJobs}
          selectedJobId={selectedJobId}
          hasRunningJob={hasRunningJob}
          jobsError={jobsError}
          onSelectJob={handleSelectJob}
          onPause={handlePauseJob}
          onResume={handleResumeJob}
          onStart={handleStartJob}
          onCancel={handleCancelJob}
          onDelete={handleDeleteJob}
          onClearCompleted={handleClearCompleted}
        />

          <GallerySection
            images={images}
            videos={videos}
            models={galleryModels}
            samplers={samplers}
            onDeleteImage={handleDeleteImage}
            onCopyImage={handleCopyBase64}
            onExportImage={handleExportImage}
            onClearImages={handleClearGallery}
            onDeleteVideo={handleDeleteVideo}
            onCopyVideo={handleCopyVideo}
            onExportVideo={handleExportVideo}
          />
        </section>
      </main>

      <PromptModal
        isOpen={showNamePrompt}
        onClose={() => setShowNamePrompt(false)}
        title="Nom du preset"
        placeholder="Preset personnalisé"
        defaultValue="Preset personnalisé"
        onConfirm={handleNameConfirm}
        confirmLabel="Suivant"
        cancelLabel="Annuler"
      />

      <PromptModal
        isOpen={showDescPrompt}
        onClose={() => setShowDescPrompt(false)}
        title="Description du preset (optionnel)"
        placeholder="Preset sauvegardé depuis l'interface"
        defaultValue="Preset sauvegardé depuis l'interface"
        onConfirm={handleDescConfirm}
        confirmLabel="Créer"
        cancelLabel="Annuler"
      />

      <Modal
        isOpen={showDeleteConfirm}
        onClose={() => {
          setShowDeleteConfirm(false);
          setPresetToDelete(null);
        }}
        title="Supprimer le preset"
        onConfirm={confirmDeletePreset}
        confirmLabel="Supprimer"
        cancelLabel="Annuler"
      >
        <p>
          Êtes-vous sûr de vouloir supprimer le preset{" "}
          <strong>
            "{customPresets.find((p) => p.id === presetToDelete)?.name}"
          </strong>
          ? Cette action est irréversible.
        </p>
      </Modal>
    </div>
  );
}

