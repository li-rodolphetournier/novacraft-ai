const DEFAULT_VRAM_LIMIT_GB = Number(process.env.NEXT_PUBLIC_VRAM_LIMIT_GB ?? "8");

export type VramEstimateInput = {
  mode: "image" | "video" | "chat";
  resolution: string;
  useCustomResolution: boolean;
  width: number;
  height: number;
  steps: number;
  imageCount: number;
  model: string;
  videoMode: "img2vid" | "text2vid";
  numFrames: number;
  fps: number;
  activeLoras: number;
  vramLimitGb?: number;
};

export type VramEstimate = {
  usageGb: number;
  limitGb: number;
  percent: number;
  level: "safe" | "warning" | "danger";
  message: string;
};

const BASE_USAGE_SD15 = 1.3; // ~1.3 GB @512², 30 steps, 1 image
const BASE_USAGE_SDXL = 2.6; // ~2.6 GB @512², 30 steps, 1 image

const parseResolution = (value: string): { width: number; height: number } => {
  const normalized = value.toLowerCase();
  if (normalized.includes("x")) {
    const [w, h] = normalized.split("x").map((part) => Number(part.trim()));
    if (Number.isFinite(w) && Number.isFinite(h) && w > 0 && h > 0) {
      return { width: w, height: h };
    }
  }
  return { width: 512, height: 512 };
};

export const estimateVramUsage = (input: VramEstimateInput): VramEstimate => {
  const limitGb = input.vramLimitGb ?? DEFAULT_VRAM_LIMIT_GB;
  const resolution = input.useCustomResolution
    ? { width: input.width, height: input.height }
    : parseResolution(input.resolution);

  const pixelFactor = (resolution.width * resolution.height) / (512 * 512);
  const base =
    input.model?.includes("xl") || input.model?.includes("sdxl")
      ? BASE_USAGE_SDXL
      : BASE_USAGE_SD15;

  const stepFactor = Math.max(input.steps, 1) / 30;
  const batchFactor = Math.max(input.imageCount, 1);

  let usageGb = base * pixelFactor * stepFactor * batchFactor;

  usageGb += input.activeLoras * 0.2;

  if (input.mode === "video") {
    const frameCost = input.videoMode === "text2vid" ? 0.18 : 0.15;
    usageGb += input.numFrames * frameCost;
    if (input.videoMode === "text2vid") {
      usageGb += 0.8; // image de référence supplémentaire
    }
    usageGb += (Math.max(input.fps, 1) / 6) * 0.25;
  }

  usageGb = Number(usageGb.toFixed(2));
  const percent = Math.min(usageGb / limitGb, 2);

  let level: VramEstimate["level"] = "safe";
  if (percent >= 0.95) {
    level = "danger";
  } else if (percent >= 0.75) {
    level = "warning";
  }

  const message =
    level === "danger"
      ? "Risque élevé d'erreur CUDA. Baissez la résolution ou le batch."
      : level === "warning"
        ? "Attention, configuration gourmande en VRAM."
        : "Configuration confortable.";

  return {
    usageGb,
    limitGb,
    percent,
    level,
    message,
  };
};

