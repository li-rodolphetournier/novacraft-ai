export type Sampler = "euler" | "dpmpp_2s" | "unipc" | "ddim";

export type Resolution = "512x512" | "768x768" | "1024x1024" | "1536x1536";

export type ModelKey =
  | "realistic-vision"
  | "dreamshaper"
  | "meinamik"
  | "sdxl"
  | "cyberrealistic-pony"
  | "tsunade-il"
  | "wai-illustrious-sdxl"
  | "wan22-enhanced-nsfw-camera"
  | "hassaku-xl-illustrious-v32"
  | "duchaiten-pony-xl"
  | "lucentxl-pony"
  | "ponydiffusion-v6-xl"
  | "ishtars-gate-nsfw-sfw";

export type Mode = "image" | "video" | "chat";

export type SelectedLora = {
  key: string;
  weight: number;
};

export type LoraOption = {
  key: string;
  label: string;
  description?: string;
  defaultWeight: number;
  nsfw?: boolean;
};

export type AppliedLora = {
  key: string;
  label: string;
  weight: number;
  type?: string;
};

export type HistoryEntry = {
  id: string;
  prompt: string;
  negativePrompt: string;
  model: ModelKey;
  timestamp: number | string;
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

export type GeneratedImage = {
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

export type GenerateResponse = {
  images: GeneratedImage[];
  duration_seconds?: number;
};

export type GeneratedVideo = {
  id: string;
  mp4Base64: string;
  durationSeconds?: number;
};

export type PromptPreset = {
  name: string;
  prompt: string;
  negativePrompt: string;
  description: string;
  sampler?: Sampler;
  resolution?: Resolution;
  steps?: number;
  cfgScale?: number;
  seed?: string;
  clipSkip?: number;
  nsfw?: boolean;
  model?: ModelKey;
  loras?: SelectedLora[];
};

