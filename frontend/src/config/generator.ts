import type { PromptPreset, Resolution, Sampler } from "@/types/generator";

export const samplers: { label: string; value: Sampler }[] = [
  { label: "Euler A", value: "euler" },
  { label: "DPM++ 2S", value: "dpmpp_2s" },
  { label: "UniPC", value: "unipc" },
  { label: "DDIM", value: "ddim" },
];

export const resolutions: { label: string; value: Resolution; hint: string }[] = [
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

export type AspectRatioOption = { label: string; value: string; width: number; height: number };

export const aspectRatios: AspectRatioOption[] = [
  { label: "1:1 (Carré)", value: "1:1", width: 1024, height: 1024 },
  { label: "16:9 (Paysage)", value: "16:9", width: 1344, height: 768 },
  { label: "9:16 (Portrait)", value: "9:16", width: 768, height: 1344 },
  { label: "4:3 (Classique)", value: "4:3", width: 1152, height: 896 },
  { label: "3:4 (Portrait)", value: "3:4", width: 896, height: 1152 },
  { label: "21:9 (Ultra large)", value: "21:9", width: 1536, height: 640 },
  { label: "2:3 (Portrait)", value: "2:3", width: 832, height: 1216 },
  { label: "3:2 (Paysage)", value: "3:2", width: 1216, height: 832 },
];

export const promptPresets: PromptPreset[] = [
  {
    id: "portrait-realiste",
    name: "Portrait réaliste",
    prompt: "portrait of a person, professional photography, high quality, detailed face, natural lighting",
    negativePrompt: "blurry, bad anatomy, low quality, distorted hands, watermark, cartoon, anime",
    description: "Portrait photographique de qualité",
    sampler: "euler",
    resolution: "1024x1024",
    steps: 28,
    cfgScale: 6.5,
    seed: "3884499817",
    clipSkip: 2,
    model: "sdxl",
  },
  {
    id: "paysage-fantastique",
    name: "Paysage fantastique",
    prompt: "epic fantasy landscape, mountains, magical atmosphere, cinematic lighting, highly detailed, 4k",
    negativePrompt: "blurry, low quality, distorted, watermark, people",
    description: "Paysage épique et cinématique",
    sampler: "dpmpp_2s",
    resolution: "1536x1536",
    steps: 35,
    cfgScale: 8,
    seed: "1188995532",
    clipSkip: 2,
    model: "sdxl",
  },
  {
    id: "architecture-moderne",
    name: "Architecture moderne",
    prompt: "modern architecture, futuristic building, clean lines, minimalist design, professional photography",
    negativePrompt: "blurry, low quality, old, vintage, distorted",
    description: "Architecture contemporaine",
    sampler: "unipc",
    resolution: "768x768",
    steps: 22,
    cfgScale: 5.5,
    seed: "229944771",
    clipSkip: 1,
    model: "wai-illustrious-sdxl",
  },
  {
    id: "art-conceptuel",
    name: "Art conceptuel",
    prompt: "concept art, digital painting, vibrant colors, detailed, fantasy, artistic style",
    negativePrompt: "blurry, low quality, watermark, realistic photo",
    description: "Style artistique et coloré",
    sampler: "ddim",
    resolution: "1024x1024",
    steps: 30,
    cfgScale: 9,
    seed: "998711223",
    clipSkip: 2,
    model: "sdxl",
  },
  {
    id: "nsfw-1",
    name: "Nfsw",
    prompt:
      "anime style, manga style, A woman is at the beach, she has semen on her mouth that is dripping onto the man's penis. The woman uses her breasts to perform fellatio on the man. She squeezes her breasts together with her fists",
    negativePrompt:
      "score_5, score_4, 3d, render, simple background, zPDXL2, pointy chin, flat chested, cross eyed, sleeves, long hair, blush, fewer digits, lesser digits, missing fingers, extra hands, extra fingers, interracial,",
    description: "Style artistique et coloré",
    sampler: "euler",
    resolution: "1024x1024",
    steps: 30,
    cfgScale: 8.5,
    seed: "420690001",
    clipSkip: 2,
    nsfw: true,
    model: "wan22-enhanced-nsfw-camera",
    loras: [
      { key: "expressive-h", weight: 0.45 },
      { key: "incase-ponyxl", weight: 0.4 },
    ],
  },
  {
    id: "nsfw-2",
    name: "Nfsw2",
    prompt:
      "score_9, score_8_up, score_7_up, score_6_up, Fubuki (One-Punch Man), black leotard, jewelry, natural breasts, anime, hips, cinematic angle, cinematic lighting, volumetric lighting, solo focus, erect nipples, medium breasts, saggy breasts, teardrop breasts, face focus, sweat, sleeveless, mature woman, topless, sexy, seductive, parted lips, determined, looking at viewer, (leaning forward), arched back, 1boy, looking at viewer, (pov:1.5), crotch, (paizuri:1.2),penis between breasts, upper body (close up:1.2), low angle, cum on breasts, cum on face, ejaculation, (mouth open, tongue out), tongue, tongue out, rolling eyes, sunglasses on head, beach, sky, palms, <lora:incase-ilff-v3-4:0.5> <lora:Expressive_H:0.45>",
    negativePrompt:
      "score_5, score_4, 3d, render, simple background, zPDXL2, pointy chin, flat chested, cross eyed, sleeves, long hair, blush, fewer digits, lesser digits, missing fingers, extra hands, extra fingers, interracial,",
    description: "Style artistique et coloré 2",
    sampler: "dpmpp_2s",
    resolution: "768x768",
    steps: 26,
    cfgScale: 7.5,
    seed: "314159265",
    clipSkip: 2,
    nsfw: true,
    model: "duchaiten-pony-xl",
    loras: [
      { key: "expressive-h", weight: 0.45 },
      { key: "incase-ponyxl", weight: 0.5 },
      { key: "g0th1c-pxl", weight: 0.35 },
    ],
  },
  {
    id: "nsfw-3",
    name: "Nfsw3",
    prompt:
      "Photorealistic highly detailed face, highly detailed, realistic, shiny transparent white latex, close-up of her face:1.5, highly-detailed, best quality, masterpiece, very aesthetic, sharp, spectacular wet red hair:2, cum in hair, fake breasts, multiple neon colors:1.5, transparent polished transparent white latex suit:2, glossy skin tight 8k, source_photo, source_real, adorable, supermodel, real life photo, cute and short girl, detailed face and cock and fingers and hair:1.5, absurdres, 1girl 18yo, Cutie:1.5, beautiful face, too big for mouth, fellatio, huge penis, ((cum on face and hair and cock)), horny, blush, (above view:1.5), nude,(adorable),(spectacular curly red hair),, (short black fat cock on face), imminent blowjob, penis on face, ((excessive cum, cum on face, cum on hair, cum on breasts)), extremely fat black cock on cheek, extremely short black cock on face, perfect curved fat black cock:1.5",
    negativePrompt:
      "score_6, score_5, score_4, fcNeg, blur, low quality, worst quality, disfigured, censored, bucktooth, disproportionate body, cartoon, ugly, anime, 2D, burned, sunburned, deformed, dismembered, disembodied, detached, elderly, old, ((butterface, puffy face, puffy cheeks, wide face, ugly face, round face, fat face, chubby face, chubby cheeks)) polydactyl, amputated, contorted, painting, logo, watermark, monochrome, sketch, sketchy, low quality, worst quality, bad quality, drawing, anorexic, frail, malnourished, fat face, wide face, chubby face, round face, scrawny, elderly, wrinkles, jpeg artifacts, plastic doll, animation, claymation, teeth, bushy eyebrows, pink, pink eyeshadow, source_pony, source_anime, source_furry, source_cartoon, brunette, exposed breasts, exposed skin, non-black nipples, blowjob, penis on mouth, open mouth, veiny, veins, veiny cock",
    description: "Style artistique et coloré 3",
    sampler: "euler",
    resolution: "512x512",
    steps: 24,
    cfgScale: 10,
    seed: "777888999",
    clipSkip: 3,
    nsfw: true,
    model: "duchaiten-pony-xl",
    loras: [
      { key: "bs-pony-alpha", weight: 0.6 },
      { key: "ps-alpha", weight: 0.55 },
      { key: "pts-alpha", weight: 0.55 },
    ],
  },
];

