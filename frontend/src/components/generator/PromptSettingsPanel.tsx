"use client";

import { ChangeEvent } from "react";
import { motion } from "framer-motion";
import { FaStar, FaTimes } from "react-icons/fa";
import type {
  Mode,
  PromptPreset,
  Resolution,
  Sampler,
  VideoMode,
} from "@/types/generator";
import type { AspectRatioOption } from "@/config/generator";

type PromptSettingsPanelProps = {
  mode: Mode;
  apiBase: string;
  prompt: string;
  negativePrompt: string;
  sampler: Sampler;
  resolution: Resolution;
  useAspectRatio: boolean;
  aspectRatio: string;
  customWidth: number;
  customHeight: number;
  steps: number;
  cfgScale: number;
  stepsHint: string;
  cfgHint: string;
  videoDuration: number;
  fps: number;
  numFrames: number;
  chatMessages: Array<{ role: "user" | "assistant"; content: string; images?: string[] }>;
  chatInput: string;
  isChatting: boolean;
  chatAttachmentPreview: string | null;
  chatAttachmentName: string | null;
  promptPresets: PromptPreset[];
  samplers: { label: string; value: Sampler }[];
  resolutions: { label: string; value: Resolution; hint: string }[];
  aspectRatios: AspectRatioOption[];
  onPromptChange: (value: string) => void;
  onNegativePromptChange: (value: string) => void;
  onSamplerChange: (value: Sampler) => void;
  onResolutionChange: (value: Resolution) => void;
  onUseAspectRatioChange: (value: boolean) => void;
  onAspectRatioChange: (value: string) => void;
  onCustomWidthChange: (value: number) => void;
  onCustomHeightChange: (value: number) => void;
  onStepsChange: (value: number) => void;
  onCfgScaleChange: (value: number) => void;
  onPresetApply: (preset: PromptPreset) => void;
  onVideoDurationChange: (value: number) => void;
  onFpsChange: (value: number) => void;
  onNumFramesChange: (value: number) => void;
  videoMode: VideoMode;
  onVideoModeChange: (mode: VideoMode) => void;
  onInitImageUpload: (file: File | null) => Promise<void> | void;
  onChatInputChange: (value: string) => void;
  onSendMessage: () => void;
  onChatReset: () => void;
  onChatImageUpload: (file: File | null) => Promise<void> | void;
  onChatAttachmentClear: () => void;
  onAddPreset: () => void;
  onSavePreset: () => void;
  onDeletePreset: (presetId: string) => void;
  activePresetId: string | null;
  customPresetIds: Set<string>;
  selectedModelLabel: string;
};

export function PromptSettingsPanel({
  mode,
  prompt,
  negativePrompt,
  sampler,
  resolution,
  useAspectRatio,
  aspectRatio,
  customWidth,
  customHeight,
  steps,
  cfgScale,
  stepsHint,
  cfgHint,
  videoDuration,
  fps,
  numFrames,
  chatMessages,
  chatInput,
  isChatting,
  chatAttachmentPreview,
  chatAttachmentName,
  apiBase,
  promptPresets,
  samplers,
  resolutions,
  aspectRatios,
  onPromptChange,
  onNegativePromptChange,
  onSamplerChange,
  onResolutionChange,
  onUseAspectRatioChange,
  onAspectRatioChange,
  onCustomWidthChange,
  onCustomHeightChange,
  onStepsChange,
  onCfgScaleChange,
  onPresetApply,
  onVideoDurationChange,
  onFpsChange,
  onNumFramesChange,
  videoMode,
  onVideoModeChange,
  onInitImageUpload,
  onChatInputChange,
  onSendMessage,
  onChatReset,
  onChatImageUpload,
  onChatAttachmentClear,
  onAddPreset,
  onSavePreset,
  onDeletePreset,
  activePresetId,
  customPresetIds,
  selectedModelLabel,
}: PromptSettingsPanelProps) {
  const handleNumericChange =
    (callback: (value: number) => void) =>
    (event: ChangeEvent<HTMLInputElement>) => {
      callback(Number(event.target.value));
    };

  const renderPromptPresets = () => (
    <div className="ml-auto flex flex-col items-end gap-1">
      <div className="grid grid-cols-2 gap-1">
        {promptPresets.map((preset) => {
          const isActive = preset.id && activePresetId === preset.id;
          const isCustom = preset.id ? customPresetIds.has(preset.id) : false;
          return (
            <div key={`${preset.id ?? preset.name}`} className="flex items-center gap-1">
              <button
                onClick={() => onPresetApply(preset)}
                className={`flex-1 rounded-lg border px-2 py-0.5 text-[10px] uppercase tracking-wide transition ${
                  isActive
                    ? "border-indigo-400 bg-indigo-500/20 text-white"
                    : "border-white/10 bg-slate-900/60 text-slate-300 hover:bg-slate-800 hover:text-white"
                }`}
                title={preset.description}
                type="button"
              >
                {preset.name}
                {isCustom && <FaStar className="ml-1 inline text-yellow-400" size={10} />}
              </button>
              {isCustom && preset.id && (
                <motion.button
                  whileHover={{ scale: 1.1 }}
                  whileTap={{ scale: 0.9 }}
                  onClick={(e) => {
                    e.stopPropagation();
                    onDeletePreset(preset.id!);
                  }}
                  className="rounded-lg border border-rose-400/40 bg-rose-500/10 p-1.5 text-rose-200 transition hover:bg-rose-500/20"
                  title="Supprimer ce preset"
                  type="button"
                >
                  <FaTimes size={10} />
                </motion.button>
              )}
            </div>
          );
        })}
      </div>
      <div className="flex gap-2">
        <button
          type="button"
          onClick={onAddPreset}
          className="rounded-lg border border-white/15 bg-slate-900/60 px-2 py-0.5 text-[10px] uppercase tracking-wide text-slate-200 transition hover:bg-slate-800 hover:text-white"
        >
          Ajouter preset
        </button>
        <button
          type="button"
          onClick={onSavePreset}
          className="rounded-lg border border-indigo-400/40 bg-indigo-500/20 px-2 py-0.5 text-[10px] uppercase tracking-wide text-white transition hover:bg-indigo-500/30"
        >
          Enregistrer preset
        </button>
      </div>
    </div>
  );

  const renderPromptControls = () => (
    <>
      <div className="flex items-center mb-1.5">
        <label className="text-xs font-semibold uppercase tracking-[0.2em] text-slate-400">
          Prompt
        </label>
        {renderPromptPresets()}
      </div>
      <textarea
        className="min-h-[70px] w-full rounded-xl border border-white/10 bg-slate-900/60 p-2.5 text-xs text-white outline-none transition focus:border-indigo-400"
        placeholder="Texte principal qui décrit la scène..."
        value={prompt}
        onChange={(event) => onPromptChange(event.target.value)}
      />

      <div>
        <label className="text-xs font-semibold uppercase tracking-[0.2em] text-slate-400">
          Negative prompt
        </label>
        <textarea
          className="mt-1.5 min-h-[50px] w-full rounded-xl border border-white/10 bg-slate-900/60 p-2.5 text-xs text-white outline-none transition focus:border-rose-400"
          placeholder="Ce que vous ne voulez pas voir…"
          value={negativePrompt}
          onChange={(event) => onNegativePromptChange(event.target.value)}
        />
      </div>
    </>
  );

  return (
    <div className="w-full space-y-4 rounded-3xl border border-white/10 bg-white/5 p-4 shadow-2xl">
      {(mode === "image" || mode === "video") && (
        <>
          {renderPromptControls()}

          <div className="grid gap-3 md:grid-cols-2">
            <div>
              <p className="text-xs font-semibold uppercase tracking-[0.2em] text-slate-400">Sampler</p>
              <select
                className="mt-1.5 w-full rounded-xl border border-white/10 bg-slate-900/60 p-2 text-xs text-white"
                value={sampler}
                onChange={(event) => onSamplerChange(event.target.value as Sampler)}
              >
                {samplers.map((option) => (
                  <option key={option.value} value={option.value}>
                    {option.label}
                  </option>
                ))}
              </select>
              <p className="mt-1 text-[10px] text-slate-500">Euler A & DPM++ conseillés.</p>
            </div>

            <div>
              <div className="flex items-center justify-between mb-1.5">
                <p className="text-xs font-semibold uppercase tracking-[0.2em] text-slate-400">Résolution</p>
                <label className="flex items-center gap-1.5 text-[10px] text-slate-400">
                  <input
                    type="checkbox"
                    checked={useAspectRatio}
                    onChange={(e) => onUseAspectRatioChange(e.target.checked)}
                    className="rounded"
                  />
                  Ratio personnalisé
                </label>
              </div>
              {!useAspectRatio ? (
                <>
                  <select
                    className="mt-1.5 w-full rounded-xl border border-white/10 bg-slate-900/60 p-2 text-xs text-white"
                    value={resolution}
                    onChange={(event) => onResolutionChange(event.target.value as Resolution)}
                  >
                    {resolutions.map((option) => (
                      <option key={option.value} value={option.value}>
                        {option.label}
                      </option>
                    ))}
                  </select>
                  <p className="mt-1 text-[10px] text-slate-500">
                    {resolutions.find((item) => item.value === resolution)?.hint}
                  </p>
                </>
              ) : (
                <div className="mt-1.5 space-y-2">
                  <div>
                    <label className="text-[10px] uppercase tracking-wider text-slate-400">Ratio d&apos;aspect</label>
                    <select
                      className="mt-1 w-full rounded-lg border border-white/10 bg-slate-900/60 p-1.5 text-xs text-white"
                      value={aspectRatio}
                      onChange={(event) => onAspectRatioChange(event.target.value)}
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
                      <label className="text-[10px] uppercase tracking-wider text-slate-400">Largeur</label>
                      <input
                        type="number"
                        min={256}
                        max={2048}
                        step={64}
                        value={customWidth}
                        onChange={handleNumericChange(onCustomWidthChange)}
                        className="mt-1 w-full rounded-lg border border-white/10 bg-slate-900/60 px-2 py-1.5 text-xs text-white"
                      />
                    </div>
                    <div>
                      <label className="text-[10px] uppercase tracking-wider text-slate-400">Hauteur</label>
                      <input
                        type="number"
                        min={256}
                        max={2048}
                        step={64}
                        value={customHeight}
                        onChange={handleNumericChange(onCustomHeightChange)}
                        className="mt-1 w-full rounded-lg border border-white/10 bg-slate-900/60 px-2 py-1.5 text-xs text-white"
                      />
                    </div>
                  </div>
                  <p className="text-[9px] text-slate-500">Dimensions personnalisées (multiples de 64)</p>
                </div>
              )}
            </div>
          </div>

          <div className="grid gap-4 md:grid-cols-2">
            <div>
              <div className="flex items-center justify-between text-xs uppercase tracking-[0.2em] text-slate-400">
                <span>Steps</span>
                <span className="text-[10px] text-slate-300">{stepsHint}</span>
              </div>
              <input
                type="range"
                min={10}
                max={60}
                value={steps}
                onChange={handleNumericChange(onStepsChange)}
                className="mt-2 w-full accent-indigo-400"
              />
              <p className="mt-1 text-xs font-semibold text-white">{steps} itérations</p>
            </div>

            <div>
              <div className="flex items-center justify-between text-xs uppercase tracking-[0.2em] text-slate-400">
                <span>CFG Scale</span>
                <span className="text-[10px] text-slate-300">{cfgHint}</span>
              </div>
              <input
                type="range"
                min={1}
                max={14}
                step={0.5}
                value={cfgScale}
                onChange={handleNumericChange(onCfgScaleChange)}
                className="mt-2 w-full accent-indigo-400"
              />
              <p className="mt-1 text-xs font-semibold text-white">Intensité : {cfgScale.toFixed(1)}</p>
            </div>
          </div>

          {mode === "video" && (
            <>
              <div>
                <p className="text-xs font-semibold uppercase tracking-[0.2em] text-slate-400">Mode vidéo</p>
                <div className="mt-1 grid grid-cols-2 gap-2">
                  {[
                    { value: "img2vid", label: "Image → Vidéo" },
                    { value: "text2vid", label: "Texte → Vidéo" },
                  ].map((option) => (
                    <button
                      key={option.value}
                      type="button"
                      onClick={() => onVideoModeChange(option.value as VideoMode)}
                      className={`rounded-xl border px-3 py-1.5 text-[11px] font-semibold transition ${
                        videoMode === option.value
                          ? "border-indigo-400 bg-indigo-500/20 text-white"
                          : "border-white/10 bg-slate-900/60 text-slate-300 hover:border-white/30"
                      }`}
                    >
                      {option.label}
                    </button>
                  ))}
                </div>
                <p className="mt-1 text-[10px] text-slate-500">
                  {videoMode === "img2vid"
                    ? "Utilisez votre propre image comme référence."
                    : `L'image de départ sera générée via ${selectedModelLabel}.`}
                </p>
              </div>

              {videoMode === "img2vid" ? (
                <div>
                  <p className="text-xs font-semibold uppercase tracking-[0.2em] text-slate-400">
                    Image de départ (PNG/JPEG)
                  </p>
                  <input
                    type="file"
                    accept="image/*"
                    className="mt-1.5 text-[10px] text-slate-300"
                    onChange={async (event) => {
                      const file = event.target.files?.[0] ?? null;
                      await onInitImageUpload(file);
                    }}
                  />
                  <p className="mt-1 text-[10px] text-slate-500">Obligatoire en mode Image → Vidéo.</p>
                </div>
              ) : (
                <div className="rounded-xl border border-white/10 bg-slate-900/60 p-3 text-[11px] text-slate-300">
                  L'image de référence sera générée automatiquement en utilisant vos réglages (modèle, steps, LoRA…).
                  Idéal pour lancer une vidéo directement depuis un prompt texte.
                </div>
              )}

              <div className="grid gap-3 md:grid-cols-2">
                <div>
                  <label className="text-xs font-semibold uppercase tracking-[0.2em] text-slate-400">Durée (s)</label>
                  <input
                    type="number"
                    min={0.5}
                    max={5}
                    step={0.1}
                    value={videoDuration}
                    onChange={handleNumericChange(onVideoDurationChange)}
                    className="mt-1.5 w-full rounded-xl border border-white/10 bg-slate-900/60 p-2 text-xs text-white"
                  />
                  <p className="mt-1 text-[10px] text-slate-500">
                    {numFrames} frames ({(numFrames / fps).toFixed(2)}s)
                  </p>
                </div>

                <div>
                  <label className="text-xs font-semibold uppercase tracking-[0.2em] text-slate-400">FPS</label>
                  <input
                    type="number"
                    min={3}
                    max={30}
                    value={fps}
                    onChange={handleNumericChange(onFpsChange)}
                    className="mt-1.5 w-full rounded-xl border border-white/10 bg-slate-900/60 p-2 text-xs text-white"
                  />
                  <p className="mt-1 text-[10px] text-slate-500">Recommandé: 6-8</p>
                </div>
              </div>

              <div>
                <label className="text-xs font-semibold uppercase tracking-[0.2em] text-slate-400">Frames (avancé)</label>
                <input
                  type="number"
                  min={6}
                  max={16}
                  value={numFrames}
                  onChange={handleNumericChange(onNumFramesChange)}
                  className="mt-1.5 w-full rounded-xl border border-white/10 bg-slate-900/60 p-2 text-xs text-white"
                />
                <p className="mt-1 text-[10px] text-slate-500">Durée: {(numFrames / fps).toFixed(2)}s</p>
              </div>
            </>
          )}
        </>
      )}

      {mode === "chat" && (
        <div className="space-y-4">
          <div className="rounded-2xl border border-white/10 bg-slate-900/60 p-4 max-h-[500px] overflow-y-auto space-y-4">
            {chatMessages.length === 0 && (
              <p className="text-sm text-slate-400 text-center py-8">
                Commencez une conversation avec l&apos;IA locale. Vous pouvez lui demander de créer ou améliorer vos
                prompts.
              </p>
            )}
            {chatMessages.map((msg, idx) => (
              <div key={idx} className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}>
                <div
                  className={`max-w-[80%] rounded-2xl px-4 py-2 ${
                    msg.role === "user" ? "bg-indigo-500/20 text-indigo-100" : "bg-slate-800/60 text-slate-200"
                  }`}
                >
                  <p className="text-sm whitespace-pre-wrap">{msg.content}</p>
                  {msg.images && msg.images.length > 0 && (
                    <div className="mt-2 flex flex-wrap gap-2">
                      {msg.images.map((image, imageIdx) => {
                        const previewSrc = image.startsWith("data:") ? image : `data:image/png;base64,${image}`;
                        return (
                          <img
                            key={`${idx}-${imageIdx}`}
                            src={previewSrc}
                            alt="Image jointe"
                            className="h-28 w-28 rounded-xl border border-white/10 object-cover"
                          />
                        );
                      })}
                    </div>
                  )}
                </div>
              </div>
            ))}
            {isChatting && (
              <div className="flex justify-start">
                <div className="rounded-2xl bg-slate-800/60 px-4 py-2">
                  <p className="text-sm text-slate-400">Réflexion...</p>
                </div>
              </div>
            )}
          </div>
          <div className="rounded-2xl border border-white/10 bg-slate-900/60 p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-xs font-semibold uppercase tracking-[0.2em] text-slate-400">Image jointe</p>
                <p className="text-[10px] text-slate-500">Optionnel — l&apos;IA pourra analyser votre image.</p>
              </div>
              {chatAttachmentPreview && (
                <button
                  type="button"
                  onClick={onChatAttachmentClear}
                  className="text-[10px] uppercase tracking-[0.2em] text-rose-200 transition hover:text-white"
                >
                  Retirer
                </button>
              )}
            </div>
            {chatAttachmentPreview ? (
              <div className="mt-3 flex items-center gap-3">
                <img
                  src={
                    chatAttachmentPreview.startsWith("data:")
                      ? chatAttachmentPreview
                      : `data:image/png;base64,${chatAttachmentPreview}`
                  }
                  alt="Pièce jointe"
                  className="h-24 w-24 rounded-xl border border-white/10 object-cover"
                />
                <div className="text-xs text-slate-300">
                  <p className="font-semibold">{chatAttachmentName ?? "image.png"}</p>
                  <p className="text-[10px] text-slate-500">L&apos;image sera envoyée au prochain message.</p>
                </div>
              </div>
            ) : (
              <label className="mt-3 inline-flex cursor-pointer items-center gap-2 rounded-xl border border-white/10 bg-slate-800/60 px-3 py-2 text-[11px] text-slate-200 transition hover:border-white/30">
                Importer une image
                <input
                  type="file"
                  accept="image/*"
                  className="hidden"
                  onChange={async (event) => {
                    const inputEl = event.target as HTMLInputElement;
                    const file = inputEl.files?.[0] ?? null;
                    await onChatImageUpload(file);
                    inputEl.value = "";
                  }}
                />
              </label>
            )}
          </div>
          <div className="flex gap-2">
            <input
              type="text"
              value={chatInput}
              onChange={(e) => onChatInputChange(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  onSendMessage();
                }
              }}
              placeholder="Tapez votre message..."
              disabled={isChatting}
              className="flex-1 rounded-xl border border-white/10 bg-slate-900/60 px-4 py-2 text-sm text-white placeholder:text-slate-500 focus:border-indigo-500 focus:outline-none disabled:opacity-50"
            />
            <button
              onClick={onSendMessage}
              disabled={isChatting || !chatInput.trim()}
              className="rounded-xl bg-indigo-500 px-4 py-2 text-sm font-semibold text-white transition hover:bg-indigo-600 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              Envoyer
            </button>
            <button
              onClick={onChatReset}
              className="rounded-xl border border-white/10 bg-slate-900/60 px-4 py-2 text-sm text-slate-300 transition hover:bg-slate-800"
              title="Réinitialiser la conversation"
            >
              Reset
            </button>
          </div>
        </div>
      )}

      <div className="rounded-2xl border border-white/10 bg-slate-900/40 p-4 text-xs text-slate-400">
        Backend attendu sur <span className="font-semibold text-white">{apiBase}/generate</span>. Configurez la variable{" "}
        <code>NEXT_PUBLIC_API_BASE_URL</code> si besoin.
      </div>
    </div>
  );
}

