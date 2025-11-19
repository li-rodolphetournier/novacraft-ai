import type {
  LoraOption,
  ModelKey,
  Mode,
  SelectedLora,
} from "@/types/generator";

type ModelDescriptor = {
  label: string;
  value: ModelKey;
  note: string;
  available: boolean;
  highlight?: string;
  nsfw?: boolean;
};

type ModelSettingsPanelProps = {
  mode: Mode;
  models: ModelDescriptor[];
  model: ModelKey;
  seed: string;
  clipSkip: number;
  selectedLoras: SelectedLora[];
  availableLoras: LoraOption[];
  isSubmitting: boolean;
  imageCount: number;
  onModelChange: (value: ModelKey) => void;
  onSeedChange: (value: string) => void;
  onClipSkipChange: (value: number) => void;
  onToggleLora: (key: string) => void;
  onLoraWeightChange: (key: string, weight: number) => void;
  onClearLoras: () => void;
  onGenerate: () => void;
  onImageCountChange: (value: number) => void;
  jobLabel?: string | null;
  jobStatusText?: string | null;
  jobProgressPercent?: number | null;
};

export function ModelSettingsPanel({
  mode,
  models,
  model,
  seed,
  clipSkip,
  selectedLoras,
  availableLoras,
  isSubmitting,
  imageCount,
  onModelChange,
  onSeedChange,
  onClipSkipChange,
  onToggleLora,
  onLoraWeightChange,
  onClearLoras,
  onGenerate,
  onImageCountChange,
  jobLabel,
  jobStatusText,
  jobProgressPercent,
}: ModelSettingsPanelProps) {
  return (
    <div className="w-full space-y-4 rounded-3xl border border-white/10 bg-white/5 p-4 shadow-2xl">
      <div>
        <p className="text-xs font-semibold uppercase tracking-[0.2em] text-slate-400">Choisir le modèle (CivitAI-like)</p>
        <div className="mt-2 space-y-1.5 max-h-[200px] overflow-y-auto pr-1">
          {models.map((item) => (
            <label
              key={item.value}
              className={`flex cursor-pointer items-center justify-between rounded-xl border p-2 text-xs ${
                model === item.value ? "border-indigo-400 bg-indigo-400/10 text-white" : "border-white/10 text-slate-300"
              } ${!item.available ? "cursor-not-allowed opacity-60" : "hover:border-white/30"}`}
            >
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-1.5">
                  <p className="font-semibold text-xs truncate">{item.label}</p>
                  {item.highlight && (
                    <span className="rounded-full bg-indigo-500/20 px-1.5 py-0.5 text-[9px] font-semibold uppercase tracking-wide text-indigo-200 flex-shrink-0">
                      {item.highlight}
                    </span>
                  )}
                </div>
                {!item.available && <p className="text-[9px] text-amber-200 mt-0.5">Non disponible</p>}
              </div>
              <input
                type="radio"
                name="model"
                value={item.value}
                checked={model === item.value}
                onChange={() => item.available && onModelChange(item.value)}
                className="accent-indigo-400 ml-2 flex-shrink-0"
                disabled={!item.available}
              />
            </label>
          ))}
        </div>
      </div>

      <div className="grid gap-3 md:grid-cols-2">
        <div>
          <p className="text-xs font-semibold uppercase tracking-[0.2em] text-slate-400">Seed</p>
          <input
            className="mt-1.5 w-full rounded-xl border border-white/10 bg-slate-900/60 p-2 text-xs text-white"
            type="number"
            value={seed}
            onChange={(event) => onSeedChange(event.target.value)}
            placeholder="-1 pour aléatoire"
          />
          <p className="mt-1 text-[10px] text-slate-500">-1 = seed aléatoire</p>
        </div>

        <div>
          <p className="text-xs font-semibold uppercase tracking-[0.2em] text-slate-400">Clip skip</p>
          <input
            className="mt-1.5 w-full rounded-xl border border-white/10 bg-slate-900/60 p-2 text-xs text-white"
            type="number"
            min={1}
            max={12}
            value={clipSkip}
            onChange={(event) => onClipSkipChange(Number(event.target.value))}
          />
          <p className="mt-1 text-[10px] text-slate-500">2 recommandé SDXL.</p>
        </div>
      </div>

      <div className="border-t border-white/5 pt-3">
        <div className="flex items-center justify-between">
          <div>
            <p className="text-xs font-semibold uppercase tracking-[0.2em] text-slate-400">Styles / LoRA</p>
            <p className="text-[10px] text-slate-500">Combinez plusieurs LoRA</p>
          </div>
          {selectedLoras.length > 0 && (
            <button
              className="text-[10px] uppercase tracking-[0.2em] text-rose-200 transition hover:text-white"
              onClick={onClearLoras}
              type="button"
            >
              Effacer
            </button>
          )}
        </div>
        {availableLoras.length === 0 ? (
          <p className="mt-2 rounded-xl border border-dashed border-white/10 p-2 text-[10px] text-slate-400">
            Ajoutez vos fichiers LoRA dans <code>backend/models/lora</code> puis rechargez la page.
          </p>
        ) : (
          <div className="mt-2 space-y-2 max-h-[250px] overflow-y-auto pr-1">
            {availableLoras.map((option) => {
              const active = selectedLoras.find((item) => item.key === option.key);
              return (
                <div
                  key={option.key}
                  className={`rounded-xl border p-2 ${
                    active ? "border-indigo-400 bg-indigo-500/10" : "border-white/10 bg-slate-900/40"
                  }`}
                >
                  <label className="flex cursor-pointer items-center justify-between gap-2">
                    <div className="flex-1 min-w-0">
                      <p className="text-xs font-semibold text-white truncate">{option.label}</p>
                      {option.description && (
                        <p className="text-[10px] text-slate-400 truncate">{option.description}</p>
                      )}
                    </div>
                    <input
                      type="checkbox"
                      checked={!!active}
                      onChange={() => onToggleLora(option.key)}
                      className="h-3.5 w-3.5 accent-indigo-400 flex-shrink-0"
                    />
                  </label>
                  {active && (
                    <div className="mt-2 space-y-1.5">
                      <input
                        type="range"
                        min={-3}
                        max={3}
                        step={0.05}
                        value={active.weight}
                        onChange={(event) => onLoraWeightChange(option.key, Number(event.target.value))}
                        className="w-full accent-indigo-400"
                      />
                      <div className="flex items-center justify-between text-[10px] text-slate-300">
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

      {(mode === "image" || mode === "video") && (
        <>
          <button
            onClick={onGenerate}
            disabled={isSubmitting}
            className="relative flex w-full items-center justify-center rounded-xl bg-gradient-to-r from-indigo-500 via-purple-500 to-rose-500 py-2 text-xs font-semibold text-white shadow-lg shadow-indigo-500/30 transition hover:scale-[1.01] disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {isSubmitting ? "Création du job…" : mode === "video" ? "Programmer la vidéo" : "Programmer les images"}
          </button>

          {(jobLabel || jobStatusText || typeof jobProgressPercent === "number") && (
            <div className="space-y-1 rounded-xl border border-white/10 bg-white/5 p-3 text-[12px] text-slate-200">
              {jobLabel && <p className="font-semibold text-slate-100">{jobLabel}</p>}
              {typeof jobProgressPercent === "number" && (
                <div className="h-1.5 w-full overflow-hidden rounded-full bg-slate-800">
                  <div
                    className="h-full bg-gradient-to-r from-indigo-400 to-purple-500 transition-all duration-300"
                    style={{ width: `${Math.min(100, Math.max(0, jobProgressPercent))}%` }}
                  />
                </div>
              )}
              {jobStatusText && (
                <p className="text-[11px] text-slate-300">
                  {jobStatusText}
                  {typeof jobProgressPercent === "number" && ` (${jobProgressPercent}%)`}
                </p>
              )}
            </div>
          )}
        </>
      )}

      {mode === "image" && (
        <div className="grid gap-3 md:grid-cols-2">
          <div>
            <p className="text-xs font-semibold uppercase tracking-[0.2em] text-slate-400">Image count</p>
            <input
              className="mt-1.5 w-full rounded-xl border border-white/10 bg-slate-900/60 p-2 text-xs text-white"
              type="number"
              min={1}
              max={4}
              value={imageCount}
              onChange={(event) => onImageCountChange(Math.min(4, Math.max(1, Number(event.target.value))))}
            />
            <p className="mt-1 text-[10px] text-slate-500">Jusqu&apos;à 4 images.</p>
          </div>
        </div>
      )}
    </div>
  );
}

