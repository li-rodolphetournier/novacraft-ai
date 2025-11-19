import type { Mode } from "@/types/generator";

type GeneratorHeaderProps = {
  mode: Mode;
  onModeChange: (mode: Mode) => void;
  showNSFW: boolean;
  onToggleNSFW: (value: boolean) => void;
};

export function GeneratorHeader({ mode, onModeChange, showNSFW, onToggleNSFW }: GeneratorHeaderProps) {
  return (
    <header className="rounded-3xl border border-white/10 bg-gradient-to-r from-slate-900/80 to-slate-900/40 p-6 shadow-2xl">
      <div className="flex items-start justify-between gap-4">
        <div>
          <p className="text-xs uppercase tracking-[0.3em] text-indigo-300">NovaCraft AI</p>
          <h1 className="mt-2 text-3xl font-semibold text-white">Génération IA Locale</h1>
          <p className="mt-2 text-sm text-slate-300">
            Prompt + paramètres essentiels, fonctionnement 100% offline.
          </p>
        </div>
        <div className="flex items-center gap-4 text-xs">
          {(["image", "video", "chat"] as Mode[]).map((value) => (
            <button
              key={value}
              type="button"
              onClick={() => onModeChange(value)}
              className={`rounded-full px-3 py-1 border ${
                mode === value ? "bg-white text-slate-900 border-white" : "bg-slate-900/60 text-slate-300 border-white/20"
              }`}
            >
              {value === "image" && "Image"}
              {value === "video" && "Vidéo (beta)"}
              {value === "chat" && "Chat IA Locale"}
            </button>
          ))}
          <label className="flex items-center gap-2 text-[11px] text-slate-300">
            <input
              type="checkbox"
              checked={showNSFW}
              onChange={(event) => onToggleNSFW(event.target.checked)}
              className="h-4 w-4 rounded border-white/30 bg-transparent text-indigo-400 focus:ring-indigo-400"
            />
            Contenu NSFW
          </label>
        </div>
      </div>
    </header>
  );
}

