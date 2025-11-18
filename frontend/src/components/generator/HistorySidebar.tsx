import type { HistoryEntry } from "@/types/generator";

type HistorySidebarProps = {
  history: HistoryEntry[];
  onSelect: (entry: HistoryEntry) => void;
  onDelete: (id: string) => void;
};

export function HistorySidebar({ history, onSelect, onDelete }: HistorySidebarProps) {
  return (
    <aside className="hidden w-56 rounded-3xl border border-white/10 bg-white/5 p-4 shadow-2xl backdrop-blur md:block">
      <div>
        <h2 className="text-xs font-semibold uppercase tracking-[0.2em] text-slate-400">Historique</h2>
        <p className="mb-3 mt-1 text-[11px] text-slate-400">Prompts hors-ligne.</p>
        <div className="space-y-2 overflow-y-auto pr-1 max-h-[75vh]">
          {history.length === 0 && <p className="text-[11px] text-slate-500">Aucune génération.</p>}
          {history.map((entry) => (
            <div
              key={entry.id}
              className="group relative flex items-start gap-2 rounded-xl border border-white/10 bg-slate-900/60 p-2 transition hover:border-white/30"
            >
              <button
                type="button"
                onClick={() => onSelect(entry)}
                className="flex flex-1 items-center gap-2 text-left"
              >
                {entry.thumbnail ? (
                  // eslint-disable-next-line @next/next/no-img-element
                  <img
                    src={entry.thumbnail}
                    alt="miniature"
                    className="h-10 w-10 rounded-lg object-cover flex-shrink-0"
                  />
                ) : (
                  <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-slate-800 text-[10px] text-slate-400 flex-shrink-0">
                    {entry.model.substring(0, 4)}
                  </div>
                )}
                <div className="flex-1 min-w-0">
                  <p className="line-clamp-2 text-[11px] font-medium text-white leading-tight">{entry.prompt}</p>
                  <p className="text-[10px] text-slate-500 mt-0.5">
                    {new Date(entry.timestamp).toLocaleTimeString("fr-FR", {
                      hour: "2-digit",
                      minute: "2-digit",
                    })}
                  </p>
                </div>
              </button>
              <button
                type="button"
                onClick={() => onDelete(entry.id)}
                className="absolute right-1 top-1 flex h-4 w-4 items-center justify-center rounded-full border border-white/15 bg-slate-900/80 text-[10px] text-slate-400 opacity-0 transition group-hover:opacity-100 hover:border-rose-400 hover:text-rose-200"
                title="Supprimer"
              >
                ×
              </button>
            </div>
          ))}
        </div>
      </div>
    </aside>
  );
}

