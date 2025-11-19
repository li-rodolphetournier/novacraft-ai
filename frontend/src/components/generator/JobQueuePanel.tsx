"use client";

import { motion } from "framer-motion";
import { FaImage, FaVideo, FaTimes } from "react-icons/fa";
import type { Job } from "@/types/generator";

type JobQueuePanelProps = {
  jobs: Job[];
  selectedJobId: string | null;
  hasRunningJob: boolean;
  jobsError: string | null;
  onSelectJob: (jobId: string) => void;
  onPause: (jobId: string) => void;
  onResume: (jobId: string) => void;
  onStart: (jobId: string) => void;
  onCancel: (jobId: string) => void;
  onDelete: (jobId: string) => void;
};

const statusLabels: Record<string, string> = {
  pending: "En file",
  running: "En cours",
  paused: "En pause",
  completed: "Terminé",
  failed: "Échec",
  cancelled: "Annulé",
};

const statusColors: Record<string, string> = {
  pending: "bg-slate-600/30 text-slate-200 border-slate-500/40",
  running: "bg-emerald-500/20 text-emerald-200 border-emerald-400/40",
  paused: "bg-amber-500/20 text-amber-100 border-amber-400/30",
  completed: "bg-indigo-500/20 text-indigo-100 border-indigo-400/30",
  failed: "bg-rose-500/20 text-rose-100 border-rose-400/30",
  cancelled: "bg-slate-500/20 text-slate-200 border-slate-400/30",
};

const typeIconMap = {
  image: FaImage,
  video: FaVideo,
};

export function JobQueuePanel({
  jobs,
  selectedJobId,
  hasRunningJob,
  jobsError,
  onSelectJob,
  onPause,
  onResume,
  onStart,
  onCancel,
  onDelete,
}: JobQueuePanelProps) {
  return (
    <div className="mt-8 w-full rounded-3xl border border-white/10 bg-white/5 p-4 shadow-2xl">
      <div className="flex items-center justify-between gap-4">
        <div>
          <p className="text-xs font-semibold uppercase tracking-[0.2em] text-slate-400">File de jobs</p>
          <p className="text-[11px] text-slate-500">
            Lancez plusieurs jobs à la suite — l’ordre est respecté et chaque job est récupérable.
          </p>
        </div>
        {jobsError && <p className="text-xs text-rose-300">{jobsError}</p>}
      </div>

      {jobs.length === 0 ? (
        <p className="mt-4 rounded-2xl border border-dashed border-white/10 p-3 text-[12px] text-slate-400">
          Aucun job pour le moment. Créez un job d&apos;image ou de vidéo pour remplir la file.
        </p>
      ) : (
        <div className="mt-4 space-y-3 max-h-[380px] overflow-y-auto pr-1">
          {jobs.map((job) => {
            const statusClass =
              statusColors[job.status] ?? "bg-slate-600/30 text-slate-200 border-slate-500/40";
            const progress = job.progress;
            const percent =
              progress && progress.total > 0
                ? Math.min(100, Math.round((progress.current / progress.total) * 100))
                : null;
            const isSelected = selectedJobId === job.id;
            const IconComponent = typeIconMap[job.type as keyof typeof typeIconMap] || FaImage;

            const startDisabled = hasRunningJob;
            const resumeDisabled = hasRunningJob;
            const createdTimestamp = job.created_at ?? Date.now();
            const createdMs = createdTimestamp > 1e12 ? createdTimestamp : createdTimestamp * 1000;
            const canShowDeleteIcon = true;

            return (
              <motion.div
                key={job.id}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -10 }}
                transition={{ duration: 0.2 }}
                className={`relative rounded-2xl border p-3 text-xs transition ${
                  isSelected ? "border-indigo-400 bg-indigo-500/10" : "border-white/10 bg-slate-900/40"
                }`}
              >
                <button
                  type="button"
                  className="flex w-full items-start gap-3 text-left"
                  onClick={() => onSelectJob(job.id)}
                >
                  <IconComponent className="text-xl text-indigo-400" />
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center justify-between gap-2">
                      <div className="min-w-0">
                        <p className="font-semibold text-white truncate">
                          {job.metadata?.prompt_preview || `Job ${job.id}`}
                        </p>
                        <p className="text-[10px] text-slate-500">
                          {new Date(createdMs).toLocaleString()}
                        </p>
                      </div>
                      <span
                        className={`rounded-full border px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wide ${statusClass}`}
                      >
                        {statusLabels[job.status] ?? job.status}
                      </span>
                    </div>

                    {progress && (
                      <div className="mt-2">
                        <div className="flex items-center justify-between text-[10px] text-slate-400">
                          <span>{progress.message ?? "Progression"}</span>
                          {percent !== null && <span>{percent}%</span>}
                        </div>
                        <div className="mt-1 h-1.5 w-full overflow-hidden rounded-full bg-slate-800">
                          <div
                            className="h-full bg-gradient-to-r from-indigo-500 to-purple-500 transition-all duration-300"
                            style={{ width: `${percent ?? 0}%` }}
                          />
                        </div>
                      </div>
                    )}

                    {job.error && (
                      <p className="mt-2 rounded-xl border border-rose-400/30 bg-rose-500/10 p-2 text-[11px] text-rose-200">
                        {job.error}
                      </p>
                    )}
                  </div>
                </button>
                {canShowDeleteIcon && (
                  <button
                    type="button"
                    onClick={(event) => {
                      event.stopPropagation();
                      if (job.status === "running") {
                        onCancel(job.id);
                      } else {
                        onDelete(job.id);
                      }
                    }}
                    className="absolute right-2 top-2 rounded-full border border-rose-400/40 p-1.5 text-rose-200 hover:bg-rose-400/10 transition"
                    aria-label="Retirer le job"
                  >
                    <FaTimes className="text-xs" />
                  </button>
                )}

                <div className="mt-3 flex flex-wrap gap-2 text-[11px]">
                  {job.status === "running" && (
                    <>
                      <button
                        type="button"
                        onClick={() => onPause(job.id)}
                        className="rounded-full border border-amber-400/40 px-3 py-1 text-amber-100 hover:bg-amber-400/10"
                      >
                        Pause
                      </button>
                      <button
                        type="button"
                        onClick={() => onCancel(job.id)}
                        className="rounded-full border border-rose-400/40 px-3 py-1 text-rose-200 hover:bg-rose-400/10"
                      >
                        Stop
                      </button>
                    </>
                  )}

                  {job.status === "pending" && (
                    <>
                      <button
                        type="button"
                        onClick={() => onStart(job.id)}
                        disabled={startDisabled}
                        className="rounded-full border border-emerald-400/40 px-3 py-1 text-emerald-200 hover:bg-emerald-400/10 disabled:opacity-40"
                      >
                        Reprendre
                      </button>
                      <button
                        type="button"
                        onClick={() => onCancel(job.id)}
                        className="rounded-full border border-rose-400/40 px-3 py-1 text-rose-200 hover:bg-rose-400/10"
                      >
                        Annuler
                      </button>
                    </>
                  )}

                  {job.status === "paused" && (
                    <>
                      <button
                        type="button"
                        onClick={() => onResume(job.id)}
                        disabled={resumeDisabled}
                        className="rounded-full border border-emerald-400/40 px-3 py-1 text-emerald-200 hover:bg-emerald-400/10 disabled:opacity-40"
                      >
                        Reprendre
                      </button>
                      <button
                        type="button"
                        onClick={() => onCancel(job.id)}
                        className="rounded-full border border-rose-400/40 px-3 py-1 text-rose-200 hover:bg-rose-400/10"
                      >
                        Annuler
                      </button>
                    </>
                  )}

                  {job.status === "failed" && (
                    <>
                      <button
                        type="button"
                        onClick={() => onResume(job.id)}
                        disabled={resumeDisabled}
                        className="rounded-full border border-amber-400/40 px-3 py-1 text-amber-100 hover:bg-amber-400/10 disabled:opacity-40"
                      >
                        Relancer
                      </button>
                      <button
                        type="button"
                        onClick={() => onCancel(job.id)}
                        className="rounded-full border border-rose-400/40 px-3 py-1 text-rose-200 hover:bg-rose-400/10"
                      >
                        Supprimer
                      </button>
                    </>
                  )}

                  {job.status === "completed" && (
                    <p className="text-[10px] text-slate-400">Résultat stocké dans la galerie / historique.</p>
                  )}
                </div>
              </motion.div>
            );
          })}
        </div>
      )}
    </div>
  );
}

