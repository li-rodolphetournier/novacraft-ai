"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import type {
  GeneratedImage,
  GeneratedVideo,
  ModelKey,
  Sampler,
} from "@/types/generator";

type GallerySectionProps = {
  images: GeneratedImage[];
  videos: GeneratedVideo[];
  models: { label: string; value: ModelKey }[];
  samplers: { label: string; value: Sampler }[];
  onDeleteImage: (id: string) => void;
  onCopyImage: (image: GeneratedImage) => void;
  onExportImage: (image: GeneratedImage) => void;
  onClearImages?: () => void;
  onDeleteVideo: (id: string) => void;
  onCopyVideo: (video: GeneratedVideo) => void;
  onExportVideo: (video: GeneratedVideo) => void;
};

export function GallerySection({
  images,
  videos,
  models,
  samplers,
  onDeleteImage,
  onCopyImage,
  onExportImage,
  onClearImages,
  onDeleteVideo,
  onCopyVideo,
  onExportVideo,
}: GallerySectionProps) {
  return (
    <div className="rounded-3xl border border-white/10 bg-white/5 p-6 shadow-2xl">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <p className="text-xs uppercase tracking-[0.2em] text-slate-400">Galerie</p>
          <h2 className="text-2xl font-semibold text-white">Dernières images</h2>
        </div>
        {images.length > 0 && (
          <button
            onClick={() => {
              if (onClearImages) {
                onClearImages();
              } else {
                images.forEach((image) => onDeleteImage(image.id));
              }
            }}
            className="text-xs uppercase tracking-[0.2em] text-slate-400 transition hover:text-white"
            type="button"
          >
            Effacer
          </button>
        )}
      </div>

      <div className="mt-6 grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
        <AnimatePresence mode="popLayout">
          {images.length === 0 && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="col-span-full rounded-3xl border border-dashed border-white/20 p-10 text-center text-sm text-slate-400"
            >
              Les rendus apparaîtront ici.
            </motion.div>
          )}
          {images.map((image, index) => (
            <motion.div
              key={image.id}
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.9 }}
              transition={{ duration: 0.2, delay: index * 0.05 }}
            >
              <ImageCard
                image={image}
                onDelete={onDeleteImage}
                onCopy={onCopyImage}
                onExport={onExportImage}
                models={models}
                samplers={samplers}
              />
            </motion.div>
          ))}
        </AnimatePresence>
      </div>

      {videos.length > 0 && (
        <div className="mt-10">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-semibold text-slate-200 uppercase tracking-[0.2em]">Vidéos</h3>
            {videos.length > 0 && (
              <button
                onClick={() => {
                  videos.forEach((video) => onDeleteVideo(video.id));
                }}
                className="text-xs uppercase tracking-[0.2em] text-slate-400 transition hover:text-white"
                type="button"
              >
                Effacer toutes
              </button>
            )}
          </div>
          <div className="mt-4 grid gap-4 sm:grid-cols-2">
            {videos.map((video, index) => (
              <VideoCard
                key={video.id}
                video={video}
                index={index}
                onDelete={onDeleteVideo}
                onCopy={onCopyVideo}
                onExport={onExportVideo}
              />
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

type ImageCardProps = {
  image: GeneratedImage;
  models: { label: string; value: ModelKey }[];
  samplers: { label: string; value: Sampler }[];
  onDelete: (id: string) => void;
  onCopy: (image: GeneratedImage) => void;
  onExport: (image: GeneratedImage) => void;
};

function ImageCard({ image, models, samplers, onDelete, onCopy, onExport }: ImageCardProps) {
  const [showMetadata, setShowMetadata] = useState(false);

  const getSamplerLabel = (value?: string) => samplers.find((s) => s.value === value)?.label || value || "N/A";
  const getModelLabel = (value?: string) => models.find((m) => m.value === value)?.label || value || "N/A";

  return (
    <motion.div
      whileHover={{ scale: 1.02 }}
      className="group relative rounded-3xl border border-white/10 bg-black/30 p-2 transition hover:border-indigo-400"
    >
      <button
        type="button"
        onClick={() => onDelete(image.id)}
        className="absolute right-3 top-3 z-10 flex h-6 w-6 items-center justify-center rounded-full border border-white/15 bg-black/80 text-xs text-slate-300 opacity-0 transition group-hover:opacity-100 hover:border-rose-400 hover:text-rose-200"
        title="Retirer cette image de la galerie"
      >
        ×
      </button>
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img src={`data:image/png;base64,${image.base64}`} alt={`seed ${image.seed}`} className="h-64 w-full rounded-2xl object-cover" />
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
            <div className="mb-2 text-xs font-semibold uppercase tracking-wider text-slate-400">Resources Used</div>
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
                  <div className="text-[10px] uppercase tracking-wider text-slate-500">LoRA appliqués</div>
                  {image.loras.map((lora) => (
                    <div
                      key={`${image.id}-${lora.key}-${lora.label}`}
                      className="rounded-xl border border-white/10 bg-slate-950/40 px-3 py-2 text-xs text-slate-200"
                    >
                      <div className="flex items-center justify-between">
                        <span>{lora.label}</span>
                        <span className="text-slate-300">poids {lora.weight.toFixed(2)}</span>
                      </div>
                      <p className="text-[10px] uppercase tracking-wider text-slate-500">{lora.type ?? "LoRA"}</p>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </motion.div>
  );
}

type VideoCardProps = {
  video: GeneratedVideo;
  index: number;
  onDelete: (id: string) => void;
  onCopy: (video: GeneratedVideo) => void;
  onExport: (video: GeneratedVideo) => void;
};

function VideoCard({ video, index, onDelete, onCopy, onExport }: VideoCardProps) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3, delay: index * 0.1 }}
      whileHover={{ scale: 1.02 }}
      className="group relative rounded-3xl border border-white/10 bg-black/40 p-2 transition hover:border-indigo-400"
    >
      <button
        type="button"
        onClick={() => onDelete(video.id)}
        className="absolute right-3 top-3 z-10 flex h-6 w-6 items-center justify-center rounded-full border border-white/15 bg-black/80 text-xs text-slate-300 opacity-0 transition group-hover:opacity-100 hover:border-rose-400 hover:text-rose-200"
        title="Retirer cette vidéo de la galerie"
      >
        ×
      </button>
      <video controls className="w-full rounded-2xl" src={`data:video/mp4;base64,${video.mp4Base64}`} />
      <div className="mt-2 space-y-2">
        {video.durationSeconds != null && (
          <div className="flex items-center justify-between text-xs text-slate-400">
            <span>
              Durée&nbsp;: <span className="font-semibold text-slate-100">{video.durationSeconds.toFixed(1)}&nbsp;s</span>
            </span>
            <div className="flex gap-2">
              <button
                type="button"
                onClick={() => onCopy(video)}
                className="text-indigo-300 transition hover:text-white"
                title="Copier base64"
              >
                Copier
              </button>
              <button
                type="button"
                onClick={() => onExport(video)}
                className="text-green-300 transition hover:text-white"
                title="Télécharger MP4"
              >
                Télécharger
              </button>
            </div>
          </div>
        )}
        {!video.durationSeconds && (
          <div className="flex justify-end gap-2 text-xs">
            <button
              type="button"
              onClick={() => onCopy(video)}
              className="text-indigo-300 transition hover:text-white"
              title="Copier base64"
            >
              Copier
            </button>
            <button
              type="button"
              onClick={() => onExport(video)}
              className="text-green-300 transition hover:text-white"
              title="Télécharger MP4"
            >
              Télécharger
            </button>
          </div>
        )}
      </div>
    </motion.div>
  );
}

