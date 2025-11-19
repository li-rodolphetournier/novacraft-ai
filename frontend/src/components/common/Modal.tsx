"use client";

import { motion, AnimatePresence } from "framer-motion";
import { ReactNode, useState, useEffect } from "react";

type ModalProps = {
  isOpen: boolean;
  onClose: () => void;
  title: string;
  children: ReactNode;
  confirmLabel?: string;
  cancelLabel?: string;
  onConfirm?: () => void;
  showCancel?: boolean;
};

export function Modal({
  isOpen,
  onClose,
  title,
  children,
  confirmLabel = "Confirmer",
  cancelLabel = "Annuler",
  onConfirm,
  showCancel = true,
}: ModalProps) {
  if (!isOpen) return null;

  const handleConfirm = () => {
    if (onConfirm) {
      onConfirm();
    }
    onClose();
  };

  return (
    <AnimatePresence>
      {isOpen && (
        <>
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="fixed inset-0 z-50 bg-black/60 backdrop-blur-sm"
            onClick={onClose}
          />
          <motion.div
            initial={{ opacity: 0, scale: 0.95, y: 20 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.95, y: 20 }}
            transition={{ duration: 0.2 }}
            className="fixed left-1/2 top-1/2 z-50 w-full max-w-md -translate-x-1/2 -translate-y-1/2 rounded-2xl border border-white/10 bg-slate-900 p-6 shadow-2xl"
            onClick={(e) => e.stopPropagation()}
          >
            <h2 className="mb-4 text-xl font-semibold text-white">{title}</h2>
            <div className="mb-6 text-slate-300">{children}</div>
            <div className="flex justify-end gap-3">
              {showCancel && (
                <button
                  type="button"
                  onClick={onClose}
                  className="rounded-lg border border-white/15 bg-slate-800/60 px-4 py-2 text-sm text-slate-200 transition hover:bg-slate-700"
                >
                  {cancelLabel}
                </button>
              )}
              {onConfirm && (
                <button
                  type="button"
                  onClick={handleConfirm}
                  className="rounded-lg border border-indigo-400/40 bg-indigo-500/20 px-4 py-2 text-sm text-white transition hover:bg-indigo-500/30"
                >
                  {confirmLabel}
                </button>
              )}
            </div>
          </motion.div>
        </>
      )}
    </AnimatePresence>
  );
}

type PromptModalProps = {
  isOpen: boolean;
  onClose: () => void;
  title: string;
  placeholder?: string;
  defaultValue?: string;
  onConfirm: (value: string) => void;
  confirmLabel?: string;
  cancelLabel?: string;
};

export function PromptModal({
  isOpen,
  onClose,
  title,
  placeholder = "",
  defaultValue = "",
  onConfirm,
  confirmLabel = "Confirmer",
  cancelLabel = "Annuler",
}: PromptModalProps) {
  const [value, setValue] = useState(defaultValue);

  useEffect(() => {
    if (isOpen) {
      setValue(defaultValue);
    }
  }, [isOpen, defaultValue]);

  const handleConfirm = () => {
    if (value.trim()) {
      onConfirm(value.trim());
      onClose();
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter") {
      handleConfirm();
    } else if (e.key === "Escape") {
      onClose();
    }
  };

  return (
    <Modal
      isOpen={isOpen}
      onClose={onClose}
      title={title}
      onConfirm={handleConfirm}
      confirmLabel={confirmLabel}
      cancelLabel={cancelLabel}
    >
      <input
        type="text"
        value={value}
        onChange={(e) => setValue(e.target.value)}
        onKeyDown={handleKeyDown}
        placeholder={placeholder}
        className="w-full rounded-lg border border-white/10 bg-slate-800/60 px-4 py-2 text-white placeholder-slate-400 focus:border-indigo-400/40 focus:outline-none focus:ring-2 focus:ring-indigo-500/20"
        autoFocus
      />
    </Modal>
  );
}

