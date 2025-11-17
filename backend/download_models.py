from __future__ import annotations

import argparse
from typing import Iterable

from model_registry import MODEL_SOURCES, ModelKey, ensure_model_file


def download(model_keys: Iterable[ModelKey]) -> None:
    for key in model_keys:
        print(f"⏬ Téléchargement du modèle '{key}' ...")
        path = ensure_model_file(key)
        if path.exists():
            print(f"✅ {key}: prêt dans {path}")
        else:
            print(f"⚠️  {key}: fichier introuvable après tentative de téléchargement.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Télécharge les poids Stable Diffusion depuis Hugging Face."
    )
    parser.add_argument(
        "models",
        nargs="*",
        choices=sorted(MODEL_SOURCES.keys()),
        help="Liste des modèles à télécharger (défaut: sdxl)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Télécharge toutes les entrées connues du registre.",
    )
    args = parser.parse_args()

    if args.all:
        targets = list(MODEL_SOURCES.keys())
    elif args.models:
        targets = args.models
    else:
        targets = ["sdxl"]

    download(targets)  # type: ignore[arg-type]


if __name__ == "__main__":
    main()

