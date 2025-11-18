from __future__ import annotations

import argparse
import sys
from typing import Iterable

from model_registry import MODEL_SOURCES, ModelKey, ensure_model_file


def format_size(size_bytes: int) -> str:
    """Formate la taille en bytes en format lisible."""
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"


def download(model_keys: Iterable[ModelKey]) -> None:
    """Télécharge les modèles depuis Hugging Face."""
    keys_list = list(model_keys)
    total = len(keys_list)
    current = 0
    
    for key in keys_list:
        current += 1
        print(f"\n[{current}/{total}] ⏬ Téléchargement du modèle '{key}'...")
        
        if key not in MODEL_SOURCES:
            print(f"⚠️  {key}: pas de source Hugging Face configurée (modèle local uniquement)")
            continue
            
        source = MODEL_SOURCES[key]
        print(f"   📦 Repository: {source['repo_id']}")
        print(f"   📄 Fichier: {source['filename']}")
        
        try:
            path = ensure_model_file(key)
            if path.exists():
                size = path.stat().st_size
                print(f"✅ {key}: téléchargé avec succès")
                print(f"   📍 Chemin: {path}")
                print(f"   💾 Taille: {format_size(size)}")
            else:
                print(f"⚠️  {key}: fichier introuvable après tentative de téléchargement")
                print(f"   📍 Chemin attendu: {path}")
        except FileNotFoundError as e:
            print(f"❌ {key}: erreur de téléchargement")
            print(f"   {str(e)}")
            print(f"   💡 Vérifiez votre connexion Internet et que le repository existe sur Hugging Face")
        except Exception as e:
            print(f"❌ {key}: erreur inattendue")
            print(f"   {type(e).__name__}: {str(e)}")
    
    print("\n" + "="*60)
    print("✅ Téléchargement terminé!")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Télécharge les poids Stable Diffusion depuis Hugging Face.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples:
  # Télécharger SDXL uniquement
  python download_models.py sdxl
  
  # Télécharger plusieurs modèles
  python download_models.py sdxl realistic-vision dreamshaper
  
  # Télécharger tous les modèles disponibles
  python download_models.py --all
  
  # Lister les modèles disponibles
  python download_models.py --list
        """,
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
        help="Télécharge tous les modèles disponibles depuis Hugging Face.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Liste tous les modèles disponibles et quitte.",
    )
    args = parser.parse_args()

    if args.list:
        print("📋 Modèles disponibles depuis Hugging Face:\n")
        for i, key in enumerate(sorted(MODEL_SOURCES.keys()), 1):
            source = MODEL_SOURCES[key]
            print(f"  {i}. {key}")
            print(f"     Repository: {source['repo_id']}")
            print(f"     Fichier: {source['filename']}\n")
        sys.exit(0)

    if args.all:
        targets = list(MODEL_SOURCES.keys())
        print(f"🚀 Téléchargement de {len(targets)} modèles depuis Hugging Face...")
    elif args.models:
        targets = args.models
        print(f"🚀 Téléchargement de {len(targets)} modèle(s)...")
    else:
        targets = ["sdxl"]
        print("🚀 Téléchargement du modèle SDXL (par défaut)...")

    download(targets)  # type: ignore[arg-type]


if __name__ == "__main__":
    main()

