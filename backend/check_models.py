from __future__ import annotations

"""
Petit script utilitaire pour vérifier que les fichiers de modèles
pointés par MODEL_CONFIG existent bien sur le disque.

À lancer depuis le dossier backend :

    python check_models.py
"""

from pathlib import Path

from model_registry import MODEL_CONFIG, ModelKey


def main() -> None:
  print("Vérification des modèles déclarés dans MODEL_CONFIG:\n")
  for key in MODEL_CONFIG.keys():
    config = MODEL_CONFIG[key]  # type: ignore[index]
    path = Path(config["path"]).expanduser()
    exists = path.exists()
    status = "OK ✅" if exists else "MANQUANT ❌"
    print(f"- {key:20s} -> {path}  [{status}]")


if __name__ == "__main__":
  main()


