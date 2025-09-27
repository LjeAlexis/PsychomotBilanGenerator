#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script standalone pour télécharger et organiser les modèles
Fonctionne indépendamment du module pbg
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, Optional

try:
    from huggingface_hub import snapshot_download
    from tqdm import tqdm
except ImportError:
    print("❌ Dépendances manquantes. Installez avec:")
    print("   pip install huggingface-hub tqdm")
    sys.exit(1)

# Configuration locale
MODELS_DIR = Path(os.getenv("MODELS_DIR", "./models")).resolve()
CACHE_DIR = Path(os.getenv("CACHE_DIR", "./cache")).resolve()

# Création des dossiers
MODELS_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Configuration des modèles
LOCAL_MODELS = {
    "mistral": {
        "local_path": MODELS_DIR / "mistral-7b-instruct-v0.3",
        "hf_name": "mistralai/Mistral-7B-Instruct-v0.3",
        "size_gb": 13.5,
    },
    "qwen": {
        "local_path": MODELS_DIR / "qwen2.5-7b-instruct",
        "hf_name": "Qwen/Qwen2.5-7B-Instruct",
        "size_gb": 13.5,
    },
    "vigogne": {
        "local_path": MODELS_DIR / "vigogne-2-7b-instruct",
        "hf_name": "bofenghuang/vigogne-2-7b-instruct",
        "size_gb": 13.5,
    },
    "llama3": {
        "local_path": MODELS_DIR / "llama-3-8b-instruct",
        "hf_name": "meta-llama/Meta-Llama-3-8B-Instruct",
        "size_gb": 16.0,
    },
    "biomistral": {
        "local_path": MODELS_DIR / "biomistral-7b",
        "hf_name": "BioMistral/BioMistral-7B",
        "size_gb": 13.5,
    },
}


def print_header():
    """Affiche l'en-tête du script"""
    print("\n" + "=" * 60)
    print("🤖 Gestionnaire de Modèles - PsychomotBilanGenerator")
    print("=" * 60)
    print(f"📁 Dossier modèles : {MODELS_DIR}")
    print(f"📁 Dossier cache : {CACHE_DIR}")
    print("=" * 60 + "\n")


def get_folder_size(path: Path) -> float:
    """Calcule la taille d'un dossier en GB"""
    if not path.exists():
        return 0.0
    total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return total / (1024**3)


def check_disk_space() -> float:
    """Vérifie l'espace disque disponible en GB"""
    stat = shutil.disk_usage(MODELS_DIR)
    return stat.free / (1024**3)


def is_model_downloaded(model_key: str) -> bool:
    """Vérifie si un modèle est téléchargé"""
    model_info = LOCAL_MODELS.get(model_key)
    if not model_info:
        return False

    local_path = model_info["local_path"]
    if not local_path.exists():
        return False

    # Vérifier qu'il y a des fichiers de poids
    has_weights = any(
        p.suffix in [".bin", ".safetensors", ".gguf"] for p in local_path.rglob("*")
    )
    return has_weights


def list_models():
    """Liste tous les modèles disponibles et téléchargés"""
    print_header()
    print("📚 Modèles configurés :\n")

    total_size = 0.0
    downloaded_count = 0

    for key, info in LOCAL_MODELS.items():
        local_path = info["local_path"]
        hf_name = info["hf_name"]
        expected_size = info["size_gb"]

        if is_model_downloaded(key):
            actual_size = get_folder_size(local_path)
            total_size += actual_size
            downloaded_count += 1
            status = f"✅ Téléchargé ({actual_size:.1f} GB)"

            # Info supplémentaire
            config_file = local_path / "config.json"
            if config_file.exists():
                try:
                    with open(config_file) as f:
                        config = json.load(f)
                    model_type = config.get("model_type", "inconnu")
                    status += f" - Type: {model_type}"
                except:
                    pass
        else:
            status = f"⬇️  Non téléchargé (~{expected_size:.1f} GB)"

        print(f"📦 {key:12} : {status}")
        print(f"   HF: {hf_name}")
        print(f"   Local: {local_path}")
        print()

    # Résumé
    print("=" * 60)
    free_space = check_disk_space()
    print(f"💾 Résumé :")
    print(f"   • Modèles téléchargés : {downloaded_count}/{len(LOCAL_MODELS)}")
    print(f"   • Espace utilisé : {total_size:.1f} GB")
    print(f"   • Espace disponible : {free_space:.1f} GB")

    if downloaded_count == 0:
        print("\n💡 Pour télécharger un modèle :")
        print("   python download_models_standalone.py --download mistral")


def download_model(model_key: str, force: bool = False):
    """Télécharge un modèle depuis HuggingFace"""

    if model_key not in LOCAL_MODELS:
        print(f"❌ Modèle '{model_key}' non reconnu")
        print(f"   Modèles disponibles : {', '.join(LOCAL_MODELS.keys())}")
        return False

    model_info = LOCAL_MODELS[model_key]
    local_path = model_info["local_path"]
    hf_name = model_info["hf_name"]
    expected_size = model_info["size_gb"]

    # Vérification si déjà téléchargé
    if is_model_downloaded(model_key) and not force:
        print(f"✅ '{model_key}' est déjà téléchargé dans : {local_path}")
        response = input("Voulez-vous le re-télécharger ? (o/n) : ")
        if response.lower() != "o":
            return True

    # Vérification espace disque
    free_space = check_disk_space()
    if free_space < expected_size * 1.2:  # 20% de marge
        print(f"❌ Espace disque insuffisant")
        print(f"   Nécessaire : {expected_size * 1.2:.1f} GB")
        print(f"   Disponible : {free_space:.1f} GB")
        return False

    print(f"\n📥 Téléchargement de '{model_key}'")
    print(f"   Modèle HF : {hf_name}")
    print(f"   Destination : {local_path}")
    print(f"   Taille estimée : {expected_size:.1f} GB")
    print()

    # Confirmation
    response = input("Confirmer le téléchargement ? (o/n) : ")
    if response.lower() != "o":
        print("❌ Téléchargement annulé")
        return False

    # Création du dossier
    local_path.mkdir(parents=True, exist_ok=True)

    try:
        print("\n⏳ Téléchargement en cours (cela peut prendre plusieurs minutes)...")

        # Patterns de fichiers à télécharger
        allow_patterns = [
            "*.safetensors",
            "*.bin",
            "*.json",
            "tokenizer*",
            "*.txt",
            "*.model",
            "special_tokens_map.json",
            "vocab.json",
            "merges.txt",
        ]

        # Téléchargement
        snapshot_download(
            repo_id=hf_name,
            local_dir=str(local_path),
            local_dir_use_symlinks=False,
            allow_patterns=allow_patterns,
            resume_download=True,
            max_workers=4,
        )

        # Vérification
        if is_model_downloaded(model_key):
            actual_size = get_folder_size(local_path)
            print(f"\n✅ Modèle téléchargé avec succès !")
            print(f"   Taille : {actual_size:.1f} GB")
            print(f"   Chemin : {local_path}")
            return True
        else:
            print("⚠️ Téléchargement terminé mais aucun fichier de poids trouvé")
            return False

    except KeyboardInterrupt:
        print("\n⚠️ Téléchargement interrompu par l'utilisateur")
        return False
    except Exception as e:
        print(f"❌ Erreur lors du téléchargement : {e}")
        return False


def download_all(force: bool = False):
    """Télécharge tous les modèles recommandés"""
    print_header()

    recommended = ["mistral", "qwen", "vigogne"]
    print(f"📦 Téléchargement des modèles recommandés : {', '.join(recommended)}\n")

    success_count = 0
    for model_key in recommended:
        print(f"\n[{recommended.index(model_key) + 1}/{len(recommended)}] {model_key}")
        print("-" * 40)
        if download_model(model_key, force):
            success_count += 1

    print("\n" + "=" * 60)
    print(f"✅ Téléchargement terminé : {success_count}/{len(recommended)} modèles")


def cleanup_cache():
    """Nettoie le cache HuggingFace"""
    hf_cache = Path.home() / ".cache" / "huggingface"

    if hf_cache.exists():
        cache_size = get_folder_size(hf_cache)
        print(f"\n🗑️  Cache HuggingFace : {cache_size:.1f} GB")
        print(f"   Chemin : {hf_cache}")

        response = input("\nVoulez-vous nettoyer ce cache ? (o/n) : ")
        if response.lower() == "o":
            try:
                shutil.rmtree(hf_cache)
                print("✅ Cache nettoyé avec succès")
            except Exception as e:
                print(f"❌ Erreur : {e}")
    else:
        print("ℹ️ Aucun cache HuggingFace trouvé")


def main():
    parser = argparse.ArgumentParser(
        description="Gestionnaire de modèles pour PsychomotBilanGenerator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--list", "-l", action="store_true", help="Lister tous les modèles"
    )

    parser.add_argument(
        "--download",
        "-d",
        choices=list(LOCAL_MODELS.keys()) + ["all"],
        help="Télécharger un modèle spécifique ou tous",
    )

    parser.add_argument(
        "--force", "-f", action="store_true", help="Forcer le re-téléchargement"
    )

    parser.add_argument(
        "--cleanup", action="store_true", help="Nettoyer le cache HuggingFace"
    )

    args = parser.parse_args()

    # Si aucun argument, afficher la liste
    if not any(vars(args).values()):
        list_models()
    elif args.list:
        list_models()
    elif args.cleanup:
        cleanup_cache()
    elif args.download:
        if args.download == "all":
            download_all(args.force)
        else:
            print_header()
            download_model(args.download, args.force)
    else:
        list_models()


if __name__ == "__main__":
    main()
