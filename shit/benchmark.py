#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de benchmark pour comparer les modèles sur votre configuration
"""

import gc
import json
import time
from pathlib import Path
from typing import Dict, List

import pandas as pd
import torch
from dotenv import load_dotenv

# Chargement de la configuration
load_dotenv()

from src.config import CACHE_DIR, LOCAL_MODELS, get_model_path, list_available_models

MODELS_TO_TEST = {model: get_model_path(model) for model in list_available_models()}

if not MODELS_TO_TEST:
    print("⚠️ Aucun modèle trouvé localement. Téléchargez d'abord avec:")
    print("   python download_models.py --model mistral")
    exit(1)


def benchmark_model(
    model_key: str, model_path: str, test_text: str, num_runs: int = 3
) -> Dict:
    """Benchmark un modèle spécifique"""

    from src.config import SYSTEM_PROMPT
    from src.llm import generate_section, load_model_and_tokenizer

    results = {
        "model": model_key,
        "model_path": str(model_path),
        "load_time": 0,
        "generation_times": [],
        "tokens_per_second": [],
        "memory_used": 0,
        "quality_score": 0,
    }

    # Test de chargement
    print(f"\n🔄 Test de {model_key} ({model_path})")
    start = time.time()

    try:
        tokenizer, model = load_model_and_tokenizer(str(model_path), load_in_4bit=True)
        results["load_time"] = time.time() - start
        print(f"  ✅ Chargé en {results['load_time']:.1f}s")

        # Mémoire utilisée
        if torch.cuda.is_available():
            results["memory_used"] = torch.cuda.memory_allocated() / 1024**3
            print(f"  💾 Mémoire GPU: {results['memory_used']:.1f} GB")

        # Tests de génération
        for i in range(num_runs):
            start = time.time()

            output = generate_section(
                tok=tokenizer,
                model=model,
                system_prompt=SYSTEM_PROMPT,
                section_title="Évaluation psychomotrice",
                section_notes={"Tonus & posture": test_text},
                max_new_tokens=500,
                temperature=0.3,
            )

            gen_time = time.time() - start
            results["generation_times"].append(gen_time)

            # Calcul tokens/seconde
            num_tokens = len(tokenizer.encode(output))
            tps = num_tokens / gen_time
            results["tokens_per_second"].append(tps)

            print(f"  Run {i + 1}: {gen_time:.1f}s, {tps:.0f} tokens/s")

            # Score de qualité basique (longueur, termes pro)
            quality = len(output.split()) / 100  # Normalisation
            if any(
                term in output.lower() for term in ["tonus", "posture", "motricité"]
            ):
                quality += 0.5
            results["quality_score"] = max(results["quality_score"], quality)

        # Nettoyage
        del model
        del tokenizer
        gc.collect()
        torch.cuda.empty_cache()

    except Exception as e:
        print(f"  ❌ Erreur: {e}")
        results["error"] = str(e)

    return results


def run_benchmark(notes_file: Path = None):
    """Lance le benchmark complet"""

    print("🚀 Benchmark des modèles pour bilan psychomoteur")
    print("=" * 60)

    # Texte de test
    if notes_file and notes_file.exists():
        with open(notes_file, "r", encoding="utf-8") as f:
            test_notes = json.load(f)
            test_text = str(
                test_notes.get("sections", {}).get("Évaluation psychomotrice", "")
            )[:200]
    else:
        test_text = "Tonus axial équilibré, maintien postural satisfaisant. Légère hypotonie périphérique."

    # Info système
    print(f"📊 Configuration:")
    print(
        f"  GPU: {torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU'}"
    )
    if torch.cuda.is_available():
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"  VRAM totale: {total_memory:.1f} GB")
    print(f"  PyTorch: {torch.__version__}")

    # Benchmark de chaque modèle
    all_results = []

    for model_key, model_path in MODELS_TO_TEST.items():
        results = benchmark_model(model_key, model_path, test_text, num_runs=3)
        all_results.append(results)

        # Pause entre modèles
        time.sleep(5)

    # Rapport final
    print("\n" + "=" * 60)
    print("📈 RÉSULTATS DU BENCHMARK")
    print("=" * 60)

    # Création du DataFrame pour l'affichage
    df_data = []
    for r in all_results:
        if "error" not in r:
            avg_gen_time = (
                sum(r["generation_times"]) / len(r["generation_times"])
                if r["generation_times"]
                else 0
            )
            avg_tps = (
                sum(r["tokens_per_second"]) / len(r["tokens_per_second"])
                if r["tokens_per_second"]
                else 0
            )

            df_data.append(
                {
                    "Modèle": r["model"],
                    "Chargement (s)": f"{r['load_time']:.1f}",
                    "VRAM (GB)": f"{r['memory_used']:.1f}",
                    "Génération (s)": f"{avg_gen_time:.1f}",
                    "Tokens/s": f"{avg_tps:.0f}",
                    "Qualité": f"{r['quality_score']:.2f}",
                }
            )

    if df_data:
        df = pd.DataFrame(df_data)
        print(df.to_string(index=False))

        # Recommandation
        print("\n🏆 RECOMMANDATION:")
        best_speed = min(
            all_results, key=lambda x: sum(x.get("generation_times", [float("inf")]))
        )
        best_quality = max(all_results, key=lambda x: x.get("quality_score", 0))

        print(f"  Meilleure vitesse: {best_speed['model'].split('/')[-1]}")
        print(f"  Meilleure qualité: {best_quality['model'].split('/')[-1]}")

        # Calcul du meilleur compromis
        for r in all_results:
            if "error" not in r:
                avg_time = sum(r["generation_times"]) / len(r["generation_times"])
                r["score"] = r["quality_score"] / (avg_time / 10)  # Normalisation

        best_overall = max(
            [r for r in all_results if "error" not in r],
            key=lambda x: x.get("score", 0),
        )
        print(f"  Meilleur compromis: {best_overall['model'].split('/')[-1]}")

        # Sauvegarde des résultats
        output_file = Path("benchmark_results.json")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        print(f"\n💾 Résultats sauvegardés: {output_file}")


if __name__ == "__main__":
    import sys

    notes_file = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    run_benchmark(notes_file)
