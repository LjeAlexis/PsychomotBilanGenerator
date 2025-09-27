"""
Interface en ligne de commande pour le générateur de bilans psychomoteurs
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import List, Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import track
from rich.table import Table

from config.settings import Config, default_config
from src.core.generator import PsychomotBilanGenerator
from src.utils.logging import setup_logging

app = typer.Typer(
    name="pbg",
    help="🧠 Générateur de Bilans Psychomoteurs avec IA",
    add_completion=False,
    rich_markup_mode="rich",
)

console = Console()


@app.command()
def generate(
    notes_file: Path = typer.Argument(
        ...,
        help="Fichier JSON contenant les notes du bilan",
        exists=True,
        file_okay=True,
        readable=True,
    ),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Fichier de sortie (optionnel)"
    ),
    model: str = typer.Option(
        default_config.default_model,
        "--model",
        "-m",
        help="Modèle à utiliser pour la génération",
    ),
    temperature: float = typer.Option(
        0.3,
        "--temperature",
        "-t",
        min=0.1,
        max=1.0,
        help="Créativité du modèle (0.1-1.0)",
    ),
    quality: bool = typer.Option(
        True, "--quality/--no-quality", help="Activer le contrôle qualité avancé"
    ),
    async_mode: bool = typer.Option(
        False, "--async", help="Mode génération asynchrone"
    ),
    retries: int = typer.Option(
        2, "--retries", "-r", min=0, max=5, help="Nombre de tentatives en cas d'échec"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Mode détaillé"),
):
    """
    🚀 Génère un bilan psychomoteur complet à partir de notes

    Exemple d'usage:
    pbg generate notes.json --model mistral --output mon_bilan.docx
    """

    if verbose:
        setup_logging("DEBUG")
    else:
        setup_logging("INFO")

    console.print(
        Panel.fit(
            f"🧠 [bold blue]Générateur de Bilans Psychomoteurs[/bold blue]\n"
            f"📁 Notes: {notes_file}\n"
            f"🤖 Modèle: {model}\n"
            f"🌡️ Température: {temperature}\n"
            f"⚡ Mode: {'Asynchrone' if async_mode else 'Synchrone'}\n"
            f"🔍 Qualité: {'Activée' if quality else 'Désactivée'}",
            title="Configuration",
        )
    )

    async def run_generation():
        try:
            # Initialisation du générateur
            generator = PsychomotBilanGenerator(
                model_name=model, enable_quality_checks=quality, enable_async=async_mode
            )

            await generator.initialize()

            # Génération du bilan
            output_path = await generator.generate_full_bilan(
                notes_file=notes_file,
                output_file=output,
                temperature=temperature,
                max_retries=retries,
            )

            console.print(f"\n✅ [bold green]Bilan généré avec succès![/bold green]")
            console.print(f"📄 Fichier: {output_path}")

            return str(output_path)

        except Exception as e:
            console.print(
                f"\n❌ [bold red]Erreur lors de la génération:[/bold red] {e}"
            )
            raise typer.Exit(1)

    # Exécution
    try:
        result = asyncio.run(run_generation())
        console.print(f"\n🎉 [bold]Génération terminée: {result}[/bold]")
    except KeyboardInterrupt:
        console.print("\n⚠️ Génération interrompue par l'utilisateur")
        raise typer.Exit(1)


@app.command()
def models():
    """
    📋 Liste les modèles disponibles
    """
    config = default_config

    table = Table(title="🤖 Modèles Disponibles")
    table.add_column("Nom", style="cyan", no_wrap=True)
    table.add_column("Type", style="magenta")
    table.add_column("Chemin/HF", style="green")
    table.add_column("Statut", style="yellow")
    table.add_column("Config", style="blue")

    for name, model_config in config.models.items():
        # Vérification disponibilité locale
        if model_config.local_path and model_config.local_path.exists():
            status = "✅ Local"
            path_info = str(model_config.local_path.name)
        else:
            status = "📥 HuggingFace"
            path_info = model_config.hf_name or "N/A"

        # Informations de configuration
        config_info = f"T:{model_config.temperature} | {model_config.quantization}"

        # Type de modèle
        if "bio" in name.lower():
            model_type = "🏥 Médical"
        elif "mistral" in name.lower():
            model_type = "🔥 Généraliste"
        elif "qwen" in name.lower():
            model_type = "🚀 Performant"
        else:
            model_type = "📝 Standard"

        table.add_row(name, model_type, path_info, status, config_info)

    console.print(table)

    # Informations additionnelles
    available_local = config.list_available_models()
    console.print(f"\n📊 Résumé:")
    console.print(f"  • Modèles configurés: {len(config.models)}")
    console.print(f"  • Disponibles localement: {len(available_local)}")
    console.print(f"  • Modèle par défaut: [bold]{config.default_model}[/bold]")


@app.command()
def validate(
    notes_file: Path = typer.Argument(
        ...,
        help="Fichier JSON des notes à valider",
        exists=True,
        file_okay=True,
        readable=True,
    ),
):
    """
    ✅ Valide la structure et le contenu d'un fichier de notes
    """
    try:
        with open(notes_file, "r", encoding="utf-8") as f:
            notes_data = json.load(f)

        console.print(f"📋 Validation de: {notes_file}")

        # Validation structure de base
        required_keys = ["titre", "sections"]
        missing_keys = [key for key in required_keys if key not in notes_data]

        if missing_keys:
            console.print(f"❌ Clés manquantes: {', '.join(missing_keys)}")
            raise typer.Exit(1)

        # Validation des sections
        sections = notes_data.get("sections", {})
        expected_sections = default_config.section_order

        table = Table(title="📝 Sections du Bilan")
        table.add_column("Section", style="cyan")
        table.add_column("Statut", style="green")
        table.add_column("Contenu", style="yellow")

        for section in expected_sections:
            if section in sections:
                content = sections[section]
                if isinstance(content, dict):
                    content_info = f"{len(content)} sous-sections"
                elif isinstance(content, list):
                    content_info = f"{len(content)} éléments"
                else:
                    content_info = f"{len(str(content))} caractères"

                table.add_row(section, "✅ Présente", content_info)
            else:
                table.add_row(section, "❌ Manquante", "N/A")

        console.print(table)

        # Statistiques générales
        total_sections = len([s for s in expected_sections if s in sections])
        completion_rate = (total_sections / len(expected_sections)) * 100

        console.print(f"\n📊 Statistiques:")
        console.print(f"  • Titre: {notes_data.get('titre', 'Non défini')}")
        console.print(
            f"  • Sections complètes: {total_sections}/{len(expected_sections)}"
        )
        console.print(f"  • Taux de completion: {completion_rate:.1f}%")

        if completion_rate < 50:
            console.print(
                "⚠️ [yellow]Attention: Beaucoup de sections manquantes[/yellow]"
            )
        elif completion_rate < 80:
            console.print("✅ [green]Fichier valide mais incomplet[/green]")
        else:
            console.print("🎉 [bold green]Fichier très complet![/bold green]")

    except json.JSONDecodeError as e:
        console.print(f"❌ Erreur JSON: {e}")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"❌ Erreur: {e}")
        raise typer.Exit(1)


@app.command()
def config(
    show: bool = typer.Option(
        False, "--show", help="Afficher la configuration actuelle"
    ),
    export: Optional[Path] = typer.Option(
        None, "--export", help="Exporter la config vers un fichier"
    ),
    validate_config: bool = typer.Option(
        False, "--validate", help="Valider la configuration"
    ),
):
    """
    ⚙️ Gestion de la configuration
    """
    if show:
        console.print(Panel.fit(str(default_config), title="Configuration Actuelle"))

    if validate_config:
        issues = default_config.validate_configuration()
        if issues:
            console.print("❌ [bold red]Problèmes de configuration:[/bold red]")
            for issue in issues:
                console.print(f"  • {issue}")
        else:
            console.print("✅ [bold green]Configuration valide[/bold green]")

    if export:
        try:
            default_config.export_config(export)
            console.print(f"✅ Configuration exportée vers: {export}")
        except Exception as e:
            console.print(f"❌ Erreur d'export: {e}")


@app.command()
def benchmark(
    notes_file: Path = typer.Argument(
        ..., help="Fichier de notes pour le benchmark", exists=True
    ),
    models: Optional[List[str]] = typer.Option(
        None, "--models", help="Modèles à tester"
    ),
    iterations: int = typer.Option(
        3, "--iterations", min=1, max=10, help="Nombre d'itérations"
    ),
    output_dir: Optional[Path] = typer.Option(
        None, "--output-dir", help="Dossier de sortie"
    ),
):
    """
    📊 Benchmark des modèles disponibles
    """
    console.print("🏁 [bold]Démarrage du benchmark...[/bold]")

    # Modèles à tester
    if models is None:
        models = default_config.list_available_models()
        if not models:
            console.print("❌ Aucun modèle local disponible pour le benchmark")
            raise typer.Exit(1)

    console.print(f"🤖 Modèles à tester: {', '.join(models)}")
    console.print(f"🔄 Itérations par modèle: {iterations}")

    results = {}

    async def run_benchmark():
        for model in models:
            console.print(f"\n🧪 Test du modèle: [bold]{model}[/bold]")

            model_results = {
                "times": [],
                "word_counts": [],
                "quality_scores": [],
                "errors": 0,
            }

            for i in track(range(iterations), description=f"Test {model}"):
                try:
                    generator = PsychomotBilanGenerator(
                        model_name=model, enable_quality_checks=True
                    )

                    await generator.initialize()

                    start_time = asyncio.get_event_loop().time()
                    output_path = await generator.generate_full_bilan(
                        notes_file=notes_file,
                        output_file=output_dir / f"benchmark_{model}_{i}.docx"
                        if output_dir
                        else None,
                    )
                    end_time = asyncio.get_event_loop().time()

                    # Collecte des métriques
                    generation_time = end_time - start_time
                    word_count = generator.stats.total_words
                    quality_score = generator.stats.quality_score

                    model_results["times"].append(generation_time)
                    model_results["word_counts"].append(word_count)
                    model_results["quality_scores"].append(quality_score)

                except Exception as e:
                    console.print(f"❌ Erreur itération {i}: {e}")
                    model_results["errors"] += 1

            results[model] = model_results

    # Exécution du benchmark
    asyncio.run(run_benchmark())

    # Affichage des résultats
    table = Table(title="📊 Résultats du Benchmark")
    table.add_column("Modèle", style="cyan")
    table.add_column("Temps Moyen", style="green")
    table.add_column("Mots/s", style="yellow")
    table.add_column("Qualité", style="blue")
    table.add_column("Erreurs", style="red")

    for model, data in results.items():
        if data["times"]:
            avg_time = sum(data["times"]) / len(data["times"])
            avg_words = sum(data["word_counts"]) / len(data["word_counts"])
            words_per_sec = avg_words / avg_time if avg_time > 0 else 0
            avg_quality = sum(data["quality_scores"]) / len(data["quality_scores"])

            table.add_row(
                model,
                f"{avg_time:.1f}s",
                f"{words_per_sec:.1f}",
                f"{avg_quality:.1%}",
                str(data["errors"]),
            )

    console.print(table)


@app.command()
def setup():
    """
    🔧 Assistant de configuration initiale
    """
    console.print(
        Panel.fit(
            "🔧 [bold blue]Assistant de Configuration[/bold blue]\n"
            "Cet assistant va vous aider à configurer le générateur",
            title="Setup",
        )
    )

    # Vérification des dépendances
    console.print("\n📦 Vérification des dépendances...")

    missing_deps = []
    try:
        import torch

        if torch.cuda.is_available():
            console.print("  ✅ PyTorch avec CUDA")
        else:
            console.print("  ⚠️ PyTorch sans CUDA (CPU seulement)")
    except ImportError:
        missing_deps.append("torch")

    try:
        import transformers

        console.print("  ✅ Transformers")
    except ImportError:
        missing_deps.append("transformers")

    try:
        import spacy

        console.print("  ✅ spaCy")
    except ImportError:
        console.print("  ⚠️ spaCy non installé (fonctionnalités limitées)")

    if missing_deps:
        console.print(f"\n❌ Dépendances manquantes: {', '.join(missing_deps)}")
        console.print("Installez-les avec: pip install -r requirements.txt")
        raise typer.Exit(1)

    # Vérification de la configuration
    issues = default_config.validate_configuration()
    if issues:
        console.print("\n⚠️ Problèmes de configuration:")
        for issue in issues:
            console.print(f"  • {issue}")
    else:
        console.print("\n✅ Configuration valide")

    # Suggestion de téléchargement de modèles
    available_models = default_config.list_available_models()
    if not available_models:
        console.print("\n📥 Aucun modèle local détecté")
        console.print("Utilisez 'pbg download-model <nom>' pour télécharger un modèle")
    else:
        console.print(f"\n✅ Modèles disponibles: {', '.join(available_models)}")

    console.print("\n🎉 Configuration terminée!")


@app.command()
def create_example():
    """
    📝 Crée un fichier d'exemple de notes
    """
    example_notes = {
        "titre": "Bilan Psychomoteur - Exemple",
        "sections": {
            "Identité & contexte": {
                "nom": "Marie D.",
                "age": "8 ans",
                "classe": "CE2",
                "contexte": "Demande de l'enseignante pour difficultés scolaires",
            },
            "Motif de la demande": "Difficultés en écriture et agitation en classe",
            "Anamnèse synthétique": {
                "grossesse": "Sans particularité",
                "développement": "Marche à 14 mois, langage normal",
                "antécédents": "Aucun antécédent médical",
            },
            "Évaluation psychomotrice": {
                "Tonus & posture": "Hypotonie axiale légère observée",
                "Motricité globale": "Coordination satisfaisante",
                "Motricité fine / praxies": "Difficultés de préhension et de précision",
                "Graphisme / écriture": "Écriture laborieuse, lettres mal formées",
            },
            "Tests / outils utilisés": [
                "M-ABC-2",
                "BHK (échelle d'évaluation de l'écriture)",
                "Observation clinique libre",
            ],
            "Analyse & synthèse": "Profil compatible avec un trouble de la coordination",
            "Conclusion & recommandations": "Rééducation psychomotrice recommandée",
            "Projet thérapeutique": "Séances hebdomadaires, travail graphomoteur",
            "Modalités & consentement": "Accord parental obtenu",
        },
    }

    output_file = Path("exemple_notes.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(example_notes, f, indent=2, ensure_ascii=False)

    console.print(f"✅ Fichier d'exemple créé: {output_file}")
    console.print("Utilisez: pbg generate exemple_notes.json")


def main():
    """Point d'entrée principal"""
    try:
        app()
    except KeyboardInterrupt:
        console.print("\n👋 Au revoir!")
        sys.exit(0)
    except Exception as e:
        console.print(f"\n💥 Erreur inattendue: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
