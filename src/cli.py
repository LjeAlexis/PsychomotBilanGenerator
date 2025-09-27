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

from config.settings import settings
from src.core.generator import PsychomotBilanGenerator
from src.utils.logging import setup_logging

app = typer.Typer(
    name="pbg", help="Générateur de Bilans Psychomoteurs avec IA", add_completion=False
)

console = Console()


@app.command()
def generate(
    notes_file: str = typer.Argument(help="Fichier JSON contenant les notes du bilan"),
    output: Optional[str] = typer.Option(
        None, "--output", "-o", help="Fichier de sortie (optionnel)"
    ),
    model: str = typer.Option(
        settings.default_model, "--model", "-m", help="Modèle à utiliser"
    ),
    temperature: float = typer.Option(
        0.3, "--temperature", "-t", help="Créativité du modèle (0.1-1.0)"
    ),
    quality: bool = typer.Option(
        True, "--quality/--no-quality", help="Contrôle qualité avancé"
    ),
    async_mode: bool = typer.Option(
        False, "--async", help="Mode génération asynchrone"
    ),
    retries: int = typer.Option(
        2, "--retries", "-r", help="Nombre de tentatives en cas d'échec"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Mode détaillé"),
):
    """
    Génère un bilan psychomoteur complet à partir de notes

    Exemple: pbg generate notes.json --model mistral --output mon_bilan.docx
    """

    notes_path = Path(notes_file)
    if not notes_path.exists():
        console.print(f"❌ Fichier non trouvé: {notes_file}")
        raise typer.Exit(1)

    if verbose:
        setup_logging("DEBUG")
    else:
        setup_logging("INFO")

    console.print(
        Panel.fit(
            f"Générateur de Bilans Psychomoteurs\n"
            f"Notes: {notes_file}\n"
            f"Modèle: {model}\n"
            f"Température: {temperature}\n"
            f"Mode: {'Asynchrone' if async_mode else 'Synchrone'}\n"
            f"Qualité: {'Activée' if quality else 'Désactivée'}",
            title="Configuration",
        )
    )

    # Test simple des notes en premier
    console.print("🔍 Validation du fichier de notes...")
    try:
        with open(notes_path, "r", encoding="utf-8") as f:
            notes_data = json.load(f)
        console.print("✅ Fichier JSON valide")
    except Exception as e:
        console.print(f"❌ Erreur fichier notes: {e}")
        raise typer.Exit(1)

    async def run_generation():
        """Fonction asynchrone principale pour la génération"""
        generator = None
        try:
            # Initialisation du générateur
            console.print("🔧 Initialisation du générateur...")
            generator = PsychomotBilanGenerator(
                model_name=model, enable_quality_checks=quality, enable_async=async_mode
            )

            console.print("⚡ Initialisation en cours...")
            await generator.initialize()
            console.print("✅ Générateur initialisé")

            # Vérification des notes avant génération
            console.print("📖 Vérification des notes...")

            if "sections" not in notes_data:
                console.print("❌ Erreur: Aucune section trouvée dans les notes")
                raise typer.Exit(1)

            sections_count = len(notes_data["sections"])
            console.print(f"📋 {sections_count} sections trouvées dans les notes")

            for section_name in notes_data["sections"]:
                console.print(f"  • {section_name}")

            # Génération du bilan
            console.print("🚀 Début de la génération...")
            output_path = await generator.generate_full_bilan(
                notes_file=notes_path,
                output_file=Path(output) if output else None,
                temperature=temperature,
                max_retries=retries,
            )

            console.print(f"\n✅ Bilan généré avec succès!")
            console.print(f"📄 Fichier: {output_path}")

            # Vérification que le fichier existe
            if Path(output_path).exists():
                size = Path(output_path).stat().st_size
                console.print(f"📊 Taille du fichier: {size} octets")
            else:
                console.print("⚠️ Attention: Le fichier n'a pas été créé")

            return str(output_path)

        except Exception as e:
            console.print(f"\n❌ Erreur lors de la génération: {e}")

            # Affichage de la stack trace en mode verbose
            if verbose:
                import traceback

                console.print("\n🔍 Détails de l'erreur:")
                console.print(traceback.format_exc())

            raise typer.Exit(1)

        finally:
            # Nettoyage approprié
            if generator:
                try:
                    console.print("🧹 Nettoyage des ressources...")
                    await generator.cleanup()
                    console.print("✅ Nettoyage terminé")
                except Exception as cleanup_error:
                    console.print(f"⚠️ Erreur lors du nettoyage: {cleanup_error}")

    try:
        result = asyncio.run(run_generation())
        console.print(f"\n🎉 Génération terminée: {result}")
    except KeyboardInterrupt:
        console.print("\n⚠️ Génération interrompue par l'utilisateur")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"\n💥 Erreur inattendue: {e}")
        if verbose:
            import traceback

            console.print(traceback.format_exc())
        raise typer.Exit(1)


@app.command()
def models():
    """
    Liste les modèles disponibles
    """
    config = settings

    table = Table(title="Modèles Disponibles")
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
    console.print(f"  • Modèle par défaut: {config.default_model}")


@app.command()
def validate(notes_file: str = typer.Argument(help="Fichier JSON des notes à valider")):
    """
    Valide la structure et le contenu d'un fichier de notes
    """
    notes_path = Path(notes_file)
    if not notes_path.exists():
        console.print(f"❌ Fichier non trouvé: {notes_file}")
        raise typer.Exit(1)

    try:
        with open(notes_path, "r", encoding="utf-8") as f:
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
        expected_sections = settings.section_order

        table = Table(title="🔍 Sections du Bilan")
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
            console.print("⚠️ Attention: Beaucoup de sections manquantes")
        elif completion_rate < 80:
            console.print("✅ Fichier valide mais incomplet")
        else:
            console.print("🎉 Fichier très complet!")

    except json.JSONDecodeError as e:
        console.print(f"❌ Erreur JSON: {e}")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"❌ Erreur: {e}")
        raise typer.Exit(1)


@app.command()
def config(
    show: bool = typer.Option(False, "--show", help="Afficher la configuration"),
    export: Optional[str] = typer.Option(
        None, "--export", help="Exporter vers un fichier"
    ),
    validate_config: bool = typer.Option(
        False, "--validate", help="Valider la configuration"
    ),
):
    """
    Gestion de la configuration
    """
    if show:
        console.print(Panel.fit(str(settings), title="Configuration Actuelle"))

    if validate_config:
        issues = settings.validate_configuration()
        if issues:
            console.print("❌ Problèmes de configuration:")
            for issue in issues:
                console.print(f"  • {issue}")
        else:
            console.print("✅ Configuration valide")

    if export:
        try:
            settings.export_config(Path(export))
            console.print(f"✅ Configuration exportée vers: {export}")
        except Exception as e:
            console.print(f"❌ Erreur d'export: {e}")


@app.command()
def setup():
    """
    Assistant de configuration initiale
    """
    console.print(
        Panel.fit(
            "Assistant de Configuration\n"
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
    issues = settings.validate_configuration()
    if issues:
        console.print("\n⚠️ Problèmes de configuration:")
        for issue in issues:
            console.print(f"  • {issue}")
    else:
        console.print("\n✅ Configuration valide")

    # Suggestion de téléchargement de modèles
    available_models = settings.list_available_models()
    if not available_models:
        console.print("\n🔥 Aucun modèle local détecté")
        console.print(
            "Téléchargez un modèle avec: python scripts/download_models.py mistral"
        )
    else:
        console.print(f"\n✅ Modèles disponibles: {', '.join(available_models)}")

    console.print("\n🎉 Configuration terminée!")


@app.command()
def create_example():
    """
    Crée un fichier d'exemple de notes
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
