"""
Générateur principal de bilans psychomoteurs avec contrôle qualité avancé
"""

import asyncio
import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

from config.prompts import prompt_builder
from config.settings import settings
from src.core.cache import CacheManager
from src.core.models import ModelManager
from src.output.docx_writer import EnhancedDocxWriter
from src.processing.quality_checker import QualityChecker
from src.processing.text_processor import TextProcessor
from src.processing.validator import BilanValidator
from src.utils.logging import get_logger


@dataclass
class GenerationMetrics:
    """Métriques détaillées de la génération"""

    total_sections: int = 0
    successful_sections: int = 0
    failed_sections: int = 0
    cached_sections: int = 0
    retried_sections: int = 0

    total_words: int = 0
    total_tokens_generated: int = 0
    total_time: float = 0.0

    average_quality_score: float = 0.0
    min_quality_score: float = 1.0
    max_quality_score: float = 0.0

    model_load_time: float = 0.0
    cache_hit_rate: float = 0.0

    section_times: Dict[str, float] = None
    section_quality_scores: Dict[str, float] = None

    def __post_init__(self):
        if self.section_times is None:
            self.section_times = {}
        if self.section_quality_scores is None:
            self.section_quality_scores = {}


class PsychomotBilanGenerator:
    """
    Générateur principal de bilans psychomoteurs avec IA

    Fonctionnalités principales :
    - Génération multi-modèles avec cache intelligent
    - Contrôle qualité automatique avec retry
    - Amélioration progressive du texte
    - Traitement asynchrone optionnel
    - Métriques détaillées et reporting
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        enable_quality_checks: bool = True,
        enable_async: bool = False,
        custom_config: Optional[Dict] = None,
    ):
        self.config = settings
        if custom_config:
            # Mise à jour de la config avec les paramètres personnalisés
            for key, value in custom_config.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)

        self.console = Console()
        self.logger = get_logger(self.__class__.__name__)

        # Configuration
        self.model_name = model_name or self.config.default_model
        self.enable_quality_checks = enable_quality_checks
        self.enable_async = enable_async

        # Composants principaux
        self.model_manager = ModelManager(self.config)
        self.cache_manager = CacheManager(self.config.cache_dir, self.config.cache)

        # Processeurs (initialisés seulement si nécessaire)
        self.quality_checker = None
        self.text_processor = None
        self.validator = None
        self.docx_writer = None

        # Métriques
        self.metrics = GenerationMetrics()

        # État
        self.is_initialized = False
        self.generation_id = None

    async def initialize(self) -> None:
        """Initialisation asynchrone du générateur"""
        if self.is_initialized:
            return

        start_time = time.time()
        self.generation_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        self.console.print(
            Panel.fit(
                f"🧠 [bold blue]Générateur de Bilans Psychomoteurs v{self.config.app_version}[/bold blue]\n"
                f"🤖 Modèle: {self.model_name}\n"
                f"🔍 Qualité: {'Activée' if self.enable_quality_checks else 'Désactivée'}\n"
                f"⚡ Mode: {'Asynchrone' if self.enable_async else 'Synchrone'}",
                title="Initialisation",
            )
        )

        # Chargement du modèle
        self.logger.info(f"Chargement du modèle: {self.model_name}")
        await self.model_manager.load_model(self.model_name)

        # Initialisation des composants de traitement
        if self.enable_quality_checks:
            self.quality_checker = QualityChecker(
                enable_spacy=self.config.quality.enable_spacy_analysis,
                enable_grammar_check=self.config.quality.enable_grammar_check,
            )

        self.text_processor = TextProcessor(
            enable_advanced_nlp=self.config.processing.enable_advanced_nlp
        )

        self.validator = BilanValidator()
        self.docx_writer = EnhancedDocxWriter()

        # Chargement du cache
        await self.cache_manager.load_cache()

        self.metrics.model_load_time = time.time() - start_time
        self.is_initialized = True

        self.console.print("✅ [bold green]Générateur initialisé[/bold green]")
        self.logger.info(
            f"Initialisation terminée en {self.metrics.model_load_time:.2f}s"
        )

    async def generate_section_with_quality_control(
        self,
        section_title: str,
        section_notes: Any,
        max_retries: Optional[int] = None,
        **generation_kwargs,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Génère une section avec contrôle qualité et retry automatique

        Args:
            section_title: Nom de la section
            section_notes: Données de la section
            max_retries: Nombre max de tentatives (défaut: config)
            **generation_kwargs: Paramètres de génération

        Returns:
            Tuple (texte_généré, métadonnées)
        """
        if not self.is_initialized:
            await self.initialize()

        max_retries = max_retries or self.config.quality.max_retries
        section_start_time = time.time()

        # Vérification du cache
        cache_key = self.cache_manager.get_cache_key(
            model_name=self.model_name,
            section_title=section_title,
            section_notes=section_notes,
            generation_params=generation_kwargs,
        )

        cached_result = await self.cache_manager.get(cache_key)
        if cached_result:
            self.metrics.cached_sections += 1
            self.logger.debug(f"Cache hit pour section: {section_title}")
            return cached_result["text"], cached_result["metadata"]

        # Variables pour le retry
        best_result = None
        best_score = 0.0
        attempts_data = []

        for attempt in range(max_retries + 1):
            try:
                self.logger.debug(
                    f"Génération section {section_title}, tentative {attempt + 1}"
                )

                # Construction du prompt
                system_prompt = prompt_builder.get_system_prompt(section_title)
                instruction = prompt_builder.build_section_instruction(
                    section_title=section_title,
                    section_notes=section_notes,
                    length_hint=self.config.length_hints.get(section_title),
                )

                # Paramètres de génération adaptés
                generation_config = prompt_builder.get_generation_config(
                    section_title, self.model_name
                )
                generation_config.update(generation_kwargs)

                # Génération du texte brut
                raw_text = await self.model_manager.generate_text(
                    system_prompt=system_prompt,
                    user_prompt=instruction,
                    **generation_config,
                )

                # Post-traitement du texte
                processed_text = self.text_processor.process_section(
                    raw_text, section_title
                )

                # Évaluation de la qualité
                quality_score = 1.0
                quality_metrics = None
                validation_result = None

                if self.quality_checker:
                    quality_metrics = self.quality_checker.evaluate_section(
                        processed_text, section_title
                    )
                    quality_score = quality_metrics.overall_score

                # Validation du contenu
                validation_result = self.validator.validate_section(
                    section_title, processed_text
                )

                # Score combiné (qualité - pénalités de validation)
                combined_score = quality_score * max(
                    0.1, 1.0 - len(validation_result.issues) * 0.1
                )

                # Stockage des données de cette tentative
                attempt_data = {
                    "attempt": attempt + 1,
                    "text": processed_text,
                    "quality_score": quality_score,
                    "combined_score": combined_score,
                    "quality_metrics": quality_metrics,
                    "validation_result": validation_result,
                    "word_count": len(processed_text.split()),
                    "generation_time": time.time() - section_start_time,
                }
                attempts_data.append(attempt_data)

                # Mise à jour du meilleur résultat
                if combined_score > best_score:
                    best_score = combined_score
                    best_result = attempt_data

                # Conditions d'arrêt anticipé
                min_threshold = self.config.quality.min_quality_threshold
                if (
                    quality_score >= min_threshold
                    and len(validation_result.issues) == 0
                    and len(processed_text.split()) >= 20
                ):
                    self.logger.debug(
                        f"Qualité suffisante atteinte à la tentative {attempt + 1}"
                    )
                    break

                # Log des problèmes pour retry
                if attempt < max_retries:
                    issues = validation_result.issues if validation_result else []
                    self.logger.debug(
                        f"Tentative {attempt + 1} insuffisante: "
                        f"qualité={quality_score:.2f}, problèmes={len(issues)}"
                    )

            except Exception as e:
                self.logger.error(
                    f"Erreur génération section {section_title}, tentative {attempt + 1}: {e}"
                )
                if attempt == max_retries:
                    raise RuntimeError(
                        f"Impossible de générer la section {section_title}: {e}"
                    )

        if not best_result:
            raise RuntimeError(f"Aucun résultat valide pour la section {section_title}")

        # Préparation des métadonnées finales
        section_time = time.time() - section_start_time
        metadata = {
            "generation_id": self.generation_id,
            "section_title": section_title,
            "model_name": self.model_name,
            "quality_score": best_result["quality_score"],
            "combined_score": best_result["combined_score"],
            "attempts_made": len(attempts_data),
            "best_attempt": best_result["attempt"],
            "generation_time": section_time,
            "word_count": best_result["word_count"],
            "timestamp": time.time(),
            "cache_key": cache_key,
        }

        # Mise en cache
        await self.cache_manager.set(
            cache_key, {"text": best_result["text"], "metadata": metadata}
        )

        # Mise à jour des métriques
        self.metrics.section_times[section_title] = section_time
        self.metrics.section_quality_scores[section_title] = best_result[
            "quality_score"
        ]

        if len(attempts_data) > 1:
            self.metrics.retried_sections += 1

        return best_result["text"], metadata

    async def generate_bilan_async(
        self,
        notes_data: Dict,
        sections_to_generate: Optional[List[str]] = None,
        **generation_kwargs,
    ) -> Dict[str, str]:
        """
        Génération asynchrone du bilan complet
        """
        sections = notes_data.get("sections", {})
        sections_list = sections_to_generate or self.config.section_order
        sections_to_process = [s for s in sections_list if s in sections]

        self.metrics.total_sections = len(sections_to_process)
        sections_text = {}

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            console=self.console,
            transient=False,
        ) as progress:
            task = progress.add_task(
                "Génération en cours...", total=self.metrics.total_sections
            )

            # Traitement des sections
            for section_name in sections_to_process:
                start_time = time.time()
                progress.update(task, description=f"📝 {section_name}")

                try:
                    text, metadata = await self.generate_section_with_quality_control(
                        section_title=section_name,
                        section_notes=sections[section_name],
                        **generation_kwargs,
                    )

                    sections_text[section_name] = text
                    self.metrics.successful_sections += 1
                    self.metrics.total_words += metadata["word_count"]

                    # Affichage du résultat
                    quality_info = f"Q:{metadata['quality_score']:.2f}"
                    if metadata["attempts_made"] > 1:
                        quality_info += f" (T{metadata['attempts_made']})"
                    if metadata.get("cached", False):
                        quality_info += " 💾"

                    progress.console.print(
                        f"  ✅ {section_name} - {metadata['word_count']} mots - {quality_info}",
                        style="green",
                    )

                except Exception as e:
                    self.logger.error(f"Erreur section {section_name}: {e}")
                    sections_text[section_name] = f"[Erreur lors de la génération: {e}]"
                    self.metrics.failed_sections += 1

                    progress.console.print(f"  ❌ {section_name} - Échec", style="red")

                progress.advance(task)

                # Pause thermique
                if (
                    self.config.processing.thermal_pause > 0
                    and section_name != sections_to_process[-1]
                ):  # Pas de pause après la dernière
                    await asyncio.sleep(self.config.processing.thermal_pause)

        return sections_text

    async def generate_bilan_sync(
        self,
        notes_data: Dict,
        sections_to_generate: Optional[List[str]] = None,
        **generation_kwargs,
    ) -> Dict[str, str]:
        """Version synchrone corrigée avec validation"""
        sections = notes_data.get("sections", {})
        sections_list = sections_to_generate or self.config.section_order
        sections_text = {}

        self.logger.info(f"Début génération de {len(sections_list)} sections")

        for section_name in sections_list:
            if section_name not in sections:
                self.logger.warning(f"Section manquante dans les notes: {section_name}")
                continue

            self.console.print(f"📝 Génération: {section_name}")

            try:
                # Utilise await au lieu d'asyncio.run
                text, metadata = await self.generate_section_with_quality_control(
                    section_title=section_name,
                    section_notes=sections[section_name],
                    **generation_kwargs,
                )

                # ✅ Validation du contenu généré
                if text and isinstance(text, str) and len(text.strip()) > 0:
                    sections_text[section_name] = text.strip()
                    self.metrics.successful_sections += 1
                    self.logger.info(
                        f"Section {section_name} générée: {len(text)} caractères"
                    )
                else:
                    # Fallback si le texte est vide
                    sections_text[section_name] = (
                        f"[Section {section_name} en cours de développement]"
                    )
                    self.logger.warning(
                        f"Section {section_name} vide, fallback appliqué"
                    )

            except Exception as e:
                self.logger.error(f"Erreur section {section_name}: {e}")
                # Fallback en cas d'erreur
                sections_text[section_name] = (
                    f"[Erreur lors de la génération de la section {section_name}: {str(e)}]"
                )
                self.metrics.failed_sections += 1

        self.logger.info(f"Génération terminée: {len(sections_text)} sections créées")

        # ✅ Validation finale avant retour
        if not sections_text:
            self.logger.error("Aucune section générée !")
            # Retourne au moins une section par défaut
            sections_text = {
                "Erreur": "Aucune section n'a pu être générée. Vérifiez vos notes d'entrée."
            }

        return sections_text

    async def generate_full_bilan(
        self,
        notes_file: Path,
        output_file: Optional[Path] = None,
        format_type: str = "docx",
        **generation_kwargs,
    ) -> Path:
        """Pipeline complet avec validation renforcée"""
        if not self.is_initialized:
            await self.initialize()

        generation_start_time = time.time()

        # Chargement des notes
        self.logger.info(f"Chargement des notes: {notes_file}")
        with open(notes_file, "r", encoding="utf-8") as f:
            notes_data = json.load(f)

        titre = notes_data.get("titre", "Bilan Psychomoteur")

        self.console.print(
            Panel.fit(
                f"🎯 [bold blue]Génération du bilan[/bold blue]\n"
                f"📋 Titre: {titre}\n"
                f"📁 Source: {notes_file.name}\n"
                f"🤖 Modèle: {self.model_name}",
                title="Bilan en cours",
            )
        )

        # ✅ Validation des données d'entrée
        if "sections" not in notes_data:
            raise ValueError("Le fichier de notes ne contient pas de sections")

        if not notes_data["sections"]:
            raise ValueError("Les sections du fichier de notes sont vides")

        # Génération des sections
        if self.enable_async:
            sections_text = await self.generate_bilan_async(
                notes_data, **generation_kwargs
            )
        else:
            sections_text = await self.generate_bilan_sync(
                notes_data, **generation_kwargs
            )

        # ✅ Validation des sections générées
        if not sections_text:
            raise RuntimeError("Aucune section n'a été générée")

        # Filtre les sections vides ou None
        valid_sections = {}
        for section_name, content in sections_text.items():
            if content and isinstance(content, str) and len(content.strip()) > 0:
                valid_sections[section_name] = content.strip()
            else:
                self.logger.warning(
                    f"Section {section_name} ignorée (vide ou invalide)"
                )

        if not valid_sections:
            raise RuntimeError("Toutes les sections générées sont vides ou invalides")

        sections_text = valid_sections
        self.logger.info(f"Sections valides: {list(sections_text.keys())}")

        # Contrôle qualité global
        if self.quality_checker:
            self.logger.info("Évaluation qualité globale")
            global_quality = self.quality_checker.evaluate_full_bilan(sections_text)
            self.metrics.average_quality_score = global_quality.overall_score

            if global_quality.issues:
                self.console.print("⚠️  [yellow]Problèmes qualité détectés:[/yellow]")
                for issue in global_quality.issues[:5]:
                    self.console.print(f"  • {issue}", style="yellow")

        # Génération du document final
        if not output_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M")
            output_file = self.config.output_dir / f"bilan_{timestamp}.{format_type}"

        self.logger.info(f"Génération du document: {output_file}")

        if format_type == "docx":
            # ✅ Vérification finale avant création du document
            if not all(isinstance(content, str) for content in sections_text.values()):
                raise ValueError(
                    "Toutes les sections doivent être des chaînes de caractères"
                )

            await self.docx_writer.build_document(
                sections_text=sections_text,
                output_path=output_file,
                title=titre,
                metadata={
                    "generation_id": self.generation_id,
                    "model_name": self.model_name,
                    "quality_score": self.metrics.average_quality_score,
                    "generation_date": datetime.now().isoformat(),
                },
            )
        else:
            raise ValueError(f"Format non supporté: {format_type}")

        # Finalisation des métriques
        self.metrics.total_time = time.time() - generation_start_time
        self.metrics.cache_hit_rate = self.metrics.cached_sections / max(
            1, self.metrics.total_sections
        )

        # Sauvegarde du cache
        await self.cache_manager.save_cache()

        # Affichage du résumé
        self._print_generation_summary(output_file)

        self.logger.info(f"Génération terminée: {output_file}")

        return output_file

    def _print_generation_summary(self, output_file: Path) -> None:
        """Affiche un résumé détaillé de la génération"""
        self.console.print("\n" + "=" * 70, style="blue")
        self.console.print(
            "📊 RÉSUMÉ DE GÉNÉRATION", style="bold blue", justify="center"
        )
        self.console.print("=" * 70, style="blue")

        # Informations de base
        self.console.print(f"📁 Fichier généré: [bold]{output_file}[/bold]")
        self.console.print(f"🆔 ID génération: {self.generation_id}")
        self.console.print(f"🤖 Modèle utilisé: [bold]{self.model_name}[/bold]")

        # Statistiques sections
        self.console.print(f"\n📑 Sections:")
        self.console.print(
            f"  • Générées avec succès: [green]{self.metrics.successful_sections}[/green]"
        )
        self.console.print(f"  • Échecs: [red]{self.metrics.failed_sections}[/red]")
        self.console.print(
            f"  • Depuis le cache: [blue]{self.metrics.cached_sections}[/blue]"
        )
        self.console.print(
            f"  • Nécessitant retry: [yellow]{self.metrics.retried_sections}[/yellow]"
        )

        # Métriques de contenu
        self.console.print(f"\n📖 Contenu:")
        self.console.print(
            f"  • Mots totaux: [bold]{self.metrics.total_words:,}[/bold]"
        )
        if self.metrics.total_tokens_generated > 0:
            self.console.print(
                f"  • Tokens générés: {self.metrics.total_tokens_generated:,}"
            )

        # Métriques de qualité
        if self.quality_checker and self.metrics.average_quality_score > 0:
            self.console.print(f"\n⭐ Qualité:")
            self.console.print(
                f"  • Score moyen: [bold]{self.metrics.average_quality_score:.1%}[/bold]"
            )

            if self.metrics.section_quality_scores:
                min_score = min(self.metrics.section_quality_scores.values())
                max_score = max(self.metrics.section_quality_scores.values())
                self.console.print(
                    f"  • Score min/max: {min_score:.1%} / {max_score:.1%}"
                )

        # Métriques de performance
        self.console.print(f"\n⏱️  Performance:")
        self.console.print(
            f"  • Temps total: [bold]{self.metrics.total_time:.1f}s[/bold]"
        )
        self.console.print(
            f"  • Chargement modèle: {self.metrics.model_load_time:.1f}s"
        )
        self.console.print(f"  • Taux cache: {self.metrics.cache_hit_rate:.1%}")

        # Vitesse de génération
        if self.metrics.total_time > 0 and self.metrics.total_words > 0:
            words_per_second = self.metrics.total_words / self.metrics.total_time
            self.console.print(
                f"  • Vitesse: [bold]{words_per_second:.1f} mots/s[/bold]"
            )

        self.console.print("=" * 70, style="blue")

    def get_detailed_metrics(self) -> Dict[str, Any]:
        """Retourne les métriques détaillées pour export"""
        return {
            "generation_id": self.generation_id,
            "model_name": self.model_name,
            "timestamp": datetime.now().isoformat(),
            "config": {
                "quality_checks": self.enable_quality_checks,
                "async_mode": self.enable_async,
                "thermal_pause": self.config.processing.thermal_pause,
            },
            "metrics": {
                "sections": {
                    "total": self.metrics.total_sections,
                    "successful": self.metrics.successful_sections,
                    "failed": self.metrics.failed_sections,
                    "cached": self.metrics.cached_sections,
                    "retried": self.metrics.retried_sections,
                },
                "content": {
                    "total_words": self.metrics.total_words,
                    "total_tokens": self.metrics.total_tokens_generated,
                },
                "quality": {
                    "average_score": self.metrics.average_quality_score,
                    "section_scores": self.metrics.section_quality_scores,
                },
                "performance": {
                    "total_time": self.metrics.total_time,
                    "model_load_time": self.metrics.model_load_time,
                    "cache_hit_rate": self.metrics.cache_hit_rate,
                    "section_times": self.metrics.section_times,
                },
            },
        }

    async def cleanup(self) -> None:
        """Nettoyage des ressources"""
        if self.model_manager:
            await self.model_manager.cleanup()

        if self.cache_manager:
            await self.cache_manager.save_cache()

        self.logger.info("Nettoyage terminé")

    def __del__(self):
        """Nettoyage automatique à la destruction"""
        if hasattr(self, "model_manager") and self.model_manager:
            try:
                asyncio.run(self.model_manager.cleanup())
            except:
                pass  # En cas d'erreur lors du nettoyage


# Fonction helper pour usage simple
async def generate_bilan_simple(
    notes_file: str,
    model: str = "mistral",
    output_file: Optional[str] = None,
    temperature: float = 0.3,
    enable_quality: bool = True,
    async_mode: bool = False,
) -> str:
    """
    Interface simplifiée pour génération de bilan

    Args:
        notes_file: Fichier JSON des notes
        model: Nom du modèle à utiliser
        output_file: Fichier de sortie (optionnel)
        temperature: Créativité du modèle
        enable_quality: Activer le contrôle qualité
        async_mode: Mode asynchrone

    Returns:
        Chemin du fichier généré
    """
    generator = PsychomotBilanGenerator(
        model_name=model, enable_quality_checks=enable_quality, enable_async=async_mode
    )

    try:
        await generator.initialize()

        output_path = await generator.generate_full_bilan(
            notes_file=Path(notes_file),
            output_file=Path(output_file) if output_file else None,
            temperature=temperature,
        )

        return str(output_path)

    finally:
        await generator.cleanup()
