"""
Formatage intelligent et analyse de structure pour les documents de bilans
"""

import re
import statistics
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from config.settings import settings
from src.utils.logging import get_logger


@dataclass
class DocumentMetadata:
    """Métadonnées d'un document généré"""

    file_path: Path
    title: str
    theme: str
    creation_date: datetime
    word_count: int
    section_count: int
    quality_score: Optional[float] = None
    generation_time: Optional[float] = None
    model_used: Optional[str] = None


@dataclass
class SectionAnalysis:
    """Analyse d'une section"""

    name: str
    word_count: int
    sentence_count: int
    paragraph_count: int
    has_subsections: bool
    subsections: List[str]
    key_terms: List[str]
    readability_score: Optional[float] = None
    completeness_score: Optional[float] = None


class TextFormatter:
    """
    Formateur de texte intelligent pour bilans psychomoteurs

    Fonctionnalités :
    - Détection automatique de structure
    - Formatage clinique spécialisé
    - Mise en évidence des termes importants
    - Gestion des listes et énumérations
    - Normalisation typographique
    """

    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)

        # Patterns de formatage
        self.emphasis_patterns = {
            "important": r"\b(recommandé|conseillé|urgent|priorité|attention|important)\b",
            "clinical": r"\b(observe|constate|présente|manifeste|révèle|indique)\b",
            "quantitative": r"\b(\d+(?:[,.]\d+)?)\s*(ans?|mois|cm|kg|%|points?)\b",
            "negative": r"\b(difficultés?|troubles?|déficits?|retards?|problèmes?)\b",
            "positive": r"\b(progrès|amélioration|réussite|acquis|satisfaisant)\b",
        }

        # Termes techniques psychomoteurs
        self.technical_terms = {
            "tonus",
            "posture",
            "motricité",
            "praxie",
            "dyspraxie",
            "coordination",
            "équilibre",
            "latéralité",
            "dominance",
            "schéma corporel",
            "proprioception",
            "visuo-spatial",
            "graphomoteur",
            "attention",
            "exécutif",
            "inhibition",
            "régulation",
            "intégration",
            "sensoriel",
            "vestibulaire",
        }

        # Connecteurs logiques
        self.connectors = {
            "addition": ["par ailleurs", "de plus", "également", "aussi"],
            "opposition": ["cependant", "néanmoins", "toutefois", "en revanche"],
            "consequence": ["ainsi", "par conséquent", "de ce fait", "donc"],
            "temporal": ["lors de", "durant", "au cours de", "pendant"],
        }

    def format_clinical_text(self, text: str) -> str:
        """
        Formate un texte selon les standards cliniques

        Args:
            text: Texte à formater

        Returns:
            Texte formaté
        """
        if not text or not text.strip():
            return text

        # Nettoyage de base
        formatted = self._clean_basic_formatting(text)

        # Amélioration de la ponctuation clinique
        formatted = self._improve_clinical_punctuation(formatted)

        # Normalisation des termes techniques
        formatted = self._normalize_technical_terms(formatted)

        # Amélioration des transitions
        formatted = self._improve_transitions(formatted)

        # Formatage des données quantitatives
        formatted = self._format_quantitative_data(formatted)

        return formatted.strip()

    def detect_subsections(
        self, text: str, possible_subsections: List[str]
    ) -> Dict[str, str]:
        """
        Détecte automatiquement les sous-sections dans un texte

        Args:
            text: Texte à analyser
            possible_subsections: Liste des sous-sections possibles

        Returns:
            Dictionnaire {nom_sous_section: contenu}
        """
        subsections = {}
        current_section = None
        current_content = []

        lines = text.split("\n")

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Recherche de correspondance avec les sous-sections possibles
            matched_subsection = None
            for subsection in possible_subsections:
                # Correspondance exacte ou partielle
                if subsection.lower() in line.lower() or any(
                    word in line.lower() for word in subsection.lower().split()[:2]
                ):
                    # Vérification que c'est bien un titre (courte ligne, position, etc.)
                    if len(line) < 100 and (
                        ":" in line or line.endswith(".") or len(line.split()) <= 6
                    ):
                        matched_subsection = subsection
                        break

            if matched_subsection:
                # Sauvegarde de la section précédente
                if current_section and current_content:
                    subsections[current_section] = "\n".join(current_content).strip()

                # Nouvelle section
                current_section = matched_subsection
                current_content = []

                # Ajout du contenu restant de la ligne (après le titre)
                remaining_content = re.sub(
                    rf"{re.escape(matched_subsection)}[:.]?\s*",
                    "",
                    line,
                    flags=re.IGNORECASE,
                ).strip()

                if remaining_content:
                    current_content.append(remaining_content)
            else:
                # Ajout à la section courante
                if current_section:
                    current_content.append(line)
                else:
                    # Contenu avant toute sous-section
                    if "Introduction" not in subsections:
                        subsections["Introduction"] = ""
                    subsections["Introduction"] += line + "\n"


class StructureAnalyzer:
    """
    Analyseur de structure de document

    Analyse la structure logique et la qualité organisationnelle
    des bilans psychomoteurs générés.
    """

    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)

    def analyze_sections(self, sections_text: Dict[str, str]) -> Dict[str, Any]:
        """
        Analyse la structure complète du document

        Args:
            sections_text: Dictionnaire des sections

        Returns:
            Analyse structurelle complète
        """
        analysis = {
            "total_sections": len(sections_text),
            "total_words": 0,
            "total_sentences": 0,
            "sections": {},
            "structure_quality": 0.0,
            "completeness": 0.0,
            "balance_score": 0.0,
        }

        section_analyses = []

        # Analyse section par section
        for section_name, content in sections_text.items():
            if content and content.strip():
                section_analysis = self.analyze_section(section_name, content)
                analysis["sections"][section_name] = section_analysis.__dict__
                section_analyses.append(section_analysis)

                analysis["total_words"] += section_analysis.word_count
                analysis["total_sentences"] += section_analysis.sentence_count

        # Calculs globaux
        if section_analyses:
            # Score de complétude (sections présentes vs attendues)
            expected_sections = len(settings.section_order)
            analysis["completeness"] = len(section_analyses) / expected_sections

            # Score d'équilibre (distribution des longueurs)
            word_counts = [sa.word_count for sa in section_analyses]
            if len(word_counts) > 1:
                word_std = statistics.stdev(word_counts)
                word_mean = statistics.mean(word_counts)
                analysis["balance_score"] = max(0, 1 - (word_std / word_mean))
            else:
                analysis["balance_score"] = 1.0

            # Score de structure global
            analysis["structure_quality"] = (
                analysis["completeness"] * 0.6 + analysis["balance_score"] * 0.4
            )

        return analysis

    def analyze_section(self, section_name: str, content: str) -> SectionAnalysis:
        """
        Analyse une section individuelle

        Args:
            section_name: Nom de la section
            content: Contenu de la section

        Returns:
            Analyse de la section
        """

        # Métriques de base
        words = content.split()
        word_count = len(words)

        sentences = re.split(r"[.!?]+", content)
        sentences = [s.strip() for s in sentences if s.strip()]
        sentence_count = len(sentences)

        paragraphs = content.split("\n\n")
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        paragraph_count = len(paragraphs)

        # Détection de sous-sections
        has_subsections = False
        subsections = []

        if section_name == "Évaluation psychomotrice":
            detected_subsections = self._detect_eval_subsections(content)
            has_subsections = len(detected_subsections) > 0
            subsections = list(detected_subsections.keys())

        # Extraction des termes clés
        key_terms = self._extract_key_terms(content)

        # Score de complétude
        completeness_score = self._calculate_completeness(content, section_name)

        return SectionAnalysis(
            name=section_name,
            word_count=word_count,
            sentence_count=sentence_count,
            paragraph_count=paragraph_count,
            has_subsections=has_subsections,
            subsections=subsections,
            key_terms=key_terms,
            completeness_score=completeness_score,
        )

    def _detect_eval_subsections(self, content: str) -> Dict[str, str]:
        """Détecte les sous-sections d'évaluation"""
        formatter = TextFormatter()
        return formatter.detect_subsections(content, settings.eval_subsections)

    def _extract_key_terms(self, content: str) -> List[str]:
        """Extrait les termes clés d'une section"""

        # Termes techniques psychomoteurs
        technical_terms = {
            "tonus",
            "posture",
            "motricité",
            "praxie",
            "coordination",
            "équilibre",
            "latéralité",
            "schéma corporel",
            "proprioception",
            "visuo-spatial",
            "attention",
            "exécutif",
            "graphisme",
        }

        content_lower = content.lower()
        found_terms = []

        for term in technical_terms:
            if term in content_lower:
                found_terms.append(term)

        return found_terms

    def _calculate_completeness(self, content: str, section_name: str) -> float:
        """Calcule le score de complétude d'une section"""

        content_lower = content.lower()

        # Pénalité pour contenu manquant
        empty_indicators = ["non observé", "non renseigné", "non disponible"]
        empty_count = sum(
            content_lower.count(indicator) for indicator in empty_indicators
        )

        word_count = len(content.split())
        empty_ratio = empty_count / max(1, word_count)

        # Score basé sur la longueur attendue
        expected_lengths = {
            "Identité & contexte": 50,
            "Motif de la demande": 40,
            "Anamnèse synthétique": 80,
            "Évaluation psychomotrice": 200,
            "Tests / outils utilisés": 30,
            "Analyse & synthèse": 100,
            "Conclusion & recommandations": 80,
            "Projet thérapeutique": 60,
            "Modalités & consentement": 30,
        }

        expected_length = expected_lengths.get(section_name, 60)
        length_ratio = min(1.0, word_count / expected_length)

        # Score final
        completeness = length_ratio * (1 - empty_ratio * 2)
        return max(0.0, min(1.0, completeness))
