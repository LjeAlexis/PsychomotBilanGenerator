"""
Système de validation du contenu des bilans psychomoteurs
"""

import re
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from config.settings import settings
from src.utils.logging import get_logger


class ValidationLevel(Enum):
    """Niveaux de sévérité des problèmes de validation"""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ValidationIssue:
    """Représente un problème de validation"""

    level: ValidationLevel
    category: str
    message: str
    section: Optional[str] = None
    position: Optional[int] = None
    suggestion: Optional[str] = None
    auto_fixable: bool = False


@dataclass
class ValidationResult:
    """Résultat de validation d'une section ou document"""

    is_valid: bool
    score: float  # 0.0 à 1.0
    issues: List[ValidationIssue]
    word_count: int
    sections_validated: int

    @property
    def has_errors(self) -> bool:
        return any(
            issue.level in [ValidationLevel.ERROR, ValidationLevel.CRITICAL]
            for issue in self.issues
        )

    @property
    def has_warnings(self) -> bool:
        return any(issue.level == ValidationLevel.WARNING for issue in self.issues)

    def get_issues_by_level(self, level: ValidationLevel) -> List[ValidationIssue]:
        return [issue for issue in self.issues if issue.level == level]


class ContentValidator:
    """Validateur de contenu spécialisé pour les bilans psychomoteurs"""

    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)

        # Dictionnaires de validation
        self._setup_validation_rules()
        self._setup_medical_terminology()
        self._setup_forbidden_patterns()

    def _setup_validation_rules(self):
        """Configure les règles de validation"""

        # Longueurs minimales par section (en mots)
        self.min_word_counts = {
            "Identité & contexte": 20,
            "Motif de la demande": 15,
            "Anamnèse synthétique": 40,
            "Évaluation psychomotrice": 80,
            "Tests / outils utilisés": 10,
            "Analyse & synthèse": 50,
            "Conclusion & recommandations": 40,
            "Projet thérapeutique": 30,
            "Modalités & consentement": 15,
        }

        # Longueurs maximales (pour détecter le bavardage)
        self.max_word_counts = {
            "Identité & contexte": 150,
            "Motif de la demande": 100,
            "Anamnèse synthétique": 300,
            "Évaluation psychomotrice": 800,
            "Tests / outils utilisés": 80,
            "Analyse & synthèse": 250,
            "Conclusion & recommandations": 200,
            "Projet thérapeutique": 150,
            "Modalités & consentement": 80,
        }

        # Éléments obligatoires par section
        self.required_elements = {
            "Identité & contexte": ["âge", "contexte"],
            "Motif de la demande": ["demande", "difficultés"],
            "Évaluation psychomotrice": ["observation", "évaluation"],
            "Conclusion & recommandations": ["conclusion", "recommandation"],
            "Projet thérapeutique": ["objectif", "modalité"],
        }

    def _setup_medical_terminology(self):
        """Configure la terminologie médicale acceptée"""

        # Termes psychomoteurs valides
        self.valid_terms = {
            # Domaines psychomoteurs
            "tonus",
            "hypotonie",
            "hypertonie",
            "régulation tonique",
            "posture",
            "équilibre statique",
            "équilibre dynamique",
            "motricité globale",
            "motricité fine",
            "coordination",
            "dissociation",
            "praxie",
            "dyspraxie",
            "apraxie",
            "schéma corporel",
            "image du corps",
            "conscience corporelle",
            "latéralité",
            "dominance",
            "latéralisation",
            "proprioception",
            "exteroception",
            "interoception",
            "vestibulaire",
            "sensoriel",
            "sensori-moteur",
            "visuo-spatial",
            "visuo-constructif",
            "visuo-moteur",
            "attention",
            "concentration",
            "fonctions exécutives",
            "inhibition",
            "flexibilité",
            "mémoire de travail",
            "graphisme",
            "graphomoteur",
            "écriture",
            "préhension",
            # Observations cliniques
            "observe",
            "constate",
            "note",
            "présente",
            "manifeste",
            "révèle",
            "indique",
            "suggère",
            "évoque",
            "témoigne",
            # Évaluations
            "évalue",
            "analyse",
            "mesure",
            "teste",
            "examine",
            "apprécie",
            "quantifie",
            "objective",
            # Recommandations
            "recommande",
            "conseille",
            "préconise",
            "suggère",
            "propose",
            "indique",
            "nécessite",
            "requiert",
        }

        # Synonymes et variantes acceptées
        self.term_variants = {
            "enfant": ["patient", "sujet", "jeune"],
            "difficultés": ["troubles", "déficits", "problèmes"],
            "amélioration": ["progrès", "évolution positive", "progression"],
            "observation": ["constatation", "remarque", "notation"],
        }

    def _setup_forbidden_patterns(self):
        """Configure les patterns interdits"""

        # Expressions à éviter (non professionnelles)
        self.forbidden_expressions = [
            r"\bnormal\b",  # Préférer "dans la norme"
            r"\banormal\b",  # Préférer "atypique"
            r"\bbizarre\b",
            r"\bétrange\b",
            r"\bdrôle\b",
            r"\bmarrant\b",
        ]

        # Hallucinations typiques des LLM
        self.hallucination_patterns = [
            r"selon les études récentes",
            r"il est prouvé que",
            r"toutes les recherches montrent",
            r"de nombreux experts",
            r"statistiquement parlant",
            r"\d+\s*%\s*des\s*(enfants|patients)",  # Pourcentages inventés
            r"selon la littérature",
            r"il est bien établi que",
        ]

        # Expressions trop vagues
        self.vague_expressions = [
            r"assez bien",
            r"plutôt bon",
            r"pas trop mal",
            r"correct",
            r"moyen",
            r"comme il faut",
        ]


class BilanValidator:
    """
    Validateur principal pour les bilans psychomoteurs

    Effectue une validation complète du contenu généré pour s'assurer
    de la qualité professionnelle et de la conformité clinique.
    """

    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)
        self.content_validator = ContentValidator()

        # Statistiques de validation
        self.validation_stats = {
            "total_validations": 0,
            "sections_validated": 0,
            "issues_found": 0,
            "auto_fixes_applied": 0,
        }

    def validate_section(
        self, section_name: str, content: str, strict_mode: bool = False
    ) -> ValidationResult:
        """
        Valide une section individuelle

        Args:
            section_name: Nom de la section
            content: Contenu à valider
            strict_mode: Mode strict avec validation renforcée

        Returns:
            Résultat de validation
        """
        self.logger.debug(f"Validation section: {section_name}")

        issues = []
        word_count = len(content.split()) if content else 0

        # Validation de base
        issues.extend(self._validate_basic_structure(section_name, content))

        # Validation du contenu
        issues.extend(self._validate_content_quality(section_name, content))

        # Validation terminologique
        issues.extend(self._validate_terminology(section_name, content))

        # Validation spécifique par section
        issues.extend(self._validate_section_specific(section_name, content))

        # Validation stricte si demandée
        if strict_mode:
            issues.extend(self._validate_strict_rules(section_name, content))

        # Calcul du score
        score = self._calculate_validation_score(issues, word_count)

        # Détermination de la validité
        is_valid = not any(
            issue.level in [ValidationLevel.ERROR, ValidationLevel.CRITICAL]
            for issue in issues
        )

        # Mise à jour des statistiques
        self.validation_stats["total_validations"] += 1
        self.validation_stats["sections_validated"] += 1
        self.validation_stats["issues_found"] += len(issues)

        return ValidationResult(
            is_valid=is_valid,
            score=score,
            issues=issues,
            word_count=word_count,
            sections_validated=1,
        )

    def validate_full_bilan(
        self, sections_text: Dict[str, str], strict_mode: bool = False
    ) -> ValidationResult:
        """
        Valide un bilan complet

        Args:
            sections_text: Dictionnaire des sections
            strict_mode: Mode strict avec validation renforcée

        Returns:
            Résultat de validation globale
        """
        self.logger.info("Validation du bilan complet")

        all_issues = []
        total_words = 0
        sections_count = 0
        section_scores = []

        # Validation section par section
        for section_name, content in sections_text.items():
            if content and content.strip():
                result = self.validate_section(section_name, content, strict_mode)

                all_issues.extend(result.issues)
                total_words += result.word_count
                sections_count += 1
                section_scores.append(result.score)

        # Validation de la cohérence globale
        all_issues.extend(self._validate_global_coherence(sections_text))

        # Validation de la complétude
        all_issues.extend(self._validate_completeness(sections_text))

        # Calcul du score global
        if section_scores:
            average_score = sum(section_scores) / len(section_scores)
            # Pénalité pour problèmes globaux
            global_penalty = len([i for i in all_issues if i.section is None]) * 0.05
            global_score = max(0.0, average_score - global_penalty)
        else:
            global_score = 0.0

        # Détermination de la validité globale
        is_valid = not any(
            issue.level in [ValidationLevel.ERROR, ValidationLevel.CRITICAL]
            for issue in all_issues
        )

        return ValidationResult(
            is_valid=is_valid,
            score=global_score,
            issues=all_issues,
            word_count=total_words,
            sections_validated=sections_count,
        )

    def _validate_basic_structure(
        self, section_name: str, content: str
    ) -> List[ValidationIssue]:
        """Validation de la structure de base"""
        issues = []

        if not content or not content.strip():
            issues.append(
                ValidationIssue(
                    level=ValidationLevel.ERROR,
                    category="structure",
                    message="Section vide ou manquante",
                    section=section_name,
                    suggestion="Générer du contenu pour cette section",
                )
            )
            return issues

        word_count = len(content.split())

        # Vérification longueur minimale
        min_words = self.content_validator.min_word_counts.get(section_name, 10)
        if word_count < min_words:
            issues.append(
                ValidationIssue(
                    level=ValidationLevel.WARNING,
                    category="longueur",
                    message=f"Section trop courte ({word_count} mots, minimum: {min_words})",
                    section=section_name,
                    suggestion=f"Développer le contenu (ajouter ~{min_words - word_count} mots)",
                )
            )

        # Vérification longueur maximale
        max_words = self.content_validator.max_word_counts.get(section_name, 500)
        if word_count > max_words:
            issues.append(
                ValidationIssue(
                    level=ValidationLevel.WARNING,
                    category="longueur",
                    message=f"Section très longue ({word_count} mots, maximum suggéré: {max_words})",
                    section=section_name,
                    suggestion="Synthétiser le contenu en gardant l'essentiel",
                )
            )

        # Vérification des phrases
        sentences = re.split(r"[.!?]+", content)
        valid_sentences = [s.strip() for s in sentences if s.strip()]

        if len(valid_sentences) < 2 and word_count > 20:
            issues.append(
                ValidationIssue(
                    level=ValidationLevel.WARNING,
                    category="structure",
                    message="Manque de ponctuation (phrases trop longues)",
                    section=section_name,
                    suggestion="Diviser en phrases plus courtes",
                )
            )

        return issues

    def _validate_content_quality(
        self, section_name: str, content: str
    ) -> List[ValidationIssue]:
        """Validation de la qualité du contenu"""
        issues = []

        # Détection d'expressions interdites
        for pattern in self.content_validator.forbidden_expressions:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                issues.append(
                    ValidationIssue(
                        level=ValidationLevel.WARNING,
                        category="expression",
                        message=f"Expression non professionnelle: '{match.group()}'",
                        section=section_name,
                        position=match.start(),
                        suggestion="Utiliser une formulation plus professionnelle",
                    )
                )

        # Détection d'hallucinations
        for pattern in self.content_validator.hallucination_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                issues.append(
                    ValidationIssue(
                        level=ValidationLevel.ERROR,
                        category="hallucination",
                        message=f"Possible hallucination détectée: '{match.group()}'",
                        section=section_name,
                        position=match.start(),
                        suggestion="Remplacer par des observations factuelles",
                    )
                )

        # Détection d'expressions vagues
        for pattern in self.content_validator.vague_expressions:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                issues.append(
                    ValidationIssue(
                        level=ValidationLevel.INFO,
                        category="précision",
                        message=f"Expression vague: '{match.group()}'",
                        section=section_name,
                        position=match.start(),
                        suggestion="Préciser l'observation ou la mesure",
                    )
                )

        # Vérification des répétitions
        words = re.findall(r"\b\w+\b", content.lower())
        if len(words) > 0:
            word_freq = {}
            for word in words:
                if len(word) > 4:  # Ignorer les mots très courts
                    word_freq[word] = word_freq.get(word, 0) + 1

            # Signaler les mots très répétés
            total_words = len(words)
            for word, count in word_freq.items():
                if count > 3 and count / total_words > 0.05:  # Plus de 5% du texte
                    issues.append(
                        ValidationIssue(
                            level=ValidationLevel.INFO,
                            category="répétition",
                            message=f"Mot très répété: '{word}' ({count} fois)",
                            section=section_name,
                            suggestion="Varier le vocabulaire",
                        )
                    )

        return issues

    def _validate_terminology(
        self, section_name: str, content: str
    ) -> List[ValidationIssue]:
        """Validation de la terminologie"""
        issues = []

        # Vérification de la présence de termes techniques appropriés
        content_lower = content.lower()

        if section_name == "Évaluation psychomotrice":
            # Cette section doit contenir des termes psychomoteurs
            psychomotor_terms = [
                "tonus",
                "motricité",
                "coordination",
                "équilibre",
                "praxie",
                "schéma corporel",
                "latéralité",
            ]

            found_terms = [term for term in psychomotor_terms if term in content_lower]

            if len(found_terms) < 2:
                issues.append(
                    ValidationIssue(
                        level=ValidationLevel.WARNING,
                        category="terminologie",
                        message="Peu de termes psychomoteurs spécialisés détectés",
                        section=section_name,
                        suggestion="Utiliser un vocabulaire technique approprié",
                    )
                )

        # Vérification des observations cliniques
        observation_verbs = ["observe", "constate", "note", "présente", "manifeste"]
        has_observation_verb = any(verb in content_lower for verb in observation_verbs)

        if (
            section_name in ["Évaluation psychomotrice", "Analyse & synthèse"]
            and not has_observation_verb
        ):
            issues.append(
                ValidationIssue(
                    level=ValidationLevel.INFO,
                    category="style",
                    message="Aucun verbe d'observation clinique détecté",
                    section=section_name,
                    suggestion="Utiliser des verbes comme 'observe', 'constate', 'présente'",
                )
            )

        return issues

    def _validate_section_specific(
        self, section_name: str, content: str
    ) -> List[ValidationIssue]:
        """Validation spécifique par type de section"""
        issues = []
        content_lower = content.lower()

        if section_name == "Identité & contexte":
            # Doit mentionner l'âge
            if not re.search(r"\b\d+\s*ans?\b", content):
                issues.append(
                    ValidationIssue(
                        level=ValidationLevel.WARNING,
                        category="contenu",
                        message="Âge du patient non mentionné",
                        section=section_name,
                        suggestion="Préciser l'âge du patient",
                    )
                )

        elif section_name == "Motif de la demande":
            # Doit mentionner qui fait la demande
            requesters = ["enseignant", "parent", "médecin", "famille", "école"]
            has_requester = any(req in content_lower for req in requesters)

            if not has_requester:
                issues.append(
                    ValidationIssue(
                        level=ValidationLevel.INFO,
                        category="contenu",
                        message="Origine de la demande non précisée",
                        section=section_name,
                        suggestion="Mentionner qui formule la demande",
                    )
                )

        elif section_name == "Conclusion & recommandations":
            # Doit contenir des recommandations
            recommendation_terms = ["recommand", "conseil", "suggest", "préconise"]
            has_recommendation = any(
                term in content_lower for term in recommendation_terms
            )

            if not has_recommendation:
                issues.append(
                    ValidationIssue(
                        level=ValidationLevel.WARNING,
                        category="contenu",
                        message="Aucune recommandation explicite trouvée",
                        section=section_name,
                        suggestion="Formuler des recommandations concrètes",
                    )
                )

        elif section_name == "Tests / outils utilisés":
            # Vérifier format liste si plusieurs tests
            lines = content.split("\n")
            test_indicators = ["-", "•", "*", "1.", "2.", "M-ABC", "BHK", "NEPSY"]

            if len(lines) > 1:
                has_list_format = any(
                    any(indicator in line for indicator in test_indicators)
                    for line in lines
                )

                if not has_list_format:
                    issues.append(
                        ValidationIssue(
                            level=ValidationLevel.INFO,
                            category="format",
                            message="Liste des tests pourrait être mieux structurée",
                            section=section_name,
                            suggestion="Utiliser une liste à puces pour les tests",
                        )
                    )

        return issues

    def _validate_strict_rules(
        self, section_name: str, content: str
    ) -> List[ValidationIssue]:
        """Validation en mode strict"""
        issues = []

        # En mode strict, on est plus exigeant
        word_count = len(content.split())

        # Longueurs minimales plus strictes
        strict_minimums = {
            "Évaluation psychomotrice": 150,
            "Analyse & synthèse": 80,
            "Conclusion & recommandations": 60,
        }

        strict_min = strict_minimums.get(section_name)
        if strict_min and word_count < strict_min:
            issues.append(
                ValidationIssue(
                    level=ValidationLevel.ERROR,
                    category="longueur_stricte",
                    message=f"Section insuffisamment développée en mode strict ({word_count}/{strict_min} mots)",
                    section=section_name,
                    suggestion="Développer davantage le contenu",
                )
            )

        # Vérification plus stricte du contenu "Non observé"
        non_observe_count = content.lower().count(
            "non observé"
        ) + content.lower().count("non renseigné")
        if non_observe_count > 1:
            issues.append(
                ValidationIssue(
                    level=ValidationLevel.ERROR,
                    category="contenu_strict",
                    message=f"Trop de mentions 'Non observé' en mode strict ({non_observe_count})",
                    section=section_name,
                    suggestion="Remplacer par des observations concrètes",
                )
            )

        return issues

    def _validate_global_coherence(
        self, sections_text: Dict[str, str]
    ) -> List[ValidationIssue]:
        """Validation de la cohérence globale entre sections"""
        issues = []

        # Extraction d'informations clés
        patient_age = None
        patient_level = None

        # Recherche de l'âge dans l'identité
        identity_content = sections_text.get("Identité & contexte", "")
        age_match = re.search(r"(\d+)\s*ans?", identity_content)
        if age_match:
            patient_age = int(age_match.group(1))

        # Recherche du niveau scolaire
        level_match = re.search(
            r"(CP|CE1|CE2|CM1|CM2|6ème|5ème|4ème|3ème)", identity_content
        )
        if level_match:
            patient_level = level_match.group(1)

        # Vérification cohérence âge/niveau
        if patient_age and patient_level:
            expected_ages = {
                "CP": (6, 7),
                "CE1": (7, 8),
                "CE2": (8, 9),
                "CM1": (9, 10),
                "CM2": (10, 11),
                "6ème": (11, 12),
                "5ème": (12, 13),
                "4ème": (13, 14),
                "3ème": (14, 15),
            }

            if patient_level in expected_ages:
                min_age, max_age = expected_ages[patient_level]
                if not (min_age <= patient_age <= max_age + 2):  # Tolérance de 2 ans
                    issues.append(
                        ValidationIssue(
                            level=ValidationLevel.WARNING,
                            category="cohérence",
                            message=f"Incohérence âge ({patient_age} ans) / niveau scolaire ({patient_level})",
                            suggestion="Vérifier la cohérence âge/scolarité",
                        )
                    )

        # Vérification cohérence motif/évaluation
        motif = sections_text.get("Motif de la demande", "").lower()
        evaluation = sections_text.get("Évaluation psychomotrice", "").lower()

        if "graphisme" in motif or "écriture" in motif:
            if "graphisme" not in evaluation and "écriture" not in evaluation:
                issues.append(
                    ValidationIssue(
                        level=ValidationLevel.WARNING,
                        category="cohérence",
                        message="Motif mentionne l'écriture/graphisme mais pas l'évaluation",
                        suggestion="Évaluer les aspects graphomoteurs mentionnés dans le motif",
                    )
                )

        return issues

    def _validate_completeness(
        self, sections_text: Dict[str, str]
    ) -> List[ValidationIssue]:
        """Validation de la complétude du bilan"""
        issues = []

        # Sections obligatoires
        required_sections = [
            "Identité & contexte",
            "Motif de la demande",
            "Évaluation psychomotrice",
            "Conclusion & recommandations",
        ]

        missing_sections = []
        for section in required_sections:
            if section not in sections_text or not sections_text[section].strip():
                missing_sections.append(section)

        if missing_sections:
            issues.append(
                ValidationIssue(
                    level=ValidationLevel.ERROR,
                    category="complétude",
                    message=f"Sections obligatoires manquantes: {', '.join(missing_sections)}",
                    suggestion="Compléter toutes les sections obligatoires",
                )
            )

        # Vérification sections optionnelles importantes
        important_optional = ["Analyse & synthèse", "Projet thérapeutique"]
        missing_optional = [
            s
            for s in important_optional
            if s not in sections_text or not sections_text[s].strip()
        ]

        if missing_optional:
            issues.append(
                ValidationIssue(
                    level=ValidationLevel.INFO,
                    category="complétude",
                    message=f"Sections importantes manquantes: {', '.join(missing_optional)}",
                    suggestion="Considérer l'ajout de ces sections pour un bilan plus complet",
                )
            )

        return issues

    def _calculate_validation_score(
        self, issues: List[ValidationIssue], word_count: int
    ) -> float:
        """Calcule un score de validation de 0.0 à 1.0"""

        if not issues:
            return 1.0

        # Pondération des pénalités par niveau
        penalties = {
            ValidationLevel.INFO: 0.02,
            ValidationLevel.WARNING: 0.05,
            ValidationLevel.ERROR: 0.15,
            ValidationLevel.CRITICAL: 0.30,
        }

        total_penalty = 0.0
        for issue in issues:
            total_penalty += penalties.get(issue.level, 0.05)

        # Bonus pour longueur appropriée
        length_bonus = min(0.1, word_count / 1000)  # Bonus jusqu'à 10% pour 1000+ mots

        # Score final
        score = max(0.0, min(1.0, 1.0 - total_penalty + length_bonus))

        return score

    def get_validation_summary(self, result: ValidationResult) -> str:
        """Génère un résumé de validation lisible"""

        summary_lines = [
            f"🎯 Score de validation: {result.score:.1%}",
            f"📝 Mots analysés: {result.word_count:,}",
            f"📋 Sections validées: {result.sections_validated}",
        ]

        if result.issues:
            by_level = {}
            for issue in result.issues:
                level = issue.level.value
                by_level[level] = by_level.get(level, 0) + 1

            summary_lines.append("⚠️  Problèmes détectés:")
            for level, count in by_level.items():
                emoji = {"info": "ℹ️", "warning": "⚠️", "error": "❌", "critical": "🚨"}
                summary_lines.append(
                    f"  {emoji.get(level, '•')} {level.title()}: {count}"
                )
        else:
            summary_lines.append("✅ Aucun problème détecté")

        return "\n".join(summary_lines)

    def get_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques de validation"""
        return {
            **self.validation_stats,
            "average_issues_per_section": (
                self.validation_stats["issues_found"]
                / max(1, self.validation_stats["sections_validated"])
            ),
        }
