"""
Système de contrôle qualité avancé pour les bilans psychomoteurs
"""

import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import language_tool_python
import spacy
from textstat import flesch_reading_ease


@dataclass
class QualityMetrics:
    """Métriques de qualité détaillées"""

    overall_score: float
    readability_score: float
    professional_score: float
    coherence_score: float
    completeness_score: float
    linguistic_quality: float

    word_count: int
    sentence_count: int
    avg_sentence_length: float
    professional_terms_ratio: float
    repetition_ratio: float

    issues: List[str]
    suggestions: List[str]


class QualityChecker:
    """
    Analyseur de qualité pour les bilans psychomoteurs
    """

    def __init__(self, enable_spacy: bool = True, enable_grammar_check: bool = False):
        # Termes professionnels psychomoteurs (élargi)
        self.professional_terms = {
            # Bases psychomotrices
            "tonus",
            "hypotonie",
            "hypertonie",
            "posture",
            "motricité",
            "praxie",
            "dyspraxie",
            "apraxie",
            "schéma corporel",
            "image du corps",
            "latéralité",
            "dominance",
            "coordination",
            "dissociation",
            "équilibre",
            "statique",
            "dynamique",
            # Développement et fonctions
            "développement",
            "maturation",
            "intégration",
            "régulation",
            "adaptation",
            "compensation",
            "attention",
            "concentration",
            "exécutif",
            "inhibition",
            "flexibilité",
            "mémoire de travail",
            # Sensoriel et perceptif
            "sensoriel",
            "proprioception",
            "exteroception",
            "interoception",
            "vestibulaire",
            "tactile",
            "visuo-spatial",
            "visuo-constructif",
            "visuo-moteur",
            # Graphisme et écriture
            "graphisme",
            "écriture",
            "préhension",
            "geste",
            "ductus",
            "liaison",
            "formation des lettres",
            # Évaluation et observation
            "évaluation",
            "observation",
            "clinique",
            "diagnostic",
            "anamnèse",
            "bilan",
            "synthèse",
            "analyse",
            "recommandation",
            "thérapeutique",
            "rééducation",
            # Troubles et pathologies
            "trouble",
            "retard",
            "déficit",
            "handicap",
            "autisme",
            "TDAH",
            "DYS",
            "déficience",
        }

        # Expressions à éviter (hallucinations/imprécisions)
        self.problematic_patterns = [
            r"selon les études récentes",
            r"il est important de noter que",
            r"de nombreux experts",
            r"statistiquement",
            r"\d+\s*%\s*des\s*(enfants|patients)",
            r"toutes les recherches montrent",
            r"il est prouvé que",
            r"généralement accepté que",
        ]

        # Indicateurs de qualité narrative
        self.quality_indicators = {
            "factual": ["observe", "constate", "note", "présente", "manifeste"],
            "professional": ["évalue", "analyse", "suggère", "recommande"],
            "objective": ["mesure", "test", "épreuve", "protocole"],
            "coherent": ["ainsi", "par ailleurs", "cependant", "néanmoins", "de plus"],
        }

        # Chargement spaCy pour analyse linguistique
        self.nlp = None
        if enable_spacy:
            try:
                self.nlp = spacy.load("fr_core_news_sm")
            except OSError:
                print(
                    "⚠️ Modèle spaCy français non trouvé. Analyse linguistique limitée."
                )

        # Correcteur grammatical
        self.grammar_checker = None
        if enable_grammar_check:
            try:
                self.grammar_checker = language_tool_python.LanguageTool("fr")
            except:
                print(
                    "⚠️ LanguageTool non disponible. Vérification grammaticale désactivée."
                )

    def evaluate_section(self, text: str, section_name: str) -> QualityMetrics:
        """
        Évalue la qualité d'une section
        """
        issues = []
        suggestions = []

        # Métriques de base
        word_count = len(text.split())
        sentences = self._extract_sentences(text)
        sentence_count = len(sentences)
        avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0

        # 1. Analyse de lisibilité
        readability_score = self._calculate_readability(text)

        # 2. Score professionnel
        professional_score, prof_ratio = self._calculate_professional_score(text)

        # 3. Cohérence narrative
        coherence_score = self._calculate_coherence(text, sentences)

        # 4. Complétude
        completeness_score = self._calculate_completeness(text, section_name)

        # 5. Qualité linguistique
        linguistic_quality = self._calculate_linguistic_quality(text)

        # 6. Détection de répétitions
        repetition_ratio = self._calculate_repetition_ratio(text)

        # Vérifications spécifiques
        self._check_length_appropriateness(
            word_count, section_name, issues, suggestions
        )
        self._check_hallucinations(text, issues, suggestions)
        self._check_section_specific_requirements(
            text, section_name, issues, suggestions
        )

        # Score global (pondéré)
        overall_score = (
            readability_score * 0.15
            + professional_score * 0.25
            + coherence_score * 0.20
            + completeness_score * 0.25
            + linguistic_quality * 0.15
        )

        # Pénalités
        if repetition_ratio > 0.3:
            overall_score *= 0.8
            issues.append(f"Taux de répétition élevé: {repetition_ratio:.1%}")

        if word_count < 30:
            overall_score *= 0.5
            issues.append("Section trop courte")

        return QualityMetrics(
            overall_score=overall_score,
            readability_score=readability_score,
            professional_score=professional_score,
            coherence_score=coherence_score,
            completeness_score=completeness_score,
            linguistic_quality=linguistic_quality,
            word_count=word_count,
            sentence_count=sentence_count,
            avg_sentence_length=avg_sentence_length,
            professional_terms_ratio=prof_ratio,
            repetition_ratio=repetition_ratio,
            issues=issues,
            suggestions=suggestions,
        )

    def evaluate_full_bilan(self, sections_text: Dict[str, str]) -> QualityMetrics:
        """
        Évalue la qualité globale du bilan
        """
        all_metrics = []
        global_issues = []
        global_suggestions = []

        # Évaluation section par section
        for section_name, text in sections_text.items():
            metrics = self.evaluate_section(text, section_name)
            all_metrics.append(metrics)
            global_issues.extend(
                [f"{section_name}: {issue}" for issue in metrics.issues]
            )
            global_suggestions.extend(
                [f"{section_name}: {sugg}" for sugg in metrics.suggestions]
            )

        # Calculs globaux
        if all_metrics:
            avg_overall = sum(m.overall_score for m in all_metrics) / len(all_metrics)
            total_words = sum(m.word_count for m in all_metrics)
            total_sentences = sum(m.sentence_count for m in all_metrics)

            # Vérifications de cohérence globale
            self._check_global_coherence(
                sections_text, global_issues, global_suggestions
            )

            return QualityMetrics(
                overall_score=avg_overall,
                readability_score=sum(m.readability_score for m in all_metrics)
                / len(all_metrics),
                professional_score=sum(m.professional_score for m in all_metrics)
                / len(all_metrics),
                coherence_score=sum(m.coherence_score for m in all_metrics)
                / len(all_metrics),
                completeness_score=sum(m.completeness_score for m in all_metrics)
                / len(all_metrics),
                linguistic_quality=sum(m.linguistic_quality for m in all_metrics)
                / len(all_metrics),
                word_count=total_words,
                sentence_count=total_sentences,
                avg_sentence_length=total_words / total_sentences
                if total_sentences > 0
                else 0,
                professional_terms_ratio=sum(
                    m.professional_terms_ratio for m in all_metrics
                )
                / len(all_metrics),
                repetition_ratio=sum(m.repetition_ratio for m in all_metrics)
                / len(all_metrics),
                issues=global_issues,
                suggestions=global_suggestions,
            )

        return QualityMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, [], [])

    def _extract_sentences(self, text: str) -> List[str]:
        """Extrait les phrases du texte"""
        sentences = re.split(r"[.!?]+", text)
        return [s.strip() for s in sentences if s.strip()]

    def _calculate_readability(self, text: str) -> float:
        """Calcule le score de lisibilité"""
        try:
            flesch_score = flesch_reading_ease(text)
            # Normalisation pour contexte médical (plus technique accepté)
            return min(1.0, max(0.0, (flesch_score + 20) / 100))
        except:
            return 0.5  # Score neutre si calcul impossible

    def _calculate_professional_score(self, text: str) -> Tuple[float, float]:
        """Calcule le score de professionnalisme"""
        text_lower = text.lower()
        words = text.split()

        # Comptage des termes professionnels
        prof_terms_found = sum(
            1 for term in self.professional_terms if term in text_lower
        )
        prof_ratio = prof_terms_found / len(words) if words else 0

        # Score basé sur la densité de termes professionnels
        score = min(1.0, prof_ratio * 50)  # Facteur d'ajustement

        return score, prof_ratio

    def _calculate_coherence(self, text: str, sentences: List[str]) -> float:
        """Calcule le score de cohérence narrative"""
        if len(sentences) < 2:
            return 0.5

        coherence_indicators = 0
        total_possible = len(sentences) - 1

        for sentence in sentences:
            sentence_lower = sentence.lower()
            for category, indicators in self.quality_indicators.items():
                if any(indicator in sentence_lower for indicator in indicators):
                    coherence_indicators += 1
                    break

        return min(
            1.0, coherence_indicators / total_possible if total_possible > 0 else 0
        )

    def _calculate_completeness(self, text: str, section_name: str) -> float:
        """Calcule le score de complétude"""
        text_lower = text.lower()

        # Pénalités pour contenus incomplets
        empty_indicators = [
            "non observé",
            "non renseigné",
            "non disponible",
            "à compléter",
        ]
        empty_count = sum(text_lower.count(indicator) for indicator in empty_indicators)

        # Score basé sur le ratio de contenu vide
        word_count = len(text.split())
        empty_ratio = empty_count / word_count if word_count > 0 else 1

        return max(0.0, 1.0 - empty_ratio * 3)  # Pénalité pour contenu vide

    def _calculate_linguistic_quality(self, text: str) -> float:
        """Calcule la qualité linguistique"""
        if not self.nlp:
            return 0.7  # Score par défaut

        try:
            doc = self.nlp(text)

            # Analyse syntaxique
            complete_sentences = 0
            total_sentences = 0

            for sent in doc.sents:
                total_sentences += 1
                # Vérification structure basique (sujet + verbe)
                has_subject = any(
                    token.dep_ in ["nsubj", "nsubjpass"] for token in sent
                )
                has_verb = any(token.pos_ == "VERB" for token in sent)
                if has_subject and has_verb:
                    complete_sentences += 1

            syntax_score = (
                complete_sentences / total_sentences if total_sentences > 0 else 0
            )

            # Vérification grammaticale si disponible
            grammar_score = 1.0
            if self.grammar_checker:
                matches = self.grammar_checker.check(text)
                error_ratio = len(matches) / len(text.split()) if text.split() else 0
                grammar_score = max(0.0, 1.0 - error_ratio * 2)

            return (syntax_score + grammar_score) / 2

        except Exception:
            return 0.7

    def _calculate_repetition_ratio(self, text: str) -> float:
        """Calcule le ratio de répétitions"""
        words = [w.lower() for w in text.split() if len(w) > 3]
        if not words:
            return 0

        word_counts = Counter(words)
        repeated_words = sum(count - 1 for count in word_counts.values() if count > 1)

        return repeated_words / len(words)

    def _check_length_appropriateness(
        self,
        word_count: int,
        section_name: str,
        issues: List[str],
        suggestions: List[str],
    ):
        """Vérifie la longueur appropriée selon la section"""
        expected_ranges = {
            "Identité & contexte": (30, 80),
            "Motif de la demande": (25, 60),
            "Anamnèse synthétique": (60, 120),
            "Évaluation psychomotrice": (100, 300),
            "Tests / outils utilisés": (20, 60),
            "Analyse & synthèse": (80, 150),
            "Conclusion & recommandations": (60, 120),
            "Projet thérapeutique": (40, 100),
            "Modalités & consentement": (20, 60),
        }

        if section_name in expected_ranges:
            min_words, max_words = expected_ranges[section_name]

            if word_count < min_words:
                issues.append(
                    f"Section trop courte ({word_count} mots, min recommandé: {min_words})"
                )
                suggestions.append(
                    "Développer davantage le contenu avec des observations spécifiques"
                )
            elif word_count > max_words:
                issues.append(
                    f"Section trop longue ({word_count} mots, max recommandé: {max_words})"
                )
                suggestions.append("Synthétiser le contenu en gardant l'essentiel")

    def _check_hallucinations(
        self, text: str, issues: List[str], suggestions: List[str]
    ):
        """Détecte les potentielles hallucinations"""
        for pattern in self.problematic_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                issues.append(f"Possible hallucination détectée: {pattern[:30]}...")
                suggestions.append(
                    "Remplacer par des observations factuelles spécifiques au patient"
                )

    def _check_section_specific_requirements(
        self, text: str, section_name: str, issues: List[str], suggestions: List[str]
    ):
        """Vérifications spécifiques par section"""
        text_lower = text.lower()

        if section_name == "Évaluation psychomotrice":
            required_domains = ["tonus", "motricité", "coordination", "équilibre"]
            missing_domains = [
                domain for domain in required_domains if domain not in text_lower
            ]

            if len(missing_domains) > 2:
                issues.append(
                    f"Domaines d'évaluation manquants: {', '.join(missing_domains)}"
                )
                suggestions.append(
                    "S'assurer de couvrir tous les domaines psychomoteurs principaux"
                )

        elif section_name == "Conclusion & recommandations":
            if "recommandation" not in text_lower and "conseil" not in text_lower:
                issues.append("Aucune recommandation explicite trouvée")
                suggestions.append(
                    "Inclure des recommandations concrètes et actionables"
                )

        elif section_name == "Projet thérapeutique":
            therapeutic_terms = ["objectif", "séance", "fréquence", "durée"]
            found_terms = sum(1 for term in therapeutic_terms if term in text_lower)

            if found_terms < 2:
                issues.append("Informations thérapeutiques insuffisantes")
                suggestions.append(
                    "Préciser objectifs, modalités et planning thérapeutique"
                )

    def _check_global_coherence(
        self, sections_text: Dict[str, str], issues: List[str], suggestions: List[str]
    ):
        """Vérifie la cohérence globale entre sections"""

        # Extraction d'informations clés
        patient_info = self._extract_patient_info(sections_text)

        # Vérification cohérence âge/niveau
        if patient_info.get("age") and patient_info.get("niveau"):
            age = patient_info["age"]
            niveau = patient_info["niveau"]

            # Cohérence âge/classe approximative
            age_niveau_map = {
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

            if niveau in age_niveau_map:
                expected_min, expected_max = age_niveau_map[niveau]
                if not (expected_min <= age <= expected_max + 2):
                    issues.append(
                        f"Incohérence âge ({age} ans) / niveau scolaire ({niveau})"
                    )

        # Vérification cohérence problématiques
        motif = sections_text.get("Motif de la demande", "").lower()
        evaluation = sections_text.get("Évaluation psychomotrice", "").lower()
        conclusion = sections_text.get("Conclusion & recommandations", "").lower()

        # Si un trouble est mentionné dans le motif, il devrait être abordé dans l'évaluation
        troubles_mentionnes = ["dyspraxie", "tdah", "trouble", "retard", "difficulté"]
        troubles_in_motif = [t for t in troubles_mentionnes if t in motif]

        if troubles_in_motif:
            troubles_in_eval = [t for t in troubles_in_motif if t in evaluation]
            if len(troubles_in_eval) < len(troubles_in_motif) / 2:
                suggestions.append(
                    "S'assurer que les problématiques du motif sont explorées dans l'évaluation"
                )

    def _extract_patient_info(self, sections_text: Dict[str, str]) -> Dict:
        """Extrait les informations patient pour vérifications"""
        info = {}

        identite_text = sections_text.get("Identité & contexte", "")

        # Extraction âge
        age_match = re.search(r"(\d+)\s*ans?", identite_text)
        if age_match:
            info["age"] = int(age_match.group(1))

        # Extraction niveau scolaire
        niveau_match = re.search(
            r"(CP|CE1|CE2|CM1|CM2|6ème|5ème|4ème|3ème)", identite_text
        )
        if niveau_match:
            info["niveau"] = niveau_match.group(1)

        return info

    def get_improvement_suggestions(
        self, metrics: QualityMetrics, section_name: str
    ) -> List[str]:
        """Génère des suggestions d'amélioration personnalisées"""
        suggestions = []

        if metrics.overall_score < 0.6:
            suggestions.append(
                "⚠️ Qualité globale faible - révision complète recommandée"
            )

        if metrics.readability_score < 0.5:
            suggestions.append(
                "📖 Simplifier la structure des phrases pour améliorer la lisibilité"
            )

        if metrics.professional_score < 0.4:
            suggestions.append(
                "🎯 Intégrer davantage de terminologie psychomotrice spécialisée"
            )

        if metrics.coherence_score < 0.5:
            suggestions.append("🔗 Améliorer les transitions et la logique narrative")

        if metrics.completeness_score < 0.6:
            suggestions.append(
                "📝 Réduire les mentions 'Non observé' par des observations concrètes"
            )

        if metrics.repetition_ratio > 0.3:
            suggestions.append("🔄 Varier le vocabulaire pour éviter les répétitions")

        if metrics.avg_sentence_length > 25:
            suggestions.append("✂️ Raccourcir les phrases trop longues")
        elif metrics.avg_sentence_length < 8:
            suggestions.append("➕ Développer davantage les phrases courtes")

        return suggestions + metrics.suggestions

    def generate_quality_report(
        self, metrics: QualityMetrics, section_name: str
    ) -> str:
        """Génère un rapport de qualité détaillé"""
        report = f"""
📊 RAPPORT QUALITÉ - {section_name}
{"=" * 50}

🎯 Score global: {metrics.overall_score:.1%}
📖 Lisibilité: {metrics.readability_score:.1%}
👨‍⚕️ Professionnalisme: {metrics.professional_score:.1%}
🔗 Cohérence: {metrics.coherence_score:.1%}
📝 Complétude: {metrics.completeness_score:.1%}
🔤 Qualité linguistique: {metrics.linguistic_quality:.1%}

📈 MÉTRIQUES DÉTAILLÉES:
- Mots: {metrics.word_count}
- Phrases: {metrics.sentence_count}
- Longueur moyenne: {metrics.avg_sentence_length:.1f} mots/phrase
- Termes professionnels: {metrics.professional_terms_ratio:.1%}
- Taux répétition: {metrics.repetition_ratio:.1%}

"""

        if metrics.issues:
            report += "⚠️ PROBLÈMES DÉTECTÉS:\n"
            for issue in metrics.issues:
                report += f"  • {issue}\n"
            report += "\n"

        suggestions = self.get_improvement_suggestions(metrics, section_name)
        if suggestions:
            report += "💡 SUGGESTIONS D'AMÉLIORATION:\n"
            for suggestion in suggestions:
                report += f"  • {suggestion}\n"

        return report
