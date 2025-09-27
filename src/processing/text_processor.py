"""
Processeur de texte avancé pour l'amélioration qualitative des bilans
"""

import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import spacy


@dataclass
class ProcessingResult:
    """Résultat du traitement avec métriques"""

    processed_text: str
    improvements_made: List[str]
    processing_time: float
    quality_gain: float


class TextProcessor:
    """
    Processeur de texte intelligent pour l'amélioration des bilans psychomoteurs
    """

    def __init__(self, enable_advanced_nlp: bool = True):
        # Dictionnaires de remplacement pour améliorer le style
        self.style_improvements = {
            # Remplacements pour un style plus professionnel
            "l'enfant a": "l'enfant présente",
            "il a": "il présente",
            "elle a": "elle présente",
            "on observe": "l'observation révèle",
            "on voit": "l'examen met en évidence",
            "il fait": "il réalise",
            "elle fait": "elle réalise",
            "très bien": "de manière satisfaisante",
            "très mal": "avec des difficultés importantes",
            "pas bien": "avec des difficultés",
            "normal": "dans la norme attendue",
            "pas normal": "atypique",
            "bizarre": "inhabituel",
            "étrange": "atypique",
        }

        # Expressions temporelles plus précises
        self.temporal_improvements = {
            "parfois": "de manière intermittente",
            "souvent": "fréquemment",
            "tout le temps": "de manière constante",
            "jamais": "aucune occurrence observée",
            "toujours": "systématiquement",
        }

        # Vocabulaire spécialisé pour remplacer les termes génériques
        self.professional_vocabulary = {
            "problème": "difficulté",
            "souci": "difficulté",
            "difficulté motrice": "trouble psychomoteur",
            "bouge beaucoup": "présente une agitation motrice",
            "ne tient pas en place": "manifeste une instabilité posturale",
            "maladroit": "présente des difficultés de coordination",
            "lent": "présente un ralentissement psychomoteur",
            "rapide": "présente une accélération du rythme",
            "nerveux": "présente des signes d'anxiété motrice",
            "calme": "présente une régulation tonique adaptée",
        }

        # Connecteurs logiques pour améliorer la cohérence
        self.coherence_connectors = {
            "addition": ["par ailleurs", "de plus", "également", "en outre"],
            "opposition": ["cependant", "néanmoins", "toutefois", "en revanche"],
            "cause": ["en effet", "ainsi", "par conséquent", "de ce fait"],
            "temporal": ["lors de", "durant", "au cours de", "pendant"],
        }

        # Expressions de nuance professionnelle
        self.nuance_expressions = {
            "certainty_high": [
                "l'examen confirme",
                "l'observation établit",
                "il est constaté",
            ],
            "certainty_medium": [
                "il semble que",
                "on peut supposer",
                "l'impression clinique suggère",
            ],
            "certainty_low": [
                "à explorer",
                "à approfondir",
                "nécessite une observation complémentaire",
            ],
        }

        # Chargement du modèle NLP
        self.nlp = None
        if enable_advanced_nlp:
            try:
                self.nlp = spacy.load("fr_core_news_sm")
            except OSError:
                print("⚠️ Modèle spaCy non disponible. Traitement basique uniquement.")

    def process_section(self, text: str, section_name: str) -> str:
        """
        Traite une section complète avec toutes les améliorations
        """
        # Étapes de traitement progressif
        processed_text = text

        # 1. Nettoyage de base
        processed_text = self._clean_basic_formatting(processed_text)

        # 2. Amélioration du style professionnel
        processed_text = self._improve_professional_style(processed_text)

        # 3. Enrichissement vocabulaire
        processed_text = self._enrich_vocabulary(processed_text)

        # 4. Amélioration cohérence narrative
        processed_text = self._improve_narrative_coherence(processed_text, section_name)

        # 5. Optimisation des transitions
        processed_text = self._optimize_sentence_transitions(processed_text)

        # 6. Corrections linguistiques avancées
        if self.nlp:
            processed_text = self._advanced_linguistic_corrections(processed_text)

        # 7. Finalisation et validation
        processed_text = self._finalize_formatting(processed_text)

        return processed_text

    def _clean_basic_formatting(self, text: str) -> str:
        """Nettoyage et formatage de base"""
        # Suppression des espaces multiples
        text = re.sub(r"\s+", " ", text)

        # Correction de la ponctuation
        text = re.sub(r"\s+([,.;!?:])", r"\1", text)
        text = re.sub(r"([,.;!?:])([A-ZÀ-ÿ])", r"\1 \2", text)

        # Suppression des répétitions de mots consécutifs
        text = re.sub(r"\b(\w+)\s+\1\b", r"\1", text, flags=re.IGNORECASE)

        # Correction des apostrophes
        text = re.sub(r"'", "'", text)
        text = re.sub(r"\s+\'", "'", text)

        # Gestion des majuscules après points
        text = re.sub(
            r"\.(\s*)([a-zà-ÿ])", lambda m: "." + m.group(1) + m.group(2).upper(), text
        )

        return text.strip()

    def _improve_professional_style(self, text: str) -> str:
        """Améliore le style professionnel"""
        result = text

        # Remplacement des expressions courantes par du vocabulaire professionnel
        for casual, professional in self.style_improvements.items():
            result = re.sub(
                r"\b" + re.escape(casual) + r"\b",
                professional,
                result,
                flags=re.IGNORECASE,
            )

        # Amélioration des expressions temporelles
        for casual, professional in self.temporal_improvements.items():
            result = re.sub(
                r"\b" + re.escape(casual) + r"\b",
                professional,
                result,
                flags=re.IGNORECASE,
            )

        return result

    def _enrich_vocabulary(self, text: str) -> str:
        """Enrichit le vocabulaire avec des termes spécialisés"""
        result = text

        # Remplacement par du vocabulaire psychomoteur spécialisé
        for generic, specialized in self.professional_vocabulary.items():
            # Utilisation de regex avec frontières de mots
            pattern = r"\b" + re.escape(generic) + r"\b"
            result = re.sub(pattern, specialized, result, flags=re.IGNORECASE)

        return result

    def _improve_narrative_coherence(self, text: str, section_name: str) -> str:
        """Améliore la cohérence narrative de la section"""
        sentences = self._split_into_sentences(text)

        if len(sentences) < 2:
            return text

        improved_sentences = []

        for i, sentence in enumerate(sentences):
            if i == 0:
                # Première phrase : introduction appropriée à la section
                if section_name == "Évaluation psychomotrice":
                    if not any(
                        word in sentence.lower()
                        for word in ["lors", "durant", "l'examen"]
                    ):
                        sentence = "Lors de l'évaluation, " + sentence.lower()
                improved_sentences.append(sentence)
            else:
                # Phrases suivantes : ajout de connecteurs logiques si nécessaire
                if not self._has_connector(sentence):
                    connector = self._select_appropriate_connector(
                        sentences[i - 1], sentence, section_name
                    )
                    if connector:
                        sentence = connector + ", " + sentence.lower()

                improved_sentences.append(sentence)

        return " ".join(improved_sentences)

    def _optimize_sentence_transitions(self, text: str) -> str:
        """Optimise les transitions entre phrases"""
        sentences = self._split_into_sentences(text)

        if len(sentences) < 2:
            return text

        optimized = []

        for i, sentence in enumerate(sentences):
            if i > 0:
                # Analyse de la relation avec la phrase précédente
                prev_sentence = sentences[i - 1]
                relationship = self._analyze_sentence_relationship(
                    prev_sentence, sentence
                )

                # Ajout de connecteur approprié si besoin
                if relationship and not self._has_connector(sentence):
                    if relationship == "contrast":
                        sentence = "Cependant, " + sentence.lower()
                    elif relationship == "addition":
                        sentence = "Par ailleurs, " + sentence.lower()
                    elif relationship == "consequence":
                        sentence = "Ainsi, " + sentence.lower()

            optimized.append(sentence)

        return " ".join(optimized)

    def _advanced_linguistic_corrections(self, text: str) -> str:
        """Corrections linguistiques avancées avec spaCy"""
        try:
            doc = self.nlp(text)
            corrections = []

            # Analyse des entités et amélioration
            for ent in doc.ents:
                # Correction des entités temporelles
                if ent.label_ == "DATE" and "ans" in ent.text:
                    # Standardisation format âge
                    corrected = re.sub(r"(\d+)\s*ans?", r"\1 ans", ent.text)
                    corrections.append((ent.text, corrected))

            # Application des corrections
            result = text
            for original, corrected in corrections:
                result = result.replace(original, corrected)

            return result

        except Exception:
            return text

    def _finalize_formatting(self, text: str) -> str:
        """Finalisation du formatage"""
        # Nettoyage final
        text = re.sub(r"\s+", " ", text)
        text = text.strip()

        # Assurer que le texte se termine par une ponctuation
        if text and text[-1] not in ".!?":
            text += "."

        # Capitalisation de la première lettre
        if text:
            text = text[0].upper() + text[1:]

        return text

    def _split_into_sentences(self, text: str) -> List[str]:
        """Divise le texte en phrases"""
        sentences = re.split(r"[.!?]+", text)
        return [s.strip() for s in sentences if s.strip()]

    def _has_connector(self, sentence: str) -> bool:
        """Vérifie si la phrase a déjà un connecteur logique"""
        sentence_lower = sentence.lower().strip()

        # Liste des connecteurs courants
        connectors = [
            "cependant",
            "néanmoins",
            "toutefois",
            "en revanche",
            "par ailleurs",
            "de plus",
            "également",
            "en outre",
            "ainsi",
            "par conséquent",
            "de ce fait",
            "en effet",
            "lors de",
            "durant",
            "au cours de",
            "pendant",
        ]

        return any(sentence_lower.startswith(conn) for conn in connectors)

    def _select_appropriate_connector(
        self, prev_sentence: str, current_sentence: str, section_name: str
    ) -> Optional[str]:
        """Sélectionne un connecteur approprié selon le contexte"""

        # Analyse simple du contenu pour déterminer le type de relation
        prev_lower = prev_sentence.lower()
        curr_lower = current_sentence.lower()

        # Mots-clés indiquant opposition
        opposition_keywords = ["mais", "cependant", "difficile", "problème", "limite"]
        if any(keyword in curr_lower for keyword in opposition_keywords):
            return "Cependant"

        # Mots-clés indiquant addition
        addition_keywords = ["aussi", "également", "observe", "note", "présente"]
        if any(keyword in curr_lower for keyword in addition_keywords):
            return "Par ailleurs"

        # Contexte spécifique à la section
        if section_name == "Évaluation psychomotrice":
            if "test" in curr_lower or "épreuve" in curr_lower:
                return "Lors des épreuves"

        return None

    def _analyze_sentence_relationship(
        self, prev_sentence: str, current_sentence: str
    ) -> Optional[str]:
        """Analyse la relation entre deux phrases consécutives"""
        prev_lower = prev_sentence.lower()
        curr_lower = current_sentence.lower()

        # Détection de contraste
        contrast_indicators = [
            "difficile",
            "problème",
            "échec",
            "limite",
            "mais",
            "cependant",
        ]
        if any(indicator in curr_lower for indicator in contrast_indicators):
            return "contrast"

        # Détection d'addition
        addition_indicators = ["également", "aussi", "de plus", "observe", "note"]
        if any(indicator in curr_lower for indicator in addition_indicators):
            return "addition"

        # Détection de conséquence
        consequence_indicators = ["donc", "ainsi", "par conséquent", "résultat"]
        if any(indicator in curr_lower for indicator in consequence_indicators):
            return "consequence"

        return None

    def enhance_section_content(
        self, text: str, section_name: str, notes_data: Dict
    ) -> str:
        """
        Enrichit le contenu d'une section en utilisant les données des notes
        """
        enhanced_text = text

        # Enrichissement spécifique par section
        if section_name == "Identité & contexte":
            enhanced_text = self._enhance_identity_section(enhanced_text, notes_data)
        elif section_name == "Évaluation psychomotrice":
            enhanced_text = self._enhance_evaluation_section(enhanced_text, notes_data)
        elif section_name == "Conclusion & recommandations":
            enhanced_text = self._enhance_conclusion_section(enhanced_text, notes_data)

        return enhanced_text

    def _enhance_identity_section(self, text: str, notes_data: Dict) -> str:
        """Enrichit la section identité avec des détails contextuels"""
        # Ajout de détails sur le contexte familial, scolaire, etc.
        # Implémentation spécifique selon les données disponibles
        return text

    def _enhance_evaluation_section(self, text: str, notes_data: Dict) -> str:
        """Enrichit la section évaluation avec des détails cliniques"""
        # Ajout de précisions sur les observations, les tests utilisés, etc.
        return text

    def _enhance_conclusion_section(self, text: str, notes_data: Dict) -> str:
        """Enrichit la conclusion avec des recommandations personnalisées"""
        # Ajout de recommandations spécifiques selon le profil du patient
        return text

    def calculate_improvement_metrics(
        self, original_text: str, processed_text: str
    ) -> Dict[str, float]:
        """Calcule les métriques d'amélioration du traitement"""
        metrics = {}

        # Calcul de la variation de longueur
        original_words = len(original_text.split())
        processed_words = len(processed_text.split())
        metrics["word_count_change"] = (
            (processed_words - original_words) / original_words
            if original_words > 0
            else 0
        )

        # Calcul du score de professionnalisme (approximatif)
        professional_terms = [
            "présente",
            "manifeste",
            "observe",
            "révèle",
            "évalue",
            "analyse",
        ]
        original_prof_score = sum(
            1 for term in professional_terms if term in original_text.lower()
        )
        processed_prof_score = sum(
            1 for term in professional_terms if term in processed_text.lower()
        )

        metrics["professionalism_gain"] = processed_prof_score - original_prof_score

        # Score de cohérence (présence de connecteurs)
        connectors = ["cependant", "par ailleurs", "ainsi", "lors de", "durant"]
        original_cohesion = sum(
            1 for conn in connectors if conn in original_text.lower()
        )
        processed_cohesion = sum(
            1 for conn in connectors if conn in processed_text.lower()
        )

        metrics["coherence_gain"] = processed_cohesion - original_cohesion

        return metrics
