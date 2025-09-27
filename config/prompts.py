"""
Templates de prompts pour la génération de bilans psychomoteurs
"""

from typing import Dict, Optional

from pydantic import BaseModel, Field


class PromptTemplate(BaseModel):
    """Template de prompt avec métadonnées"""

    template: str
    description: str
    variables: list[str] = Field(default_factory=list)
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None


class PromptLibrary:
    """Bibliothèque centralisée des prompts"""

    # Prompt système de base
    BASE_SYSTEM = PromptTemplate(
        template="""Tu es un psychomotricien diplômé d'État avec 10 ans d'expérience clinique qui rédige des bilans professionnels.

RÈGLES STRICTES :
- Utilise UNIQUEMENT les informations fournies dans les NOTES
- Style : clair, professionnel, factuel et bienveillant
- Terminologie : vocabulaire psychomoteur précis et approprié
- Si une information manque : écris « Non observé/Non renseigné »
- Évite les généralités, reste spécifique au patient
- Pas d'invention de données ou de références externes

QUALITÉ ATTENDUE :
- Phrases bien construites et fluides
- Transitions logiques entre les idées
- Vocabulaire adapté au contexte médical
- Respect de la déontologie professionnelle""",
        description="Prompt système de base pour tous les contextes",
        variables=[],
        temperature=0.3,
    )

    # Prompt spécialisé pour l'évaluation
    EVALUATION_SYSTEM = PromptTemplate(
        template="""Tu es un psychomotricien diplômé d'État qui rédige la section d'évaluation psychomotrice d'un bilan.

SPÉCIFICITÉS ÉVALUATION :
- Structure tes observations par domaines psychomoteurs
- Utilise une terminologie clinique précise
- Décris les observations factuelles sans interprétation excessive
- Mentionne les conditions d'observation (spontané, sur consigne, etc.)
- Indique les outils/tests utilisés quand c'est précisé

DOMAINES À EXPLORER (si données disponibles) :
- Tonus et régulation tonique
- Posture et équilibre statique/dynamique
- Motricité globale et coordination
- Motricité fine et praxies
- Schéma corporel et conscience du corps
- Latéralité et dominance
- Intégration visuo-spatiale
- Attention et fonctions exécutives
- Graphisme et écriture
- Aspects sensori-moteurs

FORMULATION :
- « L'enfant présente... » plutôt que « L'enfant a... »
- « Lors de l'observation... » pour contextualiser
- « On note... » ou « Il apparaît... » pour les constats
- Éviter « normal/anormal », préférer « dans la norme attendue » ou « atypique »""",
        description="Prompt pour la section évaluation psychomotrice",
        variables=["domaines_observes", "age_patient", "contexte_evaluation"],
        max_tokens=1000,
        temperature=0.25,
    )

    # Prompt pour les conclusions
    CONCLUSION_SYSTEM = PromptTemplate(
        template="""Tu es un psychomotricien qui rédige la conclusion et les recommandations d'un bilan.

STRUCTURE CONCLUSION :
1. Synthèse des observations principales
2. Mise en perspective avec le motif initial
3. Hypothèses cliniques (si approprié)
4. Recommandations concrètes et réalisables

RECOMMANDATIONS :
- Spécifiques au profil du patient
- Réalistes et applicables
- Hiérarchisées par priorité
- Incluent modalités pratiques si pertinent

FORMULATIONS PROFESSIONNELLES :
- « Les observations mettent en évidence... »
- « Au regard du motif initial... »
- « Il conviendrait de... »
- « Une prise en charge pourrait être bénéfique pour... »
- « Un suivi spécialisé est recommandé... »

ÉVITER :
- Diagnostic différentiel détaillé
- Recommandations trop générales
- Pronostic à long terme
- Références à des pathologies non confirmées""",
        description="Prompt pour conclusions et recommandations",
        variables=["motif_initial", "observations_principales", "age_patient"],
        max_tokens=800,
        temperature=0.3,
    )

    # Prompt pour le projet thérapeutique
    THERAPEUTIC_SYSTEM = PromptTemplate(
        template="""Tu es un psychomotricien qui établit un projet thérapeutique suite à un bilan.

ÉLÉMENTS À INCLURE :
- Objectifs thérapeutiques SMART (Spécifiques, Mesurables, Atteignables, Réalistes, Temporels)
- Modalités pratiques (fréquence, durée estimée)
- Approches et méthodes envisagées
- Collaboration avec autres professionnels si nécessaire
- Réévaluation périodique

FORMULATION DES OBJECTIFS :
- « Améliorer la coordination gestuelle lors d'activités... »
- « Développer les capacités de régulation tonique... »
- « Renforcer les compétences graphomotrices... »
- « Soutenir l'intégration des informations sensorielles... »

MODALITÉS :
- Fréquence : hebdomadaire, bi-mensuelle...
- Durée : estimation prudente (ex: « 6 mois avec réévaluation »)
- Cadre : individuel, groupe, mixte
- Lieu : cabinet, structure, domicile...

COLLABORATION :
- Lien avec famille/école si pertinent
- Coordination avec autres thérapeutes
- Communication avec prescripteur""",
        description="Prompt pour projet thérapeutique",
        variables=["objectifs_identifies", "modalites_souhaitees", "contraintes"],
        max_tokens=600,
        temperature=0.3,
    )

    # Template pour l'instruction complète
    SECTION_INSTRUCTION = PromptTemplate(
        template="""Rédige la section suivante d'un bilan psychomoteur en français.

SECTION DEMANDÉE : "{section_title}"

CONTRAINTES :
- Utilise UNIQUEMENT les informations dans les NOTES ci-dessous
- Style : professionnel, sobre, factuel et bienveillant
- Si une information manque, écris « Non observé/Non renseigné »
- Longueur indicative : {length_hint}
- Terminologie psychomotrice appropriée

STRUCTURE ATTENDUE :
{structure_hint}

NOTES (pour cette section) :
{notes}

INSTRUCTIONS SPÉCIFIQUES :
{specific_instructions}""",
        description="Template principal pour génération de sections",
        variables=[
            "section_title",
            "length_hint",
            "structure_hint",
            "notes",
            "specific_instructions",
        ],
        max_tokens=800,
        temperature=0.3,
    )


class PromptBuilder:
    """Constructeur de prompts adaptatifs"""

    def __init__(self):
        self.library = PromptLibrary()

    def get_system_prompt(self, section_name: Optional[str] = None, **context) -> str:
        """Retourne le prompt système approprié selon la section"""

        section_lower = section_name.lower() if section_name else ""

        # Sélection du prompt selon la section
        if "évaluation" in section_lower and "psychomotrice" in section_lower:
            prompt_template = self.library.EVALUATION_SYSTEM
        elif any(
            keyword in section_lower for keyword in ["conclusion", "recommandation"]
        ):
            prompt_template = self.library.CONCLUSION_SYSTEM
        elif any(
            keyword in section_lower
            for keyword in ["thérapeutique", "projet", "prise en charge"]
        ):
            prompt_template = self.library.THERAPEUTIC_SYSTEM
        else:
            prompt_template = self.library.BASE_SYSTEM

        # Adaptation contextuelle
        prompt = prompt_template.template

        # Ajout d'informations contextuelles si disponibles
        if context.get("age_patient"):
            prompt += f"\n\nCONTEXTE : Patient de {context['age_patient']} ans."

        if context.get("specialites"):
            prompt += f"\n\nSPÉCIALISATION : {context['specialites']}."

        return prompt

    def build_section_instruction(
        self,
        section_title: str,
        section_notes: any,
        length_hint: str = "quelques paragraphes",
        **context,
    ) -> str:
        """Construit l'instruction complète pour une section"""

        # Formatage des notes selon leur type
        if isinstance(section_notes, dict):
            if section_title == "Évaluation psychomotrice":
                notes_text = self._format_eval_notes(section_notes)
                structure_hint = self._get_eval_structure_hint(section_notes)
            else:
                notes_text = "\n".join([f"{k}: {v}" for k, v in section_notes.items()])
                structure_hint = f"- {section_title}"
        elif isinstance(section_notes, list):
            notes_text = "\n".join([f"- {item}" for item in section_notes])
            structure_hint = "Liste structurée"
        else:
            notes_text = str(section_notes).strip() or "Non observé/Non renseigné"
            structure_hint = f"- {section_title}"

        # Instructions spécifiques selon la section
        specific_instructions = self._get_specific_instructions(section_title, context)

        return self.library.SECTION_INSTRUCTION.template.format(
            section_title=section_title,
            length_hint=length_hint,
            structure_hint=structure_hint,
            notes=notes_text,
            specific_instructions=specific_instructions,
        )

    def _format_eval_notes(self, eval_notes: Dict) -> str:
        """Formate les notes d'évaluation psychomotrice"""
        eval_subsections = [
            "Tonus & posture",
            "Motricité globale",
            "Motricité fine / praxies",
            "Schéma corporel & latéralité",
            "Visuo-spatial",
            "Attention / fonctions exécutives",
            "Graphisme / écriture",
            "Sensori-moteur",
        ]

        lines = []

        # D'abord les sous-sections standard
        for subsection in eval_subsections:
            if subsection in eval_notes and eval_notes[subsection]:
                lines.append(f"{subsection}: {eval_notes[subsection]}")

        # Ensuite les autres observations
        for key, value in eval_notes.items():
            if key not in eval_subsections and value:
                lines.append(f"{key}: {value}")

        return "\n".join(lines) if lines else "Non observé/Non renseigné"

    def _get_eval_structure_hint(self, eval_notes: Dict) -> str:
        """Génère l'indication de structure pour l'évaluation"""
        observed_domains = [key for key, value in eval_notes.items() if value]

        if len(observed_domains) > 3:
            return "Organiser par domaines psychomoteurs avec sous-titres explicites"
        else:
            return "Paragraphes structurés par domaine observé"

    def _get_specific_instructions(self, section_title: str, context: Dict) -> str:
        """Retourne les instructions spécifiques selon la section"""

        instructions = []
        section_lower = section_title.lower()

        if "identité" in section_lower:
            instructions.append(
                "Inclure âge, contexte scolaire/professionnel si disponible"
            )
            instructions.append("Mentionner le prescripteur si indiqué")

        elif "motif" in section_lower:
            instructions.append("Reprendre la demande initiale sans interprétation")
            instructions.append("Préciser qui formule la demande")

        elif "anamnèse" in section_lower:
            instructions.append(
                "Organiser chronologiquement : grossesse, développement, antécédents"
            )
            instructions.append("Rester factuel, éviter les interprétations")

        elif "évaluation" in section_lower:
            instructions.append("Décrire les observations par domaine psychomoteur")
            instructions.append("Préciser les conditions d'observation")
            instructions.append("Utiliser la terminologie clinique appropriée")

        elif "test" in section_lower or "outil" in section_lower:
            instructions.append("Lister clairement les outils utilisés")
            instructions.append("Mentionner les résultats quantitatifs si disponibles")

        elif "analyse" in section_lower or "synthèse" in section_lower:
            instructions.append("Synthétiser les observations principales")
            instructions.append("Mettre en lien avec le motif initial")

        elif "conclusion" in section_lower:
            instructions.append("Formuler des recommandations concrètes et réalisables")
            instructions.append("Hiérarchiser les priorités")

        elif "thérapeutique" in section_lower:
            instructions.append("Définir des objectifs SMART")
            instructions.append("Préciser modalités pratiques (fréquence, durée)")

        elif "modalités" in section_lower:
            instructions.append("Mentionner accords/consentements")
            instructions.append("Préciser modalités pratiques de suivi")

        # Ajout du contexte patient si disponible
        if context.get("age"):
            instructions.append(
                f"Adapter le vocabulaire à un patient de {context['age']} ans"
            )

        return (
            " • ".join(instructions)
            if instructions
            else "Rédiger de manière claire et professionnelle"
        )

    def get_generation_config(self, section_name: str, model_name: str) -> Dict:
        """Retourne la configuration de génération optimale pour une section"""

        base_config = {
            "do_sample": True,
            "pad_token_id": None,  # Sera défini par le tokenizer
            "eos_token_id": None,  # Sera défini par le tokenizer
        }

        # Ajustements selon la section
        section_lower = section_name.lower()

        if "évaluation" in section_lower:
            # Plus de créativité pour l'évaluation détaillée
            base_config.update(
                {
                    "temperature": 0.3,
                    "top_p": 0.9,
                    "max_new_tokens": 1000,
                    "repetition_penalty": 1.15,
                }
            )

        elif any(
            keyword in section_lower
            for keyword in ["conclusion", "recommandation", "thérapeutique"]
        ):
            # Plus de précision pour les conclusions
            base_config.update(
                {
                    "temperature": 0.25,
                    "top_p": 0.85,
                    "max_new_tokens": 600,
                    "repetition_penalty": 1.2,
                }
            )

        elif any(
            keyword in section_lower for keyword in ["identité", "motif", "modalités"]
        ):
            # Très factuel pour les sections administratives
            base_config.update(
                {
                    "temperature": 0.2,
                    "top_p": 0.8,
                    "max_new_tokens": 400,
                    "repetition_penalty": 1.1,
                }
            )

        else:
            # Configuration standard
            base_config.update(
                {
                    "temperature": 0.3,
                    "top_p": 0.9,
                    "max_new_tokens": 800,
                    "repetition_penalty": 1.15,
                }
            )

        # Ajustements selon le modèle
        if "biomistral" in model_name.lower():
            # Modèle médical : moins de créativité
            base_config["temperature"] *= 0.8
            base_config["repetition_penalty"] *= 1.1

        elif "qwen" in model_name.lower():
            # Modèle très capable : peut être plus créatif
            base_config["temperature"] *= 1.1
            base_config["top_p"] = min(0.95, base_config["top_p"] * 1.05)

        return base_config


# Instance globale
prompt_builder = PromptBuilder()


# Fonctions helper pour rétrocompatibilité
def get_system_prompt(section_name: str = None, **context) -> str:
    """Fonction helper pour obtenir le prompt système"""
    return prompt_builder.get_system_prompt(section_name, **context)


def build_instruction_for_section(
    section_title: str, section_notes, length_hint: str = None, **context
) -> str:
    """Fonction helper pour construire l'instruction d'une section"""
    # Mapping des hints de longueur par défaut
    default_hints = {
        "Identité & contexte": "5 à 8 lignes",
        "Motif de la demande": "4 à 6 lignes",
        "Anamnèse synthétique": "8 à 12 lignes",
        "Évaluation psychomotrice": "1 à 3 paragraphes par sous-section présente",
        "Tests / outils utilisés": "liste claire et brève",
        "Analyse & synthèse": "8 à 12 lignes",
        "Conclusion & recommandations": "8 à 12 lignes",
        "Projet thérapeutique": "6 à 10 lignes",
        "Modalités & consentement": "4 à 6 lignes",
    }

    if not length_hint:
        length_hint = default_hints.get(section_title, "quelques paragraphes")

    return prompt_builder.build_section_instruction(
        section_title=section_title,
        section_notes=section_notes,
        length_hint=length_hint,
        **context,
    )
