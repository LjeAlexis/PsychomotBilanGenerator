"""
Configuration centralisée et moderne utilisant Pydantic Settings
"""

from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import torch
from pydantic import BaseModel, Field, computed_field, validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ModelConfig(BaseModel):
    """Configuration spécifique à un modèle"""

    name: str
    local_path: Optional[Path] = None
    hf_name: Optional[str] = None
    quantization: Literal["4bit", "8bit", "none"] = "4bit"
    attention_impl: Literal["sdpa", "flash_attention_2", "eager"] = "sdpa"
    max_context: int = Field(default=4096, ge=1024, le=32768)
    optimal_batch_size: int = Field(default=2, ge=1, le=8)
    temperature: float = Field(default=0.3, ge=0.1, le=1.0)
    max_tokens: int = Field(default=800, ge=100, le=2048)
    top_p: float = Field(default=0.9, ge=0.1, le=1.0)
    repetition_penalty: float = Field(default=1.15, ge=1.0, le=2.0)

    @validator("local_path", pre=True)
    def validate_local_path(cls, v):
        return Path(v) if v and isinstance(v, str) else v


class QualityConfig(BaseModel):
    """Configuration du contrôle qualité"""

    enable_quality_checks: bool = True
    enable_grammar_check: bool = False
    enable_spacy_analysis: bool = True
    enable_coherence_check: bool = True
    enable_hallucination_detection: bool = True
    min_quality_threshold: float = Field(default=0.6, ge=0.0, le=1.0)
    max_retries: int = Field(default=2, ge=0, le=5)
    quality_weights: Dict[str, float] = Field(
        default={
            "readability": 0.15,
            "professional": 0.25,
            "coherence": 0.20,
            "completeness": 0.25,
            "linguistic": 0.15,
        }
    )


class ProcessingConfig(BaseModel):
    """Configuration du traitement de texte"""

    enable_style_improvement: bool = True
    enable_vocabulary_enrichment: bool = True
    enable_coherence_enhancement: bool = True
    enable_advanced_nlp: bool = True
    enable_auto_correction: bool = True
    thermal_pause: float = Field(default=2.0, ge=0.0, le=10.0)
    enable_async_processing: bool = False
    max_concurrent_sections: int = Field(default=3, ge=1, le=10)


class OutputConfig(BaseModel):
    """Configuration de sortie"""

    default_format: Literal["docx", "pdf", "html", "md"] = "docx"
    include_quality_report: bool = True
    include_generation_stats: bool = True
    include_debug_info: bool = False
    auto_backup: bool = True
    compression_level: int = Field(default=6, ge=0, le=9)
    output_template: Optional[str] = None


class CacheConfig(BaseModel):
    """Configuration du cache"""

    enable_cache: bool = True
    cache_size_mb: int = Field(default=1024, ge=100, le=10240)  # 1GB par défaut
    cache_ttl_hours: int = Field(default=168, ge=1, le=8760)  # 1 semaine
    cache_compression: bool = True
    auto_cleanup: bool = True


class LoggingConfig(BaseModel):
    """Configuration des logs"""

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    enable_file_logging: bool = True
    log_rotation: str = "10 MB"
    log_retention: str = "1 month"
    enable_structured_logging: bool = True
    enable_performance_logging: bool = False


class Settings(BaseSettings):
    """Configuration principale de l'application"""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=False, extra="ignore"
    )

    # Chemins principaux
    base_dir: Path = Field(default_factory=lambda: Path(__file__).parent.parent)
    models_dir: Path = Field(default_factory=lambda: Path("./models"))
    cache_dir: Path = Field(default_factory=lambda: Path("./cache"))
    logs_dir: Path = Field(default_factory=lambda: Path("./logs"))
    output_dir: Path = Field(default_factory=lambda: Path("./output"))
    config_dir: Path = Field(default_factory=lambda: Path("./config"))

    # Configuration générale
    app_name: str = "PsychomotBilanGenerator"
    app_version: str = "2.0.0"
    debug: bool = False
    verbose: bool = False

    # Modèle par défaut
    default_model: str = "mistral"

    # Sous-configurations
    quality: QualityConfig = Field(default_factory=QualityConfig)
    processing: ProcessingConfig = Field(default_factory=ProcessingConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    # Structure du bilan
    section_order: List[str] = Field(
        default=[
            "Identité & contexte",
            "Motif de la demande",
            "Anamnèse synthétique",
            "Évaluation psychomotrice",
            "Tests / outils utilisés",
            "Analyse & synthèse",
            "Conclusion & recommandations",
            "Projet thérapeutique",
            "Modalités & consentement",
        ]
    )

    eval_subsections: List[str] = Field(
        default=[
            "Tonus & posture",
            "Motricité globale",
            "Motricité fine / praxies",
            "Schéma corporel & latéralité",
            "Visuo-spatial",
            "Attention / fonctions exécutives",
            "Graphisme / écriture",
            "Sensori-moteur",
        ]
    )

    # Hints de longueur par section
    length_hints: Dict[str, str] = Field(
        default={
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
    )

    @validator(
        "models_dir", "cache_dir", "logs_dir", "output_dir", "config_dir", pre=True
    )
    def resolve_paths(cls, v, values):
        """Résout les chemins relatifs par rapport au base_dir"""
        if isinstance(v, str):
            v = Path(v)
        if not v.is_absolute():
            base_dir = values.get("base_dir", Path(__file__).parent.parent)
            v = base_dir / v
        return v.resolve()

    @computed_field
    @property
    def models(self) -> Dict[str, ModelConfig]:
        """Configuration des modèles disponibles"""
        return {
            "mistral": ModelConfig(
                name="mistral",
                local_path=self.models_dir / "mistral-7b-instruct-v0.3",
                hf_name="mistralai/Mistral-7B-Instruct-v0.3",
                quantization="4bit",
                attention_impl="sdpa",
                temperature=0.3,
                max_tokens=800,
                optimal_batch_size=2,
            ),
            "qwen": ModelConfig(
                name="qwen",
                local_path=self.models_dir / "qwen2.5-7b-instruct",
                hf_name="Qwen/Qwen2.5-7B-Instruct",
                quantization="4bit",
                attention_impl="sdpa",
                temperature=0.25,
                max_tokens=900,
                optimal_batch_size=2,
            ),
            "biomistral": ModelConfig(
                name="biomistral",
                local_path=self.models_dir / "biomistral-7b",
                hf_name="BioMistral/BioMistral-7B",
                quantization="4bit",
                attention_impl="sdpa",
                temperature=0.2,
                max_tokens=1000,
                optimal_batch_size=1,
            ),
            "llama3": ModelConfig(
                name="llama3",
                local_path=self.models_dir / "llama3-8b-instruct",
                hf_name="meta-llama/Meta-Llama-3-8B-Instruct",
                quantization="4bit",
                attention_impl="sdpa",
                temperature=0.3,
                max_tokens=750,
                optimal_batch_size=2,
            ),
            "vigogne": ModelConfig(
                name="vigogne",
                local_path=self.models_dir / "vigogne-2-7b-instruct",
                hf_name="bofenghuang/vigogne-2-7b-instruct",
                quantization="4bit",
                attention_impl="sdpa",
                temperature=0.35,
                max_tokens=800,
                optimal_batch_size=2,
            ),
        }

    def get_model_config(self, model_name: str) -> ModelConfig:
        """Retourne la configuration d'un modèle"""
        if model_name in self.models:
            return self.models[model_name]

        # Configuration par défaut pour modèles inconnus
        return ModelConfig(
            name=model_name,
            hf_name=model_name,
            quantization="4bit",
            attention_impl="sdpa",
        )

    def get_model_path(self, model_name: str) -> str:
        """Retourne le chemin ou nom HF du modèle"""
        model_config = self.get_model_config(model_name)

        # Vérifier si le modèle existe localement
        if model_config.local_path and model_config.local_path.exists():
            # Vérifier la présence de fichiers modèle
            model_files = (
                list(model_config.local_path.glob("*.bin"))
                + list(model_config.local_path.glob("*.safetensors"))
                + list(model_config.local_path.glob("*.gguf"))
            )

            if model_files:
                print(f"✅ Modèle local trouvé : {model_config.local_path}")
                return str(model_config.local_path)

        # Fallback sur HuggingFace
        if model_config.hf_name:
            print(f"📥 Utilisation du modèle HF : {model_config.hf_name}")
            return model_config.hf_name

        return model_name

    def list_available_models(self) -> List[str]:
        """Liste les modèles disponibles localement"""
        available = []

        for name, config in self.models.items():
            if config.local_path and config.local_path.exists():
                model_files = (
                    list(config.local_path.glob("*.bin"))
                    + list(config.local_path.glob("*.safetensors"))
                    + list(config.local_path.glob("*.gguf"))
                )
                if model_files:
                    available.append(name)

        return available

    def get_generation_params(self, model_name: str, **overrides) -> Dict[str, Any]:
        """Retourne les paramètres de génération pour un modèle"""
        model_config = self.get_model_config(model_name)

        params = {
            "temperature": model_config.temperature,
            "max_new_tokens": model_config.max_tokens,
            "top_p": model_config.top_p,
            "repetition_penalty": model_config.repetition_penalty,
            "do_sample": True,
            "pad_token_id": None,  # Sera défini par le tokenizer
            "eos_token_id": None,  # Sera défini par le tokenizer
        }

        # Application des overrides
        params.update(overrides)

        return params

    def create_directories(self) -> None:
        """Crée les dossiers nécessaires"""
        for directory in [
            self.models_dir,
            self.cache_dir,
            self.logs_dir,
            self.output_dir,
            self.config_dir,
        ]:
            directory.mkdir(parents=True, exist_ok=True)

    def validate_configuration(self) -> List[str]:
        """Valide la configuration et retourne les problèmes"""
        issues = []

        # Vérification des dossiers
        self.create_directories()

        # Vérification des modèles
        available_models = self.list_available_models()
        if not available_models and not torch.cuda.is_available():
            issues.append("Aucun modèle local et pas de GPU - performance limitée")

        if self.default_model not in self.models:
            issues.append(f"Modèle par défaut inconnu : {self.default_model}")

        # Vérification CUDA
        if not torch.cuda.is_available():
            issues.append("CUDA non disponible - utilisation CPU uniquement")

        # Vérification mémoire GPU
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            if gpu_memory < 6:
                issues.append(
                    f"Mémoire GPU faible ({gpu_memory:.1f}GB) - quantification requise"
                )

        return issues

    def export_config(self, file_path: Path) -> None:
        """Exporte la configuration actuelle"""
        config_data = self.model_dump(mode="json")

        with open(file_path, "w", encoding="utf-8") as f:
            import json

            json.dump(config_data, f, indent=2, ensure_ascii=False, default=str)

    def __str__(self) -> str:
        """Représentation string de la configuration"""
        available_models = self.list_available_models()
        gpu_info = "❌ Non disponible"

        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            gpu_info = f"✅ {gpu_name} ({gpu_memory:.1f}GB)"

        return f"""
{self.app_name} v{self.app_version}
{"=" * 50}
📁 Dossiers :
  - Modèles : {self.models_dir}
  - Cache : {self.cache_dir}
  - Logs : {self.logs_dir}
  - Sorties : {self.output_dir}

🤖 Modèles :
  - Par défaut : {self.default_model}
  - Disponibles : {", ".join(available_models) if available_models else "Aucun"}
  - Total configurés : {len(self.models)}

💻 Système :
  - GPU : {gpu_info}
  - PyTorch : {torch.__version__}
  - CUDA : {torch.version.cuda if torch.cuda.is_available() else "N/A"}

⚙️ Configuration :
  - Contrôle qualité : {"✅" if self.quality.enable_quality_checks else "❌"}
  - Amélioration style : {"✅" if self.processing.enable_style_improvement else "❌"}
  - Cache : {"✅" if self.cache.enable_cache else "❌"}
  - Mode async : {"✅" if self.processing.enable_async_processing else "❌"}
  - Debug : {"✅" if self.debug else "❌"}
"""


# Instance globale
settings = Settings()

# Validation et initialisation
settings.create_directories()
validation_issues = settings.validate_configuration()

if validation_issues:
    print("⚠️ Avertissements de configuration :")
    for issue in validation_issues:
        print(f"  • {issue}")
else:
    print("✅ Configuration validée avec succès")

# Affichage des informations
if settings.verbose:
    print(settings)
