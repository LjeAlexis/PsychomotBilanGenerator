"""
Gestionnaire de modèles LLM avec optimisations mémoire et performance
"""

import gc
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from config.settings import ModelConfig, Settings
from src.utils.logging import get_logger


class ModelManager:
    """
    Gestionnaire centralisé des modèles LLM

    Fonctionnalités :
    - Chargement optimisé avec quantification
    - Gestion mémoire intelligente
    - Configuration automatique par modèle
    - Support multi-GPU
    - Cache des configurations
    """

    def __init__(self, config: Settings):
        self.config = config
        self.logger = get_logger(self.__class__.__name__)

        # État actuel
        self.current_model: Optional[PreTrainedModel] = None
        self.current_tokenizer: Optional[PreTrainedTokenizer] = None
        self.current_model_name: Optional[str] = None
        self.model_config: Optional[ModelConfig] = None

        # Métriques
        self.load_time: float = 0.0
        self.memory_usage: Dict[str, float] = {}

        # Cache des configurations testées
        self._config_cache: Dict[str, Dict] = {}

        self.logger.info("ModelManager initialisé")

    async def load_model(self, model_name: str, force_reload: bool = False) -> None:
        """
        Charge un modèle avec configuration optimale

        Args:
            model_name: Nom du modèle à charger
            force_reload: Forcer le rechargement même si déjà chargé
        """
        # Vérifier si le modèle est déjà chargé
        if (
            not force_reload
            and self.current_model is not None
            and self.current_model_name == model_name
        ):
            self.logger.info(f"Modèle {model_name} déjà chargé")
            return

        start_time = time.time()
        self.logger.info(f"Chargement du modèle: {model_name}")

        # Nettoyage du modèle précédent
        if self.current_model is not None:
            await self.unload_current_model()

        # Configuration du modèle
        self.model_config = self.config.get_model_config(model_name)
        model_path = self.config.get_model_path(model_name)

        # Configuration optimale
        load_config = self._get_optimal_config(model_path, self.model_config)

        try:
            # Chargement du tokenizer
            self.logger.info("Chargement du tokenizer...")
            self.current_tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                use_fast=True,
                trust_remote_code=True,
                padding_side="left",  # Important pour la génération
            )

            # Configuration du tokenizer
            self._configure_tokenizer()

            # Chargement du modèle
            self.logger.info(f"Chargement du modèle avec config: {load_config}")
            self.current_model = AutoModelForCausalLM.from_pretrained(
                model_path, trust_remote_code=True, **load_config
            )

            # Configuration post-chargement
            self._configure_model()

            # Métriques mémoire
            self.memory_usage = self._get_memory_usage()
            self.load_time = time.time() - start_time
            self.current_model_name = model_name

            self.logger.info(
                f"Modèle {model_name} chargé en {self.load_time:.2f}s. "
                f"Mémoire: {self.memory_usage}"
            )

        except Exception as e:
            self.logger.error(f"Erreur lors du chargement de {model_name}: {e}")
            # Tentative de fallback CPU
            if torch.cuda.is_available():
                self.logger.info("Tentative de fallback CPU...")
                await self._fallback_cpu_load(model_path)
            else:
                raise RuntimeError(f"Impossible de charger le modèle {model_name}: {e}")

    def _get_optimal_config(
        self, model_path: str, model_config: ModelConfig
    ) -> Dict[str, Any]:
        """Retourne la configuration optimale pour le modèle"""

        # Vérifier le cache de configuration
        cache_key = f"{model_path}_{model_config.quantization}"
        if cache_key in self._config_cache:
            self.logger.debug("Configuration récupérée du cache")
            return self._config_cache[cache_key].copy()

        config = {}

        if torch.cuda.is_available():
            # Configuration GPU
            config["device_map"] = "auto"
            config["torch_dtype"] = torch.float16

            # Configuration de la quantification
            if model_config.quantization == "4bit":
                config["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_quant_storage=torch.uint8,
                )

            elif model_config.quantization == "8bit":
                config["quantization_config"] = BitsAndBytesConfig(
                    load_in_8bit=True, llm_int8_enable_fp32_cpu_offload=True
                )

            # Configuration de l'attention
            config["attn_implementation"] = model_config.attention_impl

            # Optimisations mémoire
            config["low_cpu_mem_usage"] = True
            config["use_cache"] = True

        else:
            # Configuration CPU
            config["torch_dtype"] = torch.float32
            config["device_map"] = "cpu"
            config["low_cpu_mem_usage"] = True

        # Optimisations spécifiques par modèle
        model_path_lower = str(model_path).lower()

        if "mistral" in model_path_lower:
            # Optimisations Mistral
            if torch.cuda.is_available():
                config["use_flash_attention_2"] = (
                    model_config.attention_impl == "flash_attention_2"
                )

        elif "llama" in model_path_lower:
            # Optimisations Llama
            config["rope_scaling"] = None

        elif "qwen" in model_path_lower:
            # Optimisations Qwen
            config["trust_remote_code"] = True

        elif "biomistral" in model_path_lower:
            # Optimisations médicales
            config["torch_dtype"] = torch.float16  # Précision importante

        # Mise en cache
        self._config_cache[cache_key] = config.copy()

        return config

    def _configure_tokenizer(self) -> None:
        """Configure le tokenizer après chargement"""
        if self.current_tokenizer.pad_token is None:
            if self.current_tokenizer.eos_token:
                self.current_tokenizer.pad_token = self.current_tokenizer.eos_token
            else:
                self.current_tokenizer.add_special_tokens({"pad_token": "[PAD]"})

        # Assurer le padding à gauche pour la génération
        self.current_tokenizer.padding_side = "left"

        self.logger.debug(
            f"Tokenizer configuré: vocab_size={self.current_tokenizer.vocab_size}"
        )

    def _configure_model(self) -> None:
        """Configure le modèle après chargement"""
        if self.current_model is None:
            return

        # Mode évaluation
        self.current_model.eval()

        # Gradient checkpointing pour économiser la mémoire
        if (
            hasattr(self.current_model, "gradient_checkpointing_enable")
            and self.config.processing.enable_advanced_nlp
        ):
            self.current_model.gradient_checkpointing_enable()

        # Mise à jour du vocabulaire si nécessaire
        if len(self.current_tokenizer) > self.current_model.config.vocab_size:
            self.current_model.resize_token_embeddings(len(self.current_tokenizer))

        self.logger.debug("Modèle configuré en mode évaluation")

    async def _fallback_cpu_load(self, model_path: str) -> None:
        """Chargement de fallback sur CPU"""
        self.logger.warning("Chargement en mode CPU de secours")

        try:
            # Configuration CPU simplifiée
            cpu_config = {
                "torch_dtype": torch.float32,
                "device_map": "cpu",
                "low_cpu_mem_usage": True,
            }

            self.current_model = AutoModelForCausalLM.from_pretrained(
                model_path, **cpu_config
            )

            self._configure_model()
            self.logger.info("Modèle chargé avec succès en mode CPU")

        except Exception as e:
            raise RuntimeError(f"Échec du chargement CPU: {e}")

    async def generate_text(
        self,
        system_prompt: str,
        user_prompt: str,
        max_new_tokens: int = 800,
        temperature: float = 0.3,
        top_p: float = 0.9,
        repetition_penalty: float = 1.15,
        do_sample: bool = True,
        **kwargs,
    ) -> str:
        """
        Génère du texte avec le modèle actuel

        Args:
            system_prompt: Prompt système
            user_prompt: Prompt utilisateur
            max_new_tokens: Nombre maximum de tokens à générer
            temperature: Contrôle de la créativité
            top_p: Nucleus sampling
            repetition_penalty: Pénalité de répétition
            do_sample: Utiliser l'échantillonnage
            **kwargs: Paramètres additionnels

        Returns:
            Texte généré
        """
        if self.current_model is None or self.current_tokenizer is None:
            raise RuntimeError("Aucun modèle chargé")

        # Construction du prompt selon le format du modèle
        formatted_prompt = self._format_prompt(system_prompt, user_prompt)

        # Tokenisation
        inputs = self.current_tokenizer(
            formatted_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.model_config.max_context - max_new_tokens,
            padding=False,
        )

        # Transfert sur le bon device
        if torch.cuda.is_available() and self.current_model.device.type == "cuda":
            inputs = {k: v.to(self.current_model.device) for k, v in inputs.items()}

        # Paramètres de génération
        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "repetition_penalty": repetition_penalty,
            "do_sample": do_sample,
            "pad_token_id": self.current_tokenizer.pad_token_id,
            "eos_token_id": self.current_tokenizer.eos_token_id,
            "use_cache": True,
            **kwargs,
        }

        # Génération avec gestion mémoire
        try:
            with torch.inference_mode():
                # Utilisation d'autocast pour optimiser la mémoire
                with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                    outputs = self.current_model.generate(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs.get("attention_mask"),
                        **generation_kwargs,
                    )

            # Décodage de la réponse
            generated_text = self.current_tokenizer.decode(
                outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True
            )

            # Extraction de la partie générée
            response = self._extract_response(generated_text, formatted_prompt)

            # Nettoyage mémoire
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            return response

        except torch.cuda.OutOfMemoryError:
            self.logger.error("Mémoire GPU insuffisante")
            torch.cuda.empty_cache()
            raise RuntimeError("Mémoire GPU insuffisante pour la génération")

        except Exception as e:
            self.logger.error(f"Erreur lors de la génération: {e}")
            raise

    def _format_prompt(self, system_prompt: str, user_prompt: str) -> str:
        """Formate le prompt selon le modèle"""

        model_name = self.current_model_name.lower()

        if "mistral" in model_name or "mixtral" in model_name:
            # Format Mistral/Mixtral
            if hasattr(self.current_tokenizer, "apply_chat_template"):
                messages = [
                    {"role": "user", "content": f"{system_prompt}\n\n{user_prompt}"}
                ]
                return self.current_tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            else:
                return f"<s>[INST] {system_prompt}\n\n{user_prompt} [/INST]"

        elif "qwen" in model_name:
            # Format Qwen
            if hasattr(self.current_tokenizer, "apply_chat_template"):
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ]
                return self.current_tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            else:
                return f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"

        elif "llama" in model_name:
            # Format Llama
            return f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

        else:
            # Format générique
            return f"### Système:\n{system_prompt}\n\n### Utilisateur:\n{user_prompt}\n\n### Assistant:\n"

    def _extract_response(self, generated_text: str, original_prompt: str) -> str:
        """Extrait la réponse générée du texte complet"""

        # Suppression du prompt original
        if generated_text.startswith(original_prompt):
            response = generated_text[len(original_prompt) :].strip()
        else:
            response = generated_text

        # Nettoyage selon le format du modèle
        model_name = self.current_model_name.lower()

        if "mistral" in model_name:
            if "[/INST]" in response:
                response = response.split("[/INST]")[-1].strip()

        elif "qwen" in model_name:
            if "<|im_start|>assistant" in response:
                response = response.split("<|im_start|>assistant")[-1].strip()
            if "<|im_end|>" in response:
                response = response.split("<|im_end|>")[0].strip()

        elif "llama" in model_name:
            if "<|start_header_id|>assistant<|end_header_id|>" in response:
                response = response.split(
                    "<|start_header_id|>assistant<|end_header_id|>"
                )[-1].strip()
            if "<|eot_id|>" in response:
                response = response.split("<|eot_id|>")[0].strip()

        else:
            # Format générique
            if "### Assistant:" in response:
                response = response.split("### Assistant:")[-1].strip()

        # Nettoyage final
        response = response.strip()
        response = response.replace("\u200b", "")  # Zero-width spaces

        return response

    def _get_memory_usage(self) -> Dict[str, float]:
        """Retourne l'utilisation mémoire actuelle"""
        usage = {}

        if torch.cuda.is_available():
            usage["gpu_allocated_gb"] = torch.cuda.memory_allocated() / 1024**3
            usage["gpu_reserved_gb"] = torch.cuda.memory_reserved() / 1024**3
            usage["gpu_total_gb"] = (
                torch.cuda.get_device_properties(0).total_memory / 1024**3
            )
            usage["gpu_utilization"] = usage["gpu_allocated_gb"] / usage["gpu_total_gb"]

        # Mémoire CPU (approximative)
        import psutil

        process = psutil.Process()
        usage["cpu_memory_gb"] = process.memory_info().rss / 1024**3

        return usage

    async def unload_current_model(self) -> None:
        """Décharge le modèle actuel de la mémoire"""
        if self.current_model is not None:
            self.logger.info(f"Déchargement du modèle: {self.current_model_name}")

            # Suppression des références
            del self.current_model
            del self.current_tokenizer

            # Nettoyage mémoire
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            # Reset des variables
            self.current_model = None
            self.current_tokenizer = None
            self.current_model_name = None
            self.model_config = None

            self.logger.info("Modèle déchargé")

    def get_model_info(self) -> Dict[str, Any]:
        """Retourne les informations du modèle actuel"""
        if self.current_model is None:
            return {"status": "no_model_loaded"}

        return {
            "status": "loaded",
            "model_name": self.current_model_name,
            "model_config": self.model_config.model_dump() if self.model_config else {},
            "load_time": self.load_time,
            "memory_usage": self.memory_usage,
            "device": str(self.current_model.device),
            "dtype": str(self.current_model.dtype),
            "num_parameters": self.current_model.num_parameters(),
            "vocab_size": self.current_tokenizer.vocab_size
            if self.current_tokenizer
            else None,
        }

    def is_model_loaded(self, model_name: Optional[str] = None) -> bool:
        """Vérifie si un modèle est chargé"""
        if self.current_model is None:
            return False

        if model_name:
            return self.current_model_name == model_name

        return True

    def get_generation_config(self, **overrides) -> Dict[str, Any]:
        """Retourne la configuration de génération optimale"""
        if self.model_config is None:
            raise RuntimeError("Aucun modèle chargé")

        config = {
            "max_new_tokens": self.model_config.max_tokens,
            "temperature": self.model_config.temperature,
            "top_p": self.model_config.top_p,
            "repetition_penalty": self.model_config.repetition_penalty,
            "do_sample": True,
            "pad_token_id": self.current_tokenizer.pad_token_id
            if self.current_tokenizer
            else None,
            "eos_token_id": self.current_tokenizer.eos_token_id
            if self.current_tokenizer
            else None,
        }

        # Application des overrides
        config.update(overrides)

        return config

    async def cleanup(self) -> None:
        """Nettoyage complet"""
        await self.unload_current_model()
        self._config_cache.clear()
        self.logger.info("ModelManager nettoyé")

    def __del__(self):
        """Nettoyage automatique"""
        try:
            if self.current_model is not None:
                del self.current_model
                del self.current_tokenizer
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        except:
            pass  # Ignore les erreurs lors du nettoyage
