#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fonctions utilitaires et helpers pour le générateur de bilans
"""

import asyncio
import hashlib
import json
import platform
import re
import time
from datetime import datetime, timedelta
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import psutil
import torch

from src.utils.logging import get_logger

logger = get_logger(__name__)


# ===== DECORATEURS UTILITAIRES =====


def async_timer(func: Callable) -> Callable:
    """Décorateur pour mesurer le temps d'exécution des fonctions async"""

    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = await func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.debug(f"{func.__name__} exécuté en {execution_time:.3f}s")
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"{func.__name__} a échoué après {execution_time:.3f}s: {e}")
            raise

    return wrapper


def retry_on_failure(
    max_retries: int = 3, delay: float = 1.0, exponential_backoff: bool = True
):
    """
    Décorateur pour retry automatique en cas d'échec

    Args:
        max_retries: Nombre maximum de tentatives
        delay: Délai initial entre les tentatives
        exponential_backoff: Utiliser un backoff exponentiel
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            last_exception = None
            current_delay = delay

            for attempt in range(max_retries + 1):
                try:
                    if asyncio.iscoroutinefunction(func):
                        return await func(*args, **kwargs)
                    else:
                        return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e

                    if attempt < max_retries:
                        logger.warning(
                            f"{func.__name__} échec tentative {attempt + 1}/{max_retries + 1}: {e}. "
                            f"Retry dans {current_delay:.1f}s"
                        )
                        await asyncio.sleep(current_delay)

                        if exponential_backoff:
                            current_delay *= 2
                    else:
                        logger.error(
                            f"{func.__name__} échec définitif après {max_retries + 1} tentatives"
                        )
                        raise last_exception

            raise last_exception

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            last_exception = None
            current_delay = delay

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e

                    if attempt < max_retries:
                        logger.warning(
                            f"{func.__name__} échec tentative {attempt + 1}/{max_retries + 1}: {e}. "
                            f"Retry dans {current_delay:.1f}s"
                        )
                        time.sleep(current_delay)

                        if exponential_backoff:
                            current_delay *= 2
                    else:
                        logger.error(
                            f"{func.__name__} échec définitif après {max_retries + 1} tentatives"
                        )
                        raise last_exception

            raise last_exception

        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

    return decorator


def cache_result(ttl_seconds: int = 3600):
    """
    Décorateur pour mise en cache des résultats de fonction

    Args:
        ttl_seconds: Durée de vie du cache en secondes
    """
    cache = {}

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Génération de la clé de cache
            cache_key = _generate_cache_key(func.__name__, args, kwargs)

            # Vérification du cache
            if cache_key in cache:
                result, timestamp = cache[cache_key]
                if time.time() - timestamp < ttl_seconds:
                    logger.debug(f"Cache hit pour {func.__name__}")
                    return result
                else:
                    # Suppression de l'entrée expirée
                    del cache[cache_key]

            # Exécution et mise en cache
            result = func(*args, **kwargs)
            cache[cache_key] = (result, time.time())
            logger.debug(f"Résultat mis en cache pour {func.__name__}")

            return result

        return wrapper

    return decorator


# ===== UTILITAIRES SYSTEME =====


def get_system_info() -> Dict[str, Any]:
    """Retourne les informations système détaillées"""

    info = {
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
        },
        "cpu": {
            "count": psutil.cpu_count(),
            "count_logical": psutil.cpu_count(logical=True),
            "freq_current": psutil.cpu_freq().current if psutil.cpu_freq() else None,
            "usage_percent": psutil.cpu_percent(interval=1),
        },
        "memory": {
            "total_gb": psutil.virtual_memory().total / (1024**3),
            "available_gb": psutil.virtual_memory().available / (1024**3),
            "used_percent": psutil.virtual_memory().percent,
        },
        "disk": {
            "total_gb": psutil.disk_usage("/").total / (1024**3),
            "free_gb": psutil.disk_usage("/").free / (1024**3),
            "used_percent": psutil.disk_usage("/").percent,
        },
    }

    # Informations GPU si disponible
    if torch.cuda.is_available():
        try:
            gpu_count = torch.cuda.device_count()
            current_device = torch.cuda.current_device()

            info["gpu"] = {
                "available": True,
                "count": gpu_count,
                "current_device": current_device,
                "name": torch.cuda.get_device_name(current_device),
                "memory_total_gb": torch.cuda.get_device_properties(
                    current_device
                ).total_memory
                / (1024**3),
                "memory_allocated_gb": torch.cuda.memory_allocated(current_device)
                / (1024**3),
                "memory_reserved_gb": torch.cuda.memory_reserved(current_device)
                / (1024**3),
                "cuda_version": torch.version.cuda,
            }
        except Exception as e:
            info["gpu"] = {"available": True, "error": str(e)}
    else:
        info["gpu"] = {"available": False}

    return info


def check_system_requirements() -> Tuple[bool, List[str]]:
    """
    Vérifie les prérequis système

    Returns:
        Tuple (requirements_met, list_of_issues)
    """
    issues = []

    # Vérification mémoire RAM
    memory_gb = psutil.virtual_memory().total / (1024**3)
    if memory_gb < 8:
        issues.append(f"Mémoire RAM insuffisante: {memory_gb:.1f}GB (minimum: 8GB)")
    elif memory_gb < 16:
        issues.append(f"Mémoire RAM limitée: {memory_gb:.1f}GB (recommandé: 16GB+)")

    # Vérification espace disque
    disk_free_gb = psutil.disk_usage("/").free / (1024**3)
    if disk_free_gb < 10:
        issues.append(
            f"Espace disque insuffisant: {disk_free_gb:.1f}GB libre (minimum: 10GB)"
        )

    # Vérification Python
    python_version = tuple(map(int, platform.python_version().split(".")))
    if python_version < (3, 9):
        issues.append(
            f"Version Python trop ancienne: {platform.python_version()} (minimum: 3.9)"
        )

    # Vérification CUDA si GPU disponible
    if torch.cuda.is_available():
        try:
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            if gpu_memory_gb < 6:
                issues.append(
                    f"Mémoire GPU limitée: {gpu_memory_gb:.1f}GB (recommandé: 6GB+)"
                )
        except Exception as e:
            issues.append(f"Erreur vérification GPU: {e}")

    return len(issues) == 0, issues


def monitor_resources(func: Callable) -> Callable:
    """Décorateur pour monitoring des ressources système"""

    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        # Mesures avant exécution
        start_time = time.time()
        start_memory = psutil.virtual_memory().used
        start_gpu_memory = None

        if torch.cuda.is_available():
            start_gpu_memory = torch.cuda.memory_allocated()

        try:
            # Exécution
            result = await func(*args, **kwargs)

            # Mesures après exécution
            end_time = time.time()
            end_memory = psutil.virtual_memory().used
            end_gpu_memory = None

            if torch.cuda.is_available():
                end_gpu_memory = torch.cuda.memory_allocated()

            # Calculs
            execution_time = end_time - start_time
            memory_delta = (end_memory - start_memory) / (1024**2)  # MB

            log_msg = (
                f"{func.__name__}: {execution_time:.2f}s, RAM: {memory_delta:+.1f}MB"
            )

            if start_gpu_memory is not None and end_gpu_memory is not None:
                gpu_delta = (end_gpu_memory - start_gpu_memory) / (1024**2)  # MB
                log_msg += f", GPU: {gpu_delta:+.1f}MB"

            logger.info(log_msg)

            return result

        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"{func.__name__} échec après {execution_time:.2f}s: {e}")
            raise

    return async_wrapper


# ===== UTILITAIRES FICHIERS =====


def ensure_directory(path: Union[str, Path]) -> Path:
    """
    S'assure qu'un répertoire existe, le crée si nécessaire

    Args:
        path: Chemin du répertoire

    Returns:
        Path object du répertoire
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_filename(filename: str, max_length: int = 255) -> str:
    """
    Génère un nom de fichier sûr

    Args:
        filename: Nom de fichier original
        max_length: Longueur maximale

    Returns:
        Nom de fichier sécurisé
    """
    # Caractères interdits
    forbidden_chars = r'<>:"/\|?*'

    # Remplacement des caractères interdits
    safe_name = filename
    for char in forbidden_chars:
        safe_name = safe_name.replace(char, "_")

    # Suppression des espaces en début/fin
    safe_name = safe_name.strip()

    # Limitation de la longueur
    if len(safe_name) > max_length:
        name_part, ext = Path(safe_name).stem, Path(safe_name).suffix
        max_name_length = max_length - len(ext)
        safe_name = name_part[:max_name_length] + ext

    # Ajout d'un nom par défaut si vide
    if not safe_name or safe_name == ".":
        safe_name = f"fichier_{int(time.time())}"

    return safe_name


def backup_file(
    file_path: Union[str, Path], backup_dir: Optional[Union[str, Path]] = None
) -> Path:
    """
    Crée une sauvegarde d'un fichier

    Args:
        file_path: Fichier à sauvegarder
        backup_dir: Répertoire de sauvegarde (optionnel)

    Returns:
        Chemin du fichier de sauvegarde
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Fichier non trouvé: {file_path}")

    # Répertoire de sauvegarde
    if backup_dir is None:
        backup_dir = file_path.parent / "backups"
    else:
        backup_dir = Path(backup_dir)

    ensure_directory(backup_dir)

    # Nom du fichier de sauvegarde avec timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_name = f"{file_path.stem}_{timestamp}{file_path.suffix}"
    backup_path = backup_dir / backup_name

    # Copie du fichier
    import shutil

    shutil.copy2(file_path, backup_path)

    logger.info(f"Sauvegarde créée: {backup_path}")
    return backup_path


def cleanup_old_files(
    directory: Union[str, Path], pattern: str = "*", max_age_days: int = 30
) -> int:
    """
    Nettoie les anciens fichiers d'un répertoire

    Args:
        directory: Répertoire à nettoyer
        pattern: Pattern des fichiers à nettoyer
        max_age_days: Âge maximum en jours

    Returns:
        Nombre de fichiers supprimés
    """
    directory = Path(directory)

    if not directory.exists():
        return 0

    cutoff_time = time.time() - (max_age_days * 24 * 3600)
    deleted_count = 0

    for file_path in directory.glob(pattern):
        if file_path.is_file():
            if file_path.stat().st_mtime < cutoff_time:
                try:
                    file_path.unlink()
                    deleted_count += 1
                    logger.debug(f"Fichier ancien supprimé: {file_path}")
                except Exception as e:
                    logger.warning(f"Impossible de supprimer {file_path}: {e}")

    if deleted_count > 0:
        logger.info(
            f"Nettoyage terminé: {deleted_count} fichiers supprimés de {directory}"
        )

    return deleted_count


# ===== UTILITAIRES HASH ET CRYPTO =====


def _generate_cache_key(func_name: str, args: tuple, kwargs: dict) -> str:
    """Génère une clé de cache pour les paramètres donnés"""
    # Création d'une chaîne déterministe
    key_data = {"function": func_name, "args": args, "kwargs": sorted(kwargs.items())}

    key_string = json.dumps(key_data, sort_keys=True, default=str)
    return hashlib.md5(key_string.encode()).hexdigest()


def generate_hash(data: Union[str, bytes, dict], algorithm: str = "sha256") -> str:
    """
    Génère un hash pour des données

    Args:
        data: Données à hasher
        algorithm: Algorithme de hash

    Returns:
        Hash hexadécimal
    """
    # Conversion en bytes si nécessaire
    if isinstance(data, dict):
        data = json.dumps(data, sort_keys=True)

    if isinstance(data, str):
        data = data.encode("utf-8")

    # Calcul du hash
    if algorithm == "md5":
        return hashlib.md5(data).hexdigest()
    elif algorithm == "sha1":
        return hashlib.sha1(data).hexdigest()
    elif algorithm == "sha256":
        return hashlib.sha256(data).hexdigest()
    else:
        raise ValueError(f"Algorithme de hash non supporté: {algorithm}")


def verify_hash(
    data: Union[str, bytes, dict], expected_hash: str, algorithm: str = "sha256"
) -> bool:
    """
    Vérifie l'intégrité de données avec un hash

    Args:
        data: Données à vérifier
        expected_hash: Hash attendu
        algorithm: Algorithme de hash

    Returns:
        True si les données sont intègres
    """
    actual_hash = generate_hash(data, algorithm)
    return actual_hash.lower() == expected_hash.lower()


# ===== UTILITAIRES TEXTE =====


def clean_text(
    text: str, normalize_spaces: bool = True, remove_special_chars: bool = False
) -> str:
    """
    Nettoie un texte

    Args:
        text: Texte à nettoyer
        normalize_spaces: Normaliser les espaces
        remove_special_chars: Supprimer les caractères spéciaux

    Returns:
        Texte nettoyé
    """
    if not text:
        return ""

    cleaned = text

    # Normalisation des espaces
    if normalize_spaces:
        cleaned = re.sub(r"\s+", " ", cleaned)
        cleaned = cleaned.strip()

    # Suppression des caractères spéciaux
    if remove_special_chars:
        cleaned = re.sub(r"[^\w\s\-.,;:!?()]", "", cleaned)

    return cleaned


def truncate_text(text: str, max_length: int, suffix: str = "...") -> str:
    """
    Tronque un texte à une longueur maximale

    Args:
        text: Texte à tronquer
        max_length: Longueur maximale
        suffix: Suffixe à ajouter si tronqué

    Returns:
        Texte tronqué
    """
    if len(text) <= max_length:
        return text

    # Troncature en respectant les mots
    if max_length > len(suffix):
        truncated = text[: max_length - len(suffix)]

        # Recherche du dernier espace pour éviter de couper un mot
        last_space = truncated.rfind(" ")
        if last_space > max_length * 0.8:  # Au moins 80% du texte
            truncated = truncated[:last_space]

        return truncated + suffix

    return text[:max_length]


def extract_numbers(text: str) -> List[float]:
    """
    Extrait tous les nombres d'un texte

    Args:
        text: Texte à analyser

    Returns:
        Liste des nombres trouvés
    """
    # Pattern pour nombres (entiers et décimaux)
    number_pattern = r"-?\d+(?:[.,]\d+)?"

    matches = re.findall(number_pattern, text)

    numbers = []
    for match in matches:
        try:
            # Normalisation des décimales (virgule -> point)
            normalized = match.replace(",", ".")
            numbers.append(float(normalized))
        except ValueError:
            continue

    return numbers


def format_duration(seconds: float) -> str:
    """
    Formate une durée en secondes en format lisible

    Args:
        seconds: Durée en secondes

    Returns:
        Durée formatée (ex: "2h 5min 30s")
    """
    if seconds < 0:
        return "0s"

    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    parts = []

    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}min")
    if secs > 0 or not parts:  # Toujours afficher les secondes si rien d'autre
        parts.append(f"{secs}s")

    return " ".join(parts)


def format_file_size(size_bytes: int) -> str:
    """
    Formate une taille de fichier en format lisible

    Args:
        size_bytes: Taille en bytes

    Returns:
        Taille formatée (ex: "1.5 MB")
    """
    if size_bytes == 0:
        return "0 B"

    units = ["B", "KB", "MB", "GB", "TB"]
    unit_index = 0
    size = float(size_bytes)

    while size >= 1024 and unit_index < len(units) - 1:
        size /= 1024
        unit_index += 1

    if unit_index == 0:
        return f"{int(size)} {units[unit_index]}"
    else:
        return f"{size:.1f} {units[unit_index]}"


# ===== UTILITAIRES VALIDATION =====


def validate_email(email: str) -> bool:
    """Valide un format d'email"""
    pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
    return bool(re.match(pattern, email))


def validate_phone(phone: str) -> bool:
    """Valide un numéro de téléphone français"""
    # Suppression des espaces et caractères de formatage
    cleaned = re.sub(r"[\s\-\.\(\)]", "", phone)

    # Patterns français
    patterns = [
        r"^0[1-9]\d{8},"  # 10 chiffres commençant par 0
        r"^\+33[1-9]\d{8},"  # +33 suivi de 9 chiffres
        r"^33[1-9]\d{8}"  # 33 suivi de 9 chiffres
    ]

    return any(re.match(pattern, cleaned) for pattern in patterns)


def validate_json(json_string: str) -> Tuple[bool, Optional[dict], Optional[str]]:
    """
    Valide une chaîne JSON

    Args:
        json_string: Chaîne JSON à valider

    Returns:
        Tuple (is_valid, parsed_data, error_message)
    """
    try:
        parsed = json.loads(json_string)
        return True, parsed, None
    except json.JSONDecodeError as e:
        return False, None, str(e)


# ===== UTILITAIRES ASYNC =====


async def run_with_timeout(coro, timeout_seconds: float):
    """
    Exécute une coroutine avec timeout

    Args:
        coro: Coroutine à exécuter
        timeout_seconds: Timeout en secondes

    Returns:
        Résultat de la coroutine

    Raises:
        asyncio.TimeoutError: Si timeout dépassé
    """
    try:
        return await asyncio.wait_for(coro, timeout=timeout_seconds)
    except asyncio.TimeoutError:
        logger.warning(f"Timeout atteint ({timeout_seconds}s)")
        raise


async def batch_process(
    items: List[Any],
    process_func: Callable,
    batch_size: int = 10,
    delay_between_batches: float = 0.0,
) -> List[Any]:
    """
    Traite une liste d'éléments par batch

    Args:
        items: Liste d'éléments à traiter
        process_func: Fonction de traitement (async)
        batch_size: Taille des batches
        delay_between_batches: Délai entre batches

    Returns:
        Liste des résultats
    """
    results = []

    for i in range(0, len(items), batch_size):
        batch = items[i : i + batch_size]

        # Traitement du batch
        batch_tasks = [process_func(item) for item in batch]
        batch_results = await asyncio.gather(*batch_tasks)

        results.extend(batch_results)

        # Délai entre batches si spécifié
        if delay_between_batches > 0 and i + batch_size < len(items):
            await asyncio.sleep(delay_between_batches)

    return results


# ===== CLASSE HELPER PRINCIPALE =====


class PBGHelpers:
    """Classe regroupant tous les helpers utilitaires"""

    @staticmethod
    def get_version_info() -> Dict[str, str]:
        """Retourne les informations de version"""
        return {
            "pbg_version": "2.0.0",
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "platform": platform.platform(),
        }

    @staticmethod
    def create_generation_id() -> str:
        """Crée un ID unique de génération"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        random_suffix = hashlib.md5(str(time.time()).encode()).hexdigest()[:6]
        return f"pbg_{timestamp}_{random_suffix}"

    @staticmethod
    def estimate_processing_time(word_count: int, model_speed: float = 20.0) -> float:
        """
        Estime le temps de traitement

        Args:
            word_count: Nombre de mots à générer
            model_speed: Vitesse du modèle (mots/seconde)

        Returns:
            Temps estimé en secondes
        """
        base_time = word_count / model_speed
        # Ajout d'overhead pour le traitement
        overhead_factor = 1.3
        return base_time * overhead_factor

    @staticmethod
    def check_memory_available(required_gb: float = 4.0) -> bool:
        """Vérifie si assez de mémoire disponible"""
        available_gb = psutil.virtual_memory().available / (1024**3)
        return available_gb >= required_gb
