"""
Système de logging avancé pour le générateur de bilans psychomoteurs
"""

import json
import logging
import logging.handlers
import sys
import traceback
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger as loguru_logger
from rich.console import Console
from rich.logging import RichHandler
from rich.text import Text

# Configuration par défaut
DEFAULT_LOG_FORMAT = (
    "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}"
)
DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_ROTATION = "10 MB"
DEFAULT_RETENTION = "1 month"


class StructuredFormatter(logging.Formatter):
    """Formateur pour logs structurés en JSON"""

    def format(self, record: logging.LogRecord) -> str:
        # Données de base
        log_data = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Ajout des données extra si présentes
        if hasattr(record, "extra_data"):
            log_data.update(record.extra_data)

        # Ajout des informations d'exception
        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "traceback": self.formatException(record.exc_info),
            }

        return json.dumps(log_data, ensure_ascii=False, default=str)


class PBGLoggerAdapter(logging.LoggerAdapter):
    """Adaptateur de logger avec contexte PBG"""

    def __init__(self, logger: logging.Logger, extra: Optional[Dict] = None):
        super().__init__(logger, extra or {})
        self.context_stack: List[Dict] = []

    def process(self, msg: str, kwargs: Dict) -> tuple:
        # Ajout du contexte accumulé
        extra = kwargs.get("extra", {})

        # Fusion de tous les contextes
        for context in self.context_stack:
            extra.update(context)

        # Ajout des données par défaut
        extra.update(self.extra)

        kwargs["extra"] = {"extra_data": extra}

        return msg, kwargs

    @contextmanager
    def contextualize(self, **context):
        """Context manager pour ajouter temporairement du contexte"""
        self.context_stack.append(context)
        try:
            yield
        finally:
            self.context_stack.pop()

    def bind(self, **context):
        """Lie définitivement du contexte au logger"""
        self.extra.update(context)
        return self


class LoggingManager:
    """
    Gestionnaire centralisé du logging pour PBG

    Gère plusieurs handlers avec différents niveaux et formats :
    - Console avec Rich pour le développement
    - Fichiers rotatifs pour la production
    - Logs structurés JSON pour l'analyse
    - Logs de performance pour le monitoring
    """

    def __init__(
        self,
        logs_dir: Path,
        app_name: str = "PBG",
        enable_rich: bool = True,
        enable_structured: bool = True,
    ):
        self.logs_dir = Path(logs_dir)
        self.app_name = app_name
        self.enable_rich = enable_rich
        self.enable_structured = enable_structured

        # Création du répertoire de logs
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        # Console Rich
        self.console = Console()

        # Registre des loggers créés
        self.loggers: Dict[str, PBGLoggerAdapter] = {}

        # Configuration initiale
        self._setup_root_logger()
        self._setup_loguru()

        # Statistiques de logging
        self.stats = {
            "messages_logged": 0,
            "errors_logged": 0,
            "warnings_logged": 0,
            "start_time": datetime.now(),
        }

    def _setup_root_logger(self):
        """Configure le logger racine"""
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)

        # Suppression des handlers existants
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

        # Handler console avec Rich
        if self.enable_rich:
            rich_handler = RichHandler(
                console=self.console,
                show_time=True,
                show_path=True,
                markup=True,
                rich_tracebacks=True,
                tracebacks_show_locals=True,
            )
            rich_handler.setLevel(logging.INFO)
            rich_formatter = logging.Formatter(fmt="%(message)s", datefmt="[%X]")
            rich_handler.setFormatter(rich_formatter)
            root_logger.addHandler(rich_handler)

        # Handler fichier principal
        main_log_file = self.logs_dir / f"{self.app_name.lower()}.log"
        file_handler = logging.handlers.RotatingFileHandler(
            main_log_file,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
            encoding="utf-8",
        )
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)

        # Handler pour logs structurés JSON
        if self.enable_structured:
            json_log_file = self.logs_dir / f"{self.app_name.lower()}_structured.jsonl"
            json_handler = logging.handlers.RotatingFileHandler(
                json_log_file,
                maxBytes=50 * 1024 * 1024,  # 50MB
                backupCount=3,
                encoding="utf-8",
            )
            json_handler.setLevel(logging.INFO)
            json_handler.setFormatter(StructuredFormatter())
            root_logger.addHandler(json_handler)

        # Handler pour les erreurs uniquement
        error_log_file = self.logs_dir / f"{self.app_name.lower()}_errors.log"
        error_handler = logging.handlers.RotatingFileHandler(
            error_log_file,
            maxBytes=5 * 1024 * 1024,  # 5MB
            backupCount=10,
            encoding="utf-8",
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(file_formatter)
        root_logger.addHandler(error_handler)

    def _setup_loguru(self):
        """Configure Loguru comme alternative"""
        # Suppression de la configuration par défaut de loguru
        loguru_logger.remove()

        # Configuration pour fichier avec rotation
        loguru_logger.add(
            self.logs_dir / f"{self.app_name.lower()}_loguru.log",
            format=DEFAULT_LOG_FORMAT,
            level=DEFAULT_LOG_LEVEL,
            rotation=DEFAULT_ROTATION,
            retention=DEFAULT_RETENTION,
            compression="zip",
            serialize=False,
            backtrace=True,
            diagnose=True,
        )

        # Configuration pour console si Rich n'est pas activé
        if not self.enable_rich:
            loguru_logger.add(
                sys.stderr,
                format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | <level>{message}</level>",
                level="INFO",
                colorize=True,
            )

    def get_logger(self, name: str, **context) -> PBGLoggerAdapter:
        """
        Récupère ou crée un logger adapté

        Args:
            name: Nom du logger
            **context: Contexte par défaut à attacher

        Returns:
            Logger adapté avec contexte
        """
        if name in self.loggers:
            logger_adapter = self.loggers[name]
            if context:
                logger_adapter.bind(**context)
            return logger_adapter

        # Création d'un nouveau logger
        base_logger = logging.getLogger(name)
        logger_adapter = PBGLoggerAdapter(base_logger, context)

        self.loggers[name] = logger_adapter

        return logger_adapter

    def setup_performance_logging(self, enable: bool = True):
        """Configure le logging de performance"""
        if not enable:
            return

        perf_log_file = self.logs_dir / f"{self.app_name.lower()}_performance.log"
        perf_handler = logging.handlers.TimedRotatingFileHandler(
            perf_log_file, when="midnight", interval=1, backupCount=30, encoding="utf-8"
        )
        perf_handler.setLevel(logging.INFO)

        # Format spécialisé pour les performances
        perf_formatter = logging.Formatter(
            "%(asctime)s | PERF | %(name)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
        perf_handler.setFormatter(perf_formatter)

        # Logger dédié aux performances
        perf_logger = logging.getLogger("performance")
        perf_logger.addHandler(perf_handler)
        perf_logger.setLevel(logging.INFO)
        perf_logger.propagate = False  # Éviter la duplication

    def log_generation_start(
        self, generation_id: str, model_name: str, sections_count: int
    ):
        """Log du début d'une génération"""
        logger = self.get_logger("generation")
        logger.info(
            "Début de génération",
            extra={
                "event_type": "generation_start",
                "generation_id": generation_id,
                "model_name": model_name,
                "sections_count": sections_count,
                "timestamp": datetime.now().isoformat(),
            },
        )

    def log_generation_end(
        self, generation_id: str, success: bool, duration: float, **metrics
    ):
        """Log de la fin d'une génération"""
        logger = self.get_logger("generation")

        log_data = {
            "event_type": "generation_end",
            "generation_id": generation_id,
            "success": success,
            "duration_seconds": duration,
            "timestamp": datetime.now().isoformat(),
        }
        log_data.update(metrics)

        if success:
            logger.info("Génération terminée avec succès", extra=log_data)
        else:
            logger.error("Génération échouée", extra=log_data)

    def log_model_operation(
        self, operation: str, model_name: str, duration: float = None, **details
    ):
        """Log des opérations sur les modèles"""
        logger = self.get_logger("model")

        log_data = {
            "event_type": "model_operation",
            "operation": operation,
            "model_name": model_name,
            "timestamp": datetime.now().isoformat(),
        }

        if duration is not None:
            log_data["duration_seconds"] = duration

        log_data.update(details)

        logger.info(f"Opération modèle: {operation}", extra=log_data)

    def log_cache_operation(
        self, operation: str, cache_key: str, hit: bool = None, **details
    ):
        """Log des opérations de cache"""
        logger = self.get_logger("cache")

        log_data = {
            "event_type": "cache_operation",
            "operation": operation,
            "cache_key": cache_key[:16] + "...",  # Clé tronquée pour la lisibilité
            "timestamp": datetime.now().isoformat(),
        }

        if hit is not None:
            log_data["cache_hit"] = hit

        log_data.update(details)

        logger.debug(f"Cache {operation}", extra=log_data)

    def log_quality_metrics(self, section_name: str, quality_score: float, **metrics):
        """Log des métriques de qualité"""
        logger = self.get_logger("quality")

        log_data = {
            "event_type": "quality_evaluation",
            "section_name": section_name,
            "quality_score": quality_score,
            "timestamp": datetime.now().isoformat(),
        }
        log_data.update(metrics)

        logger.info(f"Évaluation qualité: {section_name}", extra=log_data)

    def log_error_with_context(
        self, error: Exception, context: Dict[str, Any], logger_name: str = "error"
    ):
        """Log d'erreur avec contexte détaillé"""
        logger = self.get_logger(logger_name)

        error_data = {
            "event_type": "error",
            "error_type": type(error).__name__,
            "error_message": str(error),
            "context": context,
            "timestamp": datetime.now().isoformat(),
            "traceback": traceback.format_exc(),
        }

        logger.error(
            f"Erreur: {type(error).__name__}", extra=error_data, exc_info=error
        )

        self.stats["errors_logged"] += 1

    def create_audit_trail(self, user_action: str, **details):
        """Crée une trace d'audit"""
        logger = self.get_logger("audit")

        audit_data = {
            "event_type": "audit",
            "user_action": user_action,
            "timestamp": datetime.now().isoformat(),
            "session_id": details.get("session_id"),
            "user_id": details.get("user_id"),
        }
        audit_data.update(details)

        logger.info(f"Action utilisateur: {user_action}", extra=audit_data)

    def get_log_files(self) -> List[Path]:
        """Retourne la liste des fichiers de log"""
        return list(self.logs_dir.glob("*.log")) + list(self.logs_dir.glob("*.jsonl"))

    def get_logs_summary(self, last_hours: int = 24) -> Dict[str, Any]:
        """Génère un résumé des logs"""
        summary = {
            "total_log_files": len(self.get_log_files()),
            "logs_directory": str(self.logs_dir),
            "uptime_hours": (datetime.now() - self.stats["start_time"]).total_seconds()
            / 3600,
            "messages_logged": self.stats["messages_logged"],
            "errors_logged": self.stats["errors_logged"],
            "warnings_logged": self.stats["warnings_logged"],
        }

        # Analyse des fichiers de log récents
        main_log_file = self.logs_dir / f"{self.app_name.lower()}.log"
        if main_log_file.exists():
            summary["main_log_size_mb"] = main_log_file.stat().st_size / (1024 * 1024)

        return summary

    def cleanup_old_logs(self, max_age_days: int = 30) -> int:
        """Nettoie les anciens fichiers de log"""
        from src.utils.helpers import cleanup_old_files

        total_deleted = 0

        # Nettoyage par type de fichier
        patterns = ["*.log.*", "*.jsonl.*", "*.zip"]

        for pattern in patterns:
            deleted = cleanup_old_files(self.logs_dir, pattern, max_age_days)
            total_deleted += deleted

        if total_deleted > 0:
            logger = self.get_logger("maintenance")
            logger.info(f"Nettoyage logs: {total_deleted} fichiers supprimés")

        return total_deleted

    def export_logs_for_analysis(self, output_file: Path, format_type: str = "json"):
        """Exporte les logs pour analyse"""
        logger = self.get_logger("export")

        if format_type == "json":
            # Export des logs structurés
            json_log_file = self.logs_dir / f"{self.app_name.lower()}_structured.jsonl"

            if json_log_file.exists():
                import shutil

                shutil.copy2(json_log_file, output_file)
                logger.info(f"Logs exportés vers {output_file}")
            else:
                logger.warning("Aucun log structuré disponible pour l'export")

        else:
            raise ValueError(f"Format d'export non supporté: {format_type}")


# Instance globale du gestionnaire de logging
_logging_manager: Optional[LoggingManager] = None


def setup_logging(
    level: str = DEFAULT_LOG_LEVEL,
    logs_dir: Optional[Path] = None,
    enable_rich: bool = True,
    enable_structured: bool = True,
    enable_performance: bool = False,
) -> LoggingManager:
    """
    Configure le système de logging global

    Args:
        level: Niveau de log
        logs_dir: Répertoire des logs
        enable_rich: Activer Rich pour la console
        enable_structured: Activer les logs structurés
        enable_performance: Activer le logging de performance

    Returns:
        Gestionnaire de logging configuré
    """
    global _logging_manager

    if logs_dir is None:
        logs_dir = Path.cwd() / "logs"

    _logging_manager = LoggingManager(
        logs_dir=logs_dir, enable_rich=enable_rich, enable_structured=enable_structured
    )

    # Configuration du niveau global
    logging.getLogger().setLevel(getattr(logging, level.upper()))

    # Configuration optionnelle des performances
    if enable_performance:
        _logging_manager.setup_performance_logging(True)

    return _logging_manager


def get_logger(name: str, **context) -> PBGLoggerAdapter:
    """
    Récupère un logger configuré

    Args:
        name: Nom du logger
        **context: Contexte par défaut

    Returns:
        Logger adapté
    """
    global _logging_manager

    if _logging_manager is None:
        _logging_manager = setup_logging()

    return _logging_manager.get_logger(name, **context)


def get_loguru_logger():
    """Retourne le logger Loguru configuré"""
    return loguru_logger


@contextmanager
def log_execution_time(operation_name: str, logger_name: str = "performance"):
    """
    Context manager pour logger le temps d'exécution

    Args:
        operation_name: Nom de l'opération
        logger_name: Nom du logger
    """
    logger = get_logger(logger_name)
    start_time = datetime.now()

    try:
        yield
        duration = (datetime.now() - start_time).total_seconds()
        logger.info(
            f"Opération terminée: {operation_name}",
            extra={
                "operation": operation_name,
                "duration_seconds": duration,
                "success": True,
            },
        )
    except Exception as e:
        duration = (datetime.now() - start_time).total_seconds()
        logger.error(
            f"Opération échouée: {operation_name}",
            extra={
                "operation": operation_name,
                "duration_seconds": duration,
                "success": False,
                "error": str(e),
            },
            exc_info=e,
        )
        raise


@contextmanager
def log_context(**context):
    """
    Context manager pour ajouter du contexte temporaire aux logs

    Args:
        **context: Contexte à ajouter
    """
    # Configuration temporaire pour tous les loggers existants
    original_contexts = {}

    if _logging_manager:
        for name, logger_adapter in _logging_manager.loggers.items():
            original_contexts[name] = logger_adapter.extra.copy()
            logger_adapter.bind(**context)

    try:
        yield
    finally:
        # Restauration des contextes originaux
        if _logging_manager:
            for name, original_context in original_contexts.items():
                if name in _logging_manager.loggers:
                    _logging_manager.loggers[name].extra = original_context


# Classes spécialisées pour différents types de logs


class GenerationLogger:
    """Logger spécialisé pour les opérations de génération"""

    def __init__(self, generation_id: str):
        self.generation_id = generation_id
        self.logger = get_logger("generation", generation_id=generation_id)
        self.start_time = datetime.now()

    def log_section_start(self, section_name: str):
        self.logger.info(
            f"Début génération section: {section_name}",
            extra={"section_name": section_name, "event": "section_start"},
        )

    def log_section_end(
        self, section_name: str, word_count: int, quality_score: float = None
    ):
        self.logger.info(
            f"Section terminée: {section_name}",
            extra={
                "section_name": section_name,
                "event": "section_end",
                "word_count": word_count,
                "quality_score": quality_score,
            },
        )

    def log_retry(self, section_name: str, attempt: int, reason: str):
        self.logger.warning(
            f"Retry section {section_name} (tentative {attempt})",
            extra={
                "section_name": section_name,
                "event": "retry",
                "attempt": attempt,
                "reason": reason,
            },
        )

    def log_completion(self, total_words: int, quality_score: float = None):
        duration = (datetime.now() - self.start_time).total_seconds()
        self.logger.info(
            "Génération terminée",
            extra={
                "event": "generation_complete",
                "total_words": total_words,
                "duration_seconds": duration,
                "quality_score": quality_score,
            },
        )


class PerformanceLogger:
    """Logger spécialisé pour les métriques de performance"""

    def __init__(self):
        self.logger = get_logger("performance")

    def log_memory_usage(
        self, operation: str, memory_before: float, memory_after: float
    ):
        self.logger.info(
            f"Mémoire {operation}",
            extra={
                "operation": operation,
                "memory_before_mb": memory_before,
                "memory_after_mb": memory_after,
                "memory_delta_mb": memory_after - memory_before,
            },
        )

    def log_gpu_usage(self, operation: str, gpu_before: float, gpu_after: float):
        self.logger.info(
            f"GPU {operation}",
            extra={
                "operation": operation,
                "gpu_before_mb": gpu_before,
                "gpu_after_mb": gpu_after,
                "gpu_delta_mb": gpu_after - gpu_before,
            },
        )

    def log_model_stats(self, model_name: str, load_time: float, memory_usage: float):
        self.logger.info(
            f"Statistiques modèle: {model_name}",
            extra={
                "model_name": model_name,
                "load_time_seconds": load_time,
                "memory_usage_mb": memory_usage,
            },
        )
