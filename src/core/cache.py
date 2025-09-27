"""
Système de cache intelligent pour optimiser les performances de génération

Le cache permet de :
- Éviter de régénérer des sections identiques
- Accélérer les itérations de développement
- Économiser les ressources GPU/CPU
- Persister les résultats entre sessions
- Gérer automatiquement l'expiration et la taille
"""

import asyncio
import hashlib
import json
import pickle
import time
import zlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import aiofiles
from diskcache import Cache

from config.settings import CacheConfig
from src.utils.logging import get_logger


class CacheKey:
    """Classe pour générer des clés de cache cohérentes"""

    @staticmethod
    def generate(
        model_name: str,
        section_title: str,
        section_notes: Any,
        generation_params: Optional[Dict] = None,
        version: str = "v1",
    ) -> str:
        """
        Génère une clé de cache unique et déterministe

        Args:
            model_name: Nom du modèle utilisé
            section_title: Titre de la section
            section_notes: Contenu des notes
            generation_params: Paramètres de génération
            version: Version du cache (pour invalidation)

        Returns:
            Clé de cache hexadécimale
        """
        # Normalisation des paramètres de génération
        gen_params = generation_params or {}
        normalized_params = {
            k: v
            for k, v in sorted(gen_params.items())
            if k in ["temperature", "max_new_tokens", "top_p", "repetition_penalty"]
        }

        # Création de la chaîne de données à hasher
        data_string = json.dumps(
            {
                "version": version,
                "model": model_name,
                "section": section_title,
                "notes": section_notes,
                "params": normalized_params,
            },
            sort_keys=True,
            ensure_ascii=False,
        )

        # Hash SHA-256 pour une clé unique
        return hashlib.sha256(data_string.encode("utf-8")).hexdigest()


class CacheStats:
    """Statistiques du cache"""

    def __init__(self):
        self.hits: int = 0
        self.misses: int = 0
        self.sets: int = 0
        self.evictions: int = 0
        self.size_bytes: int = 0
        self.cleanup_count: int = 0

    @property
    def hit_rate(self) -> float:
        """Taux de succès du cache"""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    @property
    def size_mb(self) -> float:
        """Taille du cache en MB"""
        return self.size_bytes / 1024 / 1024

    def to_dict(self) -> Dict[str, Any]:
        """Conversion en dictionnaire"""
        return {
            "hits": self.hits,
            "misses": self.misses,
            "sets": self.sets,
            "evictions": self.evictions,
            "hit_rate": self.hit_rate,
            "size_mb": self.size_mb,
            "cleanup_count": self.cleanup_count,
        }


class CacheManager:
    """
    Gestionnaire de cache avancé pour la génération de bilans

    Utilise diskcache pour la persistance avec compression et expiration.
    Supporte les opérations asynchrones et le nettoyage automatique.
    """

    def __init__(self, cache_dir: Path, config: CacheConfig):
        self.cache_dir = Path(cache_dir)
        self.config = config
        self.logger = get_logger(self.__class__.__name__)

        # Cache principal (diskcache pour persistance)
        self.disk_cache: Optional[Cache] = None

        # Cache mémoire pour accès rapide (LRU)
        self._memory_cache: Dict[str, Tuple[Any, float]] = {}
        self._memory_cache_max_size = 100  # Nombre d'entrées en mémoire

        # Statistiques
        self.stats = CacheStats()

        # Configuration
        self._setup_cache_directory()

        self.logger.info(f"CacheManager initialisé: {self.cache_dir}")

    def _setup_cache_directory(self) -> None:
        """Configure le répertoire de cache"""
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Fichier de configuration du cache
        config_file = self.cache_dir / "cache_config.json"
        if not config_file.exists():
            with open(config_file, "w") as f:
                json.dump(
                    {
                        "created": datetime.now().isoformat(),
                        "version": "1.0",
                        "description": "Cache PsychomotBilanGenerator",
                    },
                    f,
                    indent=2,
                )

    async def load_cache(self) -> None:
        """Initialise le cache disque"""
        if not self.config.enable_cache:
            self.logger.info("Cache désactivé par configuration")
            return

        try:
            # Configuration du cache disque
            cache_size_bytes = self.config.cache_size_mb * 1024 * 1024

            self.disk_cache = Cache(
                directory=str(self.cache_dir / "disk_cache"),
                size_limit=cache_size_bytes,
                eviction_policy="least-recently-used",
                statistics=True,
            )

            # Chargement des statistiques existantes
            await self._load_stats()

            # Nettoyage automatique si activé
            if self.config.auto_cleanup:
                await self._cleanup_expired_entries()

            self.logger.info(f"Cache chargé: {len(self.disk_cache)} entrées")

        except Exception as e:
            self.logger.error(f"Erreur lors du chargement du cache: {e}")
            self.disk_cache = None

    async def get(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """
        Récupère une entrée du cache

        Args:
            cache_key: Clé de cache

        Returns:
            Données cachées ou None si non trouvé
        """
        if not self.config.enable_cache or not self.disk_cache:
            return None

        # Vérification du cache mémoire d'abord
        if cache_key in self._memory_cache:
            data, timestamp = self._memory_cache[cache_key]
            if self._is_cache_entry_valid(timestamp):
                self.stats.hits += 1
                self.logger.debug(f"Cache hit (mémoire): {cache_key[:16]}...")
                return data
            else:
                # Entrée expirée
                del self._memory_cache[cache_key]

        # Vérification du cache disque
        try:
            cached_data = self.disk_cache.get(cache_key)
            if cached_data:
                # Décompression si nécessaire
                if self.config.cache_compression and isinstance(cached_data, bytes):
                    cached_data = pickle.loads(zlib.decompress(cached_data))

                # Vérification de l'expiration
                if self._is_cache_entry_valid(cached_data.get("timestamp", 0)):
                    # Mise en cache mémoire
                    self._add_to_memory_cache(cache_key, cached_data)

                    self.stats.hits += 1
                    self.logger.debug(f"Cache hit (disque): {cache_key[:16]}...")
                    return cached_data
                else:
                    # Suppression de l'entrée expirée
                    del self.disk_cache[cache_key]

        except Exception as e:
            self.logger.warning(f"Erreur lecture cache {cache_key[:16]}...: {e}")

        self.stats.misses += 1
        return None

    async def set(
        self, cache_key: str, data: Dict[str, Any], expire_hours: Optional[int] = None
    ) -> bool:
        """
        Stocke une entrée dans le cache

        Args:
            cache_key: Clé de cache
            data: Données à cacher
            expire_hours: Expiration personnalisée en heures

        Returns:
            True si le stockage a réussi
        """
        if not self.config.enable_cache or not self.disk_cache:
            return False

        try:
            # Ajout des métadonnées
            cache_entry = {
                **data,
                "timestamp": time.time(),
                "cache_key": cache_key,
                "expire_hours": expire_hours or self.config.cache_ttl_hours,
            }

            # Compression si activée
            if self.config.cache_compression:
                compressed_data = zlib.compress(pickle.dumps(cache_entry))
                self.disk_cache.set(cache_key, compressed_data)
            else:
                self.disk_cache.set(cache_key, cache_entry)

            # Ajout au cache mémoire
            self._add_to_memory_cache(cache_key, cache_entry)

            self.stats.sets += 1
            self.logger.debug(f"Cache set: {cache_key[:16]}...")

            return True

        except Exception as e:
            self.logger.error(f"Erreur stockage cache {cache_key[:16]}...: {e}")
            return False

    def get_cache_key(
        self,
        model_name: str,
        section_title: str,
        section_notes: Any,
        generation_params: Optional[Dict] = None,
    ) -> str:
        """Génère une clé de cache pour les paramètres donnés"""
        return CacheKey.generate(
            model_name=model_name,
            section_title=section_title,
            section_notes=section_notes,
            generation_params=generation_params,
        )

    async def invalidate_model(self, model_name: str) -> int:
        """
        Invalide toutes les entrées d'un modèle

        Args:
            model_name: Nom du modèle

        Returns:
            Nombre d'entrées supprimées
        """
        if not self.disk_cache:
            return 0

        deleted_count = 0
        keys_to_delete = []

        # Recherche des clés à supprimer
        for key in self.disk_cache:
            try:
                data = self.disk_cache.get(key)
                if data and data.get("metadata", {}).get("model_name") == model_name:
                    keys_to_delete.append(key)
            except:
                continue

        # Suppression
        for key in keys_to_delete:
            try:
                del self.disk_cache[key]
                if key in self._memory_cache:
                    del self._memory_cache[key]
                deleted_count += 1
            except:
                continue

        self.logger.info(
            f"Invalidé {deleted_count} entrées pour le modèle {model_name}"
        )
        return deleted_count

    async def clear_cache(self, confirm: bool = False) -> bool:
        """
        Vide complètement le cache

        Args:
            confirm: Confirmation de suppression

        Returns:
            True si le cache a été vidé
        """
        if not confirm:
            self.logger.warning("clear_cache() nécessite confirm=True")
            return False

        if not self.disk_cache:
            return True

        try:
            # Vidage du cache disque
            self.disk_cache.clear()

            # Vidage du cache mémoire
            self._memory_cache.clear()

            # Reset des statistiques
            self.stats = CacheStats()

            self.logger.info("Cache complètement vidé")
            return True

        except Exception as e:
            self.logger.error(f"Erreur lors du vidage du cache: {e}")
            return False

    async def cleanup_expired_entries(self) -> int:
        """
        Nettoie les entrées expirées du cache

        Returns:
            Nombre d'entrées supprimées
        """
        return await self._cleanup_expired_entries()

    async def _cleanup_expired_entries(self) -> int:
        """Nettoyage interne des entrées expirées"""
        if not self.disk_cache:
            return 0

        deleted_count = 0
        current_time = time.time()
        keys_to_delete = []

        # Recherche des entrées expirées
        for key in self.disk_cache:
            try:
                data = self.disk_cache.get(key)
                if data and not self._is_cache_entry_valid(data.get("timestamp", 0)):
                    keys_to_delete.append(key)
            except:
                # Entrée corrompue, la supprimer aussi
                keys_to_delete.append(key)

        # Suppression des entrées expirées
        for key in keys_to_delete:
            try:
                del self.disk_cache[key]
                if key in self._memory_cache:
                    del self._memory_cache[key]
                deleted_count += 1
            except:
                continue

        if deleted_count > 0:
            self.stats.cleanup_count += 1
            self.logger.info(f"Nettoyage: {deleted_count} entrées expirées supprimées")

        return deleted_count

    def _is_cache_entry_valid(self, timestamp: float) -> bool:
        """Vérifie si une entrée de cache est encore valide"""
        if timestamp <= 0:
            return False

        current_time = time.time()
        expiry_time = timestamp + (self.config.cache_ttl_hours * 3600)

        return current_time < expiry_time

    def _add_to_memory_cache(self, key: str, data: Dict[str, Any]) -> None:
        """Ajoute une entrée au cache mémoire avec gestion LRU"""
        # Éviction si le cache mémoire est plein
        if len(self._memory_cache) >= self._memory_cache_max_size:
            # Suppression de l'entrée la plus ancienne
            oldest_key = min(
                self._memory_cache.keys(), key=lambda k: self._memory_cache[k][1]
            )
            del self._memory_cache[oldest_key]

        # Ajout de la nouvelle entrée
        self._memory_cache[key] = (data, time.time())

    async def _load_stats(self) -> None:
        """Charge les statistiques depuis le fichier"""
        stats_file = self.cache_dir / "cache_stats.json"

        try:
            if stats_file.exists():
                async with aiofiles.open(stats_file, "r") as f:
                    stats_data = json.loads(await f.read())

                self.stats.hits = stats_data.get("hits", 0)
                self.stats.misses = stats_data.get("misses", 0)
                self.stats.sets = stats_data.get("sets", 0)
                self.stats.evictions = stats_data.get("evictions", 0)
                self.stats.cleanup_count = stats_data.get("cleanup_count", 0)

        except Exception as e:
            self.logger.warning(f"Erreur chargement statistiques: {e}")

    async def save_cache(self) -> None:
        """Sauvegarde le cache et les statistiques"""
        if not self.config.enable_cache:
            return

        try:
            # Sauvegarde des statistiques
            await self._save_stats()

            # Le cache disque se sauvegarde automatiquement
            self.logger.debug("Cache sauvegardé")

        except Exception as e:
            self.logger.error(f"Erreur sauvegarde cache: {e}")

    async def _save_stats(self) -> None:
        """Sauvegarde les statistiques"""
        stats_file = self.cache_dir / "cache_stats.json"

        try:
            stats_data = {
                **self.stats.to_dict(),
                "last_updated": datetime.now().isoformat(),
                "cache_config": {
                    "size_mb": self.config.cache_size_mb,
                    "ttl_hours": self.config.cache_ttl_hours,
                    "compression": self.config.cache_compression,
                },
            }

            async with aiofiles.open(stats_file, "w") as f:
                await f.write(json.dumps(stats_data, indent=2))

        except Exception as e:
            self.logger.warning(f"Erreur sauvegarde statistiques: {e}")

    def get_cache_info(self) -> Dict[str, Any]:
        """Retourne les informations détaillées du cache"""
        info = {
            "enabled": self.config.enable_cache,
            "stats": self.stats.to_dict(),
            "config": {
                "cache_dir": str(self.cache_dir),
                "size_limit_mb": self.config.cache_size_mb,
                "ttl_hours": self.config.cache_ttl_hours,
                "compression": self.config.cache_compression,
                "auto_cleanup": self.config.auto_cleanup,
            },
        }

        if self.disk_cache:
            info.update(
                {
                    "disk_cache_size": len(self.disk_cache),
                    "memory_cache_size": len(self._memory_cache),
                    "disk_stats": dict(self.disk_cache.stats)
                    if hasattr(self.disk_cache, "stats")
                    else {},
                }
            )

        return info

    async def optimize_cache(self) -> Dict[str, int]:
        """
        Optimise le cache en supprimant les entrées peu utiles

        Returns:
            Statistiques d'optimisation
        """
        if not self.disk_cache:
            return {"deleted": 0, "optimized": 0}

        deleted_count = 0
        optimized_count = 0

        # Nettoyage des entrées expirées
        deleted_count += await self._cleanup_expired_entries()

        # Optimisation de la fragmentation (si supporté)
        try:
            if hasattr(self.disk_cache, "cull"):
                culled = self.disk_cache.cull()
                optimized_count += culled
        except:
            pass

        self.logger.info(
            f"Optimisation cache: {deleted_count} supprimées, "
            f"{optimized_count} optimisées"
        )

        return {"deleted": deleted_count, "optimized": optimized_count}

    def estimate_cache_value(
        self, section_name: str, word_count: int, generation_time: float
    ) -> float:
        """
        Estime la valeur d'une entrée de cache

        Args:
            section_name: Nom de la section
            word_count: Nombre de mots générés
            generation_time: Temps de génération

        Returns:
            Score de valeur (plus élevé = plus précieux)
        """
        # Facteurs de valeur
        time_factor = min(generation_time / 60, 10)  # Temps normalisé (max 10 min)
        size_factor = min(word_count / 100, 5)  # Taille normalisée (max 500 mots)

        # Bonus pour certaines sections importantes
        section_bonus = 1.0
        important_sections = [
            "Évaluation psychomotrice",
            "Analyse & synthèse",
            "Conclusion & recommandations",
        ]

        if any(important in section_name for important in important_sections):
            section_bonus = 1.5

        # Score final
        value_score = (time_factor + size_factor) * section_bonus

        return value_score

    async def get_cache_recommendations(self) -> List[str]:
        """
        Génère des recommandations pour optimiser l'utilisation du cache

        Returns:
            Liste de recommandations
        """
        recommendations = []

        if not self.config.enable_cache:
            recommendations.append("Activez le cache pour améliorer les performances")
            return recommendations

        hit_rate = self.stats.hit_rate

        if hit_rate < 0.2:
            recommendations.append(
                "Taux de succès du cache faible (<20%). "
                "Vérifiez que vos notes ne changent pas entre les générations."
            )
        elif hit_rate > 0.8:
            recommendations.append(
                "Excellent taux de succès du cache (>80%). "
                "Vous pouvez augmenter la taille du cache si besoin."
            )

        if self.stats.size_mb > self.config.cache_size_mb * 0.9:
            recommendations.append(
                "Cache presque plein. Considérez augmenter la taille limite "
                "ou activer le nettoyage automatique."
            )

        if self.config.cache_ttl_hours > 168:  # Plus d'une semaine
            recommendations.append(
                "TTL du cache très long. Les anciens résultats peuvent être obsolètes."
            )

        # Analyse des patterns d'utilisation
        total_operations = self.stats.hits + self.stats.misses + self.stats.sets
        if total_operations > 100:
            if self.stats.sets / total_operations > 0.7:
                recommendations.append(
                    "Beaucoup de nouvelles entrées créées. "
                    "Le cache sera plus efficace lors des prochaines utilisations."
                )

        return recommendations

    def __str__(self) -> str:
        """Représentation string du cache"""
        if not self.config.enable_cache:
            return "Cache désactivé"

        return (
            f"Cache: {self.stats.hits + self.stats.misses} accès, "
            f"taux succès: {self.stats.hit_rate:.1%}, "
            f"taille: {self.stats.size_mb:.1f}MB"
        )

    async def __aenter__(self):
        """Support du context manager asynchrone"""
        await self.load_cache()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Nettoyage automatique du context manager"""
        await self.save_cache()


# Fonction utilitaire pour créer un cache manager
def create_cache_manager(cache_dir: Path, config: CacheConfig) -> CacheManager:
    """
    Crée un gestionnaire de cache avec la configuration donnée

    Args:
        cache_dir: Répertoire de cache
        config: Configuration du cache

    Returns:
        Instance du gestionnaire de cache
    """
    return CacheManager(cache_dir, config)


# Décorateur pour mise en cache automatique des fonctions
def cached_generation(cache_manager: CacheManager, ttl_hours: int = None):
    """
    Décorateur pour mise en cache automatique des fonctions de génération

    Args:
        cache_manager: Gestionnaire de cache
        ttl_hours: Durée de vie personnalisée

    Returns:
        Décorateur
    """

    def decorator(func):
        async def wrapper(*args, **kwargs):
            # Génération de la clé de cache basée sur les arguments
            cache_key = CacheKey.generate(
                model_name=kwargs.get("model_name", "unknown"),
                section_title=kwargs.get("section_title", "unknown"),
                section_notes=kwargs.get("section_notes", ""),
                generation_params=kwargs.get("generation_params", {}),
            )

            # Tentative de récupération depuis le cache
            cached_result = await cache_manager.get(cache_key)
            if cached_result:
                return cached_result.get("result")

            # Exécution de la fonction si pas en cache
            result = await func(*args, **kwargs)

            # Mise en cache du résultat
            await cache_manager.set(
                cache_key,
                {"result": result, "function": func.__name__},
                expire_hours=ttl_hours,
            )

            return result

        return wrapper

    return decorator
