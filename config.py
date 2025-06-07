import os
from pydantic_settings import BaseSettings
from typing import Dict, Any, Tuple, List


def parse_tuple(value: str, default: Tuple[int, int] = (128, 128)) -> Tuple[int, int]:
    """Parsear tupla desde string del .env"""
    try:
        if isinstance(value, str):
            # Remover paréntesis y espacios
            clean_value = value.strip("()").replace(" ", "")
            # Dividir por coma y convertir a enteros
            parts = clean_value.split(",")
            if len(parts) == 2:
                return (int(parts[0]), int(parts[1]))
        return default
    except:
        return default


def parse_extensions(value: str) -> set:
    """Parsear extensiones desde string del .env"""
    try:
        if isinstance(value, str):
            # Dividir por comas y limpiar espacios
            extensions = [ext.strip().lower() for ext in value.split(',')]
            return set(extensions)
        return {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
    except:
        return {'png', 'jpg', 'jpeg', 'gif', 'bmp'}


class Settings(BaseSettings):
    # ===== CONFIGURACIÓN MYSQL =====
    MYSQL_HOST: str = "localhost"
    MYSQL_PORT: int = 3306
    MYSQL_USER: str = "root"
    MYSQL_PASSWORD: str = "@dmin"
    MYSQL_DATABASE: str = "reconocimiento_facial"
    MYSQL_CHARSET: str = "utf8mb4"
    MYSQL_COLLATION: str = "utf8mb4_unicode_ci"

    # ===== CONFIGURACIÓN DE LA APLICACIÓN =====
    DEBUG: bool = True
    SECRET_KEY: str = "your-secret-key-here"
    LOG_LEVEL: str = "INFO"

    # ===== CONFIGURACIÓN DE ARCHIVOS =====
    UPLOAD_DIR: str = "uploads"
    MODELS_DIR: str = "models"
    BACKUP_DIR: str = "exports"
    JSON_BACKUP_DIR: str = "json_backup"

    # Configuración de archivos subidos (como strings en .env)
    MAX_FILE_SIZE: int = 10485760  # 10MB
    ALLOWED_EXTENSIONS_STR: str = "jpg,jpeg,png,bmp,gif"  # Como string en .env

    # ===== CONFIGURACIÓN DE RECONOCIMIENTO FACIAL MEJORADO =====
    # Sistema mejorado
    USE_ENHANCED_PROCESSING: bool = True
    FEATURE_METHOD: str = "enhanced"  # "enhanced" o "traditional"

    # Umbrales adaptativos
    DEFAULT_SIMILARITY_THRESHOLD: float = 0.70
    MIN_THRESHOLD: float = 0.50
    MAX_THRESHOLD: float = 0.90
    ADAPTIVE_THRESHOLD: bool = True

    # Configuración de detección de rostros
    USE_MULTIPLE_DETECTORS: bool = True
    USE_DLIB: bool = False  # Deshabilitado por defecto

    # Configuración de extracción de características
    NORMALIZE_FEATURES: bool = True
    USE_MULTI_SCALE_LBP: bool = True
    INCLUDE_SYMMETRY_FEATURES: bool = True
    TARGET_FEATURE_SIZE: int = 512

    # Tamaños de imagen (como strings en .env, se parsean a tuplas)
    FACE_SIZE_STR: str = "(128,128)"
    MIN_IMAGE_SIZE_STR: str = "(100,100)"

    # Configuración de comparación
    USE_CONSISTENCY_CHECK: bool = True
    MIN_CONSISTENCY: float = 0.7

    # Pesos para comparación multi-métrica
    COMPARISON_WEIGHT_COSINE: float = 0.35
    COMPARISON_WEIGHT_CORRELATION: float = 0.20
    COMPARISON_WEIGHT_EUCLIDEAN: float = 0.20
    COMPARISON_WEIGHT_MANHATTAN: float = 0.15
    COMPARISON_WEIGHT_SEGMENTS: float = 0.10

    # Configuración de migración
    MIGRATION_ENABLED: bool = False
    FORCE_MIGRATION: bool = False
    BACKUP_BEFORE_MIGRATION: bool = True

    # Configuración de estudiantes
    MIN_STUDENT_ID_LENGTH: int = 6
    MAX_STUDENT_ID_LENGTH: int = 20

    # ===== VARIABLES ADICIONALES DEL .env =====
    # Detección DNN
    USE_DNN_DETECTION: bool = True
    DNN_CONFIDENCE_THRESHOLD: float = 0.5

    # Embeddings
    USE_FACE_EMBEDDINGS: bool = True
    EMBEDDING_WEIGHT: float = 0.30

    # Características avanzadas
    USE_GABOR_FEATURES: bool = True
    USE_ORB_FEATURES: bool = True
    USE_EDGE_FEATURES: bool = True
    USE_COLOR_FEATURES: bool = True

    # Sistema de votación
    USE_VOTING_SYSTEM: bool = True
    MIN_VOTES_REQUIRED: int = 3

    class Config:
        case_sensitive = False
        # Permitir campos extra para evitar errores
        extra = "ignore"

    @property
    def ALLOWED_EXTENSIONS(self) -> set:
        """Obtener ALLOWED_EXTENSIONS como set"""
        return parse_extensions(self.ALLOWED_EXTENSIONS_STR)

    @property
    def FACE_SIZE(self) -> Tuple[int, int]:
        """Obtener FACE_SIZE como tupla"""
        return parse_tuple(self.FACE_SIZE_STR, (128, 128))

    @property
    def MIN_IMAGE_SIZE(self) -> Tuple[int, int]:
        """Obtener MIN_IMAGE_SIZE como tupla"""
        return parse_tuple(self.MIN_IMAGE_SIZE_STR, (100, 100))

    @property
    def COMPARISON_WEIGHTS(self) -> Dict[str, float]:
        """Obtener pesos de comparación como diccionario"""
        return {
            'cosine': self.COMPARISON_WEIGHT_COSINE,
            'correlation': self.COMPARISON_WEIGHT_CORRELATION,
            'euclidean': self.COMPARISON_WEIGHT_EUCLIDEAN,
            'manhattan': self.COMPARISON_WEIGHT_MANHATTAN,
            'segments': self.COMPARISON_WEIGHT_SEGMENTS
        }

    def get_threshold_for_person(self, person_data: Dict[str, Any] = None) -> float:
        """Obtener umbral específico para una persona"""
        if person_data and person_data.get('umbral'):
            # Validar que el umbral esté en rango válido
            person_threshold = float(person_data['umbral'])
            return max(self.MIN_THRESHOLD, min(self.MAX_THRESHOLD, person_threshold))

        return self.DEFAULT_SIMILARITY_THRESHOLD

    def get_comparison_config(self) -> Dict[str, Any]:
        """Obtener configuración completa para comparación"""
        return {
            "weights": self.COMPARISON_WEIGHTS,
            "use_consistency": self.USE_CONSISTENCY_CHECK,
            "min_consistency": self.MIN_CONSISTENCY,
            "adaptive_threshold": self.ADAPTIVE_THRESHOLD
        }

    def get_detection_config(self) -> Dict[str, Any]:
        """Obtener configuración para detección de rostros"""
        return {
            "use_multiple_detectors": self.USE_MULTIPLE_DETECTORS,
            "use_dlib": self.USE_DLIB,
            "face_size": self.FACE_SIZE,
            "min_image_size": self.MIN_IMAGE_SIZE,
            "use_dnn_detection": self.USE_DNN_DETECTION,
            "dnn_confidence_threshold": self.DNN_CONFIDENCE_THRESHOLD
        }

    def get_feature_config(self) -> Dict[str, Any]:
        """Obtener configuración para extracción de características"""
        return {
            "method": self.FEATURE_METHOD,
            "normalize": self.NORMALIZE_FEATURES,
            "use_multi_scale_lbp": self.USE_MULTI_SCALE_LBP,
            "include_symmetry": self.INCLUDE_SYMMETRY_FEATURES,
            "target_size": self.TARGET_FEATURE_SIZE,
            "face_size": self.FACE_SIZE,
            "use_face_embeddings": self.USE_FACE_EMBEDDINGS,
            "embedding_weight": self.EMBEDDING_WEIGHT,
            "use_gabor_features": self.USE_GABOR_FEATURES,
            "use_orb_features": self.USE_ORB_FEATURES,
            "use_edge_features": self.USE_EDGE_FEATURES,
            "use_color_features": self.USE_COLOR_FEATURES
        }

    def get_file_config(self) -> Dict[str, Any]:
        """Obtener configuración de archivos"""
        return {
            "max_file_size": self.MAX_FILE_SIZE,
            "allowed_extensions": list(self.ALLOWED_EXTENSIONS),
            "upload_dir": self.UPLOAD_DIR,
            "backup_dir": self.BACKUP_DIR
        }

    def is_enhanced_mode(self) -> bool:
        """Verificar si el modo mejorado está activo"""
        return self.USE_ENHANCED_PROCESSING and self.FEATURE_METHOD == "enhanced"

    def validate_settings(self) -> Dict[str, Any]:
        """Validar configuraciones y retornar reporte"""
        issues = []
        warnings = []

        # Validar umbrales
        if not (0.0 <= self.DEFAULT_SIMILARITY_THRESHOLD <= 1.0):
            issues.append("DEFAULT_SIMILARITY_THRESHOLD debe estar entre 0.0 y 1.0")

        if self.MIN_THRESHOLD >= self.MAX_THRESHOLD:
            issues.append("MIN_THRESHOLD debe ser menor que MAX_THRESHOLD")

        # Validar pesos de comparación
        total_weight = sum(self.COMPARISON_WEIGHTS.values())
        if abs(total_weight - 1.0) > 0.01:  # Tolerancia de 1%
            warnings.append(f"Los pesos de comparación suman {total_weight:.3f}, deberían sumar 1.0")

        # Validar tamaños
        if self.TARGET_FEATURE_SIZE < 100:
            warnings.append("TARGET_FEATURE_SIZE muy pequeño, podría afectar precisión")

        # Validar directorios
        directories = [self.UPLOAD_DIR, self.MODELS_DIR, self.BACKUP_DIR, self.JSON_BACKUP_DIR]
        for directory in directories:
            if not os.path.exists(directory):
                warnings.append(f"Directorio no existe: {directory}")

        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "warnings": warnings,
            "enhanced_mode": self.is_enhanced_mode(),
            "total_weight": total_weight
        }


# Instancia global de configuración
settings = Settings()

# Validar configuración al importar (solo en desarrollo)
if settings.DEBUG:
    try:
        validation = settings.validate_settings()
        if not validation["valid"]:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Problemas en configuración: {validation['issues']}")
        if validation["warnings"]:
            import logging
            logger = logging.getLogger(__name__)
            logger.info(f"Advertencias de configuración: {validation['warnings']}")
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(f"Error validando configuración: {e}")