from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime

from app.schemas.person import PersonResponse


class RecognitionResult(BaseModel):
    """Resultado de reconocimiento facial básico (compatibilidad hacia atrás)"""
    similarity: float = Field(..., description="Similitud general calculada")
    correlation: float = Field(..., description="Correlación de Pearson")
    distance: float = Field(..., description="Distancia euclidiana")
    distance_similarity: float = Field(..., description="Similitud basada en distancia")
    is_match: bool = Field(..., description="Indica si hay coincidencia")
    threshold: float = Field(..., description="Umbral usado para la comparación")
    faces_detected: int = Field(..., description="Número de rostros detectados")
    features_compared: int = Field(..., description="Número de características comparadas")


class EnhancedRecognitionResult(RecognitionResult):
    """Resultado de reconocimiento facial mejorado con múltiples métricas"""
    cosine_similarity: Optional[float] = Field(None, description="Similitud coseno")
    euclidean_similarity: Optional[float] = Field(None, description="Similitud euclidiana normalizada")
    manhattan_similarity: Optional[float] = Field(None, description="Similitud Manhattan normalizada")
    segment_similarity: Optional[float] = Field(None, description="Similitud por segmentos")
    consistency: Optional[float] = Field(None, description="Consistencia entre métricas")
    euclidean_distance: Optional[float] = Field(None, description="Distancia euclidiana cruda")
    manhattan_distance: Optional[float] = Field(None, description="Distancia Manhattan cruda")
    adjusted_threshold: Optional[float] = Field(None, description="Umbral ajustado dinámicamente")
    confidence: Optional[float] = Field(None, description="Nivel de confianza del resultado")
    processing_method: Optional[str] = Field(None, description="Método de procesamiento usado")
    metrics_used: Optional[List[str]] = Field(None, description="Lista de métricas utilizadas")


class RecognitionResponse(BaseModel):
    """Respuesta de reconocimiento facial"""
    person: PersonResponse
    recognition_result: RecognitionResult
    system_info: Optional[Dict[str, Any]] = Field(None, description="Información del sistema")


class EnhancedRecognitionResponse(BaseModel):
    """Respuesta de reconocimiento facial mejorado"""
    person: PersonResponse
    recognition_result: EnhancedRecognitionResult
    system_info: Dict[str, Any] = Field(..., description="Información del sistema")


class IdentificationMatch(BaseModel):
    """Coincidencia individual en identificación"""
    person: Dict[str, Any] = Field(..., description="Datos de la persona")
    similarity: float = Field(..., description="Similitud obtenida")
    is_match: bool = Field(..., description="Indica si supera el umbral")
    threshold: float = Field(..., description="Umbral usado")
    confidence: Optional[float] = Field(None, description="Nivel de confianza")
    detailed_metrics: Optional[Dict[str, float]] = Field(None, description="Métricas detalladas")


class IdentificationResult(BaseModel):
    """Resultado de identificación facial"""
    best_match: Optional[IdentificationMatch] = Field(None, description="Mejor coincidencia encontrada")
    confidence: float = Field(..., description="Confianza de la mejor coincidencia")
    total_comparisons: int = Field(..., description="Total de comparaciones realizadas")
    faces_detected: int = Field(..., description="Rostros detectados en imagen")
    processing_method: Optional[str] = Field(None, description="Método de procesamiento usado")
    feature_extraction_method: Optional[str] = Field(None, description="Método de extracción usado")


class IdentificationResponse(BaseModel):
    """Respuesta completa de identificación"""
    identification_result: IdentificationResult
    all_matches: List[IdentificationMatch] = Field(..., description="Todas las coincidencias encontradas")
    system_info: Optional[Dict[str, Any]] = Field(None, description="Información del sistema")


class RecognitionStats(BaseModel):
    """Estadísticas de reconocimiento facial"""
    total_recognitions: int = Field(..., description="Total de reconocimientos realizados")
    successful_recognitions: int = Field(..., description="Reconocimientos exitosos")
    average_similarity: float = Field(..., description="Similitud promedio")
    average_processing_time: float = Field(..., description="Tiempo promedio de procesamiento")
    success_rate: float = Field(..., description="Tasa de éxito porcentual")
    methods_used: Dict[str, int] = Field(..., description="Métodos usados y frecuencia")


class ComparisonRequest(BaseModel):
    """Solicitud de comparación facial"""
    email: Optional[str] = Field(None, description="Email de la persona a comparar")
    student_id: Optional[str] = Field(None, description="ID de estudiante a comparar")
    person_id: Optional[int] = Field(None, description="ID directo de la persona")
    threshold_override: Optional[float] = Field(None, ge=0.0, le=1.0, description="Umbral personalizado")
    method_override: Optional[str] = Field(None, description="Método de procesamiento específico")

    class Config:
        schema_extra = {
            "example": {
                "email": "usuario@ejemplo.com",
                "threshold_override": 0.75,
                "method_override": "enhanced"
            }
        }


class BatchComparisonRequest(BaseModel):
    """Solicitud de comparación en lote"""
    persons: List[ComparisonRequest] = Field(..., description="Lista de personas a comparar")
    global_threshold: Optional[float] = Field(None, ge=0.0, le=1.0, description="Umbral global")
    processing_method: Optional[str] = Field(None, description="Método de procesamiento")

    class Config:
        schema_extra = {
            "example": {
                "persons": [
                    {"email": "persona1@ejemplo.com"},
                    {"student_id": "123456789"}
                ],
                "global_threshold": 0.70,
                "processing_method": "enhanced"
            }
        }


class BatchComparisonResponse(BaseModel):
    """Respuesta de comparación en lote"""
    total_comparisons: int = Field(..., description="Total de comparaciones realizadas")
    successful_comparisons: int = Field(..., description="Comparaciones exitosas")
    failed_comparisons: int = Field(..., description="Comparaciones fallidas")
    results: List[RecognitionResponse] = Field(..., description="Resultados individuales")
    processing_time: float = Field(..., description="Tiempo total de procesamiento")
    average_similarity: float = Field(..., description="Similitud promedio")


class RecognitionError(BaseModel):
    """Error en reconocimiento facial"""
    error_code: str = Field(..., description="Código del error")
    error_message: str = Field(..., description="Mensaje del error")
    error_details: Optional[Dict[str, Any]] = Field(None, description="Detalles adicionales del error")
    timestamp: datetime = Field(default_factory=datetime.now, description="Momento del error")
    processing_info: Optional[Dict[str, Any]] = Field(None, description="Información de procesamiento")


class SystemPerformance(BaseModel):
    """Rendimiento del sistema de reconocimiento"""
    avg_recognition_time: float = Field(..., description="Tiempo promedio de reconocimiento (segundos)")
    avg_feature_extraction_time: float = Field(..., description="Tiempo promedio de extracción (segundos)")
    avg_comparison_time: float = Field(..., description="Tiempo promedio de comparación (segundos)")
    current_load: float = Field(..., description="Carga actual del sistema (0-1)")
    peak_load: float = Field(..., description="Carga pico registrada")
    total_processed_today: int = Field(..., description="Total procesado hoy")
    error_rate: float = Field(..., description="Tasa de error porcentual")


class RecognitionConfiguration(BaseModel):
    """Configuración del sistema de reconocimiento"""
    enhanced_processing: bool = Field(..., description="Procesamiento mejorado habilitado")
    feature_method: str = Field(..., description="Método de extracción de características")
    default_threshold: float = Field(..., description="Umbral por defecto")
    adaptive_threshold: bool = Field(..., description="Umbrales adaptativos habilitados")
    multiple_detectors: bool = Field(..., description="Detectores múltiples habilitados")
    use_dlib: bool = Field(..., description="dlib habilitado")
    comparison_weights: Optional[Dict[str, float]] = Field(None, description="Pesos de comparación")
    migration_enabled: bool = Field(..., description="Migración habilitada")


# Esquemas de respuesta unificados para mantener compatibilidad
UnifiedRecognitionResult = RecognitionResult | EnhancedRecognitionResult
UnifiedRecognitionResponse = RecognitionResponse | EnhancedRecognitionResponse