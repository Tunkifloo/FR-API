from pydantic import BaseModel, EmailStr, validator, Field
from typing import Optional, List, Dict, Any
from datetime import datetime


class PersonCreate(BaseModel):
    """Esquema para crear nueva persona"""
    nombre: str = Field(..., min_length=1, max_length=100, description="Nombre de la persona")
    apellidos: str = Field(..., min_length=1, max_length=100, description="Apellidos de la persona")
    correo: EmailStr = Field(..., description="Correo electrónico único")
    id_estudiante: Optional[str] = Field(None, min_length=6, max_length=20, description="ID de estudiante opcional")

    @validator('nombre', 'apellidos')
    def validate_names(cls, v):
        if not v or not v.strip():
            raise ValueError('Nombre y apellidos no pueden estar vacíos')
        return v.strip()

    @validator('id_estudiante')
    def validate_student_id(cls, v):
        if v is not None:
            v = v.strip()
            if not v.isdigit():
                raise ValueError('ID de estudiante debe contener solo dígitos')
            if len(v) < 6 or len(v) > 20:
                raise ValueError('ID de estudiante debe tener entre 6-20 dígitos')
        return v

    class Config:
        schema_extra = {
            "example": {
                "nombre": "Juan",
                "apellidos": "Pérez García",
                "correo": "juan.perez@email.com",
                "id_estudiante": "123456789"
            }
        }


class PersonUpdate(BaseModel):
    """Esquema para actualizar persona existente"""
    nombre: Optional[str] = Field(None, min_length=1, max_length=100)
    apellidos: Optional[str] = Field(None, min_length=1, max_length=100)
    correo: Optional[EmailStr] = None
    id_estudiante: Optional[str] = Field(None, min_length=6, max_length=20)
    activo: Optional[bool] = None

    @validator('nombre', 'apellidos')
    def validate_names(cls, v):
        if v is not None and (not v or not v.strip()):
            raise ValueError('Nombre y apellidos no pueden estar vacíos')
        return v.strip() if v else v

    @validator('id_estudiante')
    def validate_student_id(cls, v):
        if v is not None:
            v = v.strip()
            if not v.isdigit() or len(v) < 6 or len(v) > 20:
                raise ValueError('ID de estudiante debe tener entre 6-20 dígitos')
        return v


class PersonResponse(BaseModel):
    """Respuesta básica de persona (sin datos sensibles)"""
    id: int = Field(..., description="ID único de la persona")
    nombre: str = Field(..., description="Nombre de la persona")
    apellidos: str = Field(..., description="Apellidos de la persona")
    correo: str = Field(..., description="Correo electrónico")
    id_estudiante: Optional[str] = Field(None, description="ID de estudiante")
    pk: str = Field(..., description="Clave primaria única")
    fecha_registro: datetime = Field(..., description="Fecha de registro")
    activo: bool = Field(..., description="Estado activo/inactivo")

    class Config:
        from_attributes = True
        schema_extra = {
            "example": {
                "id": 1,
                "nombre": "Juan",
                "apellidos": "Pérez García",
                "correo": "juan.perez@email.com",
                "id_estudiante": "123456789",
                "pk": "PK_20250101120000_1234",
                "fecha_registro": "2025-01-01T12:00:00",
                "activo": True
            }
        }


class PersonWithFeatures(PersonResponse):
    """Persona con información de características faciales"""
    caracteristicas: Optional[List[float]] = Field(None, description="Características faciales extraídas")
    umbral: float = Field(..., description="Umbral de similitud personalizado")
    metodo: str = Field(..., description="Método de extracción usado")
    version: Optional[str] = Field(None, description="Versión del algoritmo")
    fecha_extraccion: Optional[datetime] = Field(None, description="Fecha de extracción de características")


class PersonWithSystemInfo(PersonResponse):
    """Persona con información del sistema de procesamiento"""
    system_info: Dict[str, Any] = Field(..., description="Información del sistema de procesamiento")

    class Config:
        schema_extra = {
            "example": {
                "id": 1,
                "nombre": "Juan",
                "apellidos": "Pérez García",
                "correo": "juan.perez@email.com",
                "id_estudiante": "123456789",
                "pk": "PK_20250101120000_1234",
                "fecha_registro": "2025-01-01T12:00:00",
                "activo": True,
                "system_info": {
                    "processing_method": "enhanced",
                    "algorithm_version": "2.0",
                    "extraction_date": "2025-01-01T12:00:00",
                    "features_count": 512
                }
            }
        }


class PersonRegistrationResponse(BaseModel):
    """Respuesta del registro de persona"""
    message: str = Field(..., description="Mensaje de confirmación")
    person_id: int = Field(..., description="ID de la persona creada")
    pk: str = Field(..., description="Clave primaria única")
    features_count: int = Field(..., description="Número de características extraídas")
    faces_detected: int = Field(..., description="Número de rostros detectados")
    processing_method: str = Field(..., description="Método de procesamiento usado")
    processing_time: float = Field(..., description="Tiempo de procesamiento en segundos")
    system_info: Dict[str, Any] = Field(..., description="Información del sistema")

    class Config:
        schema_extra = {
            "example": {
                "message": "Persona registrada exitosamente",
                "person_id": 1,
                "pk": "PK_20250101120000_1234",
                "features_count": 512,
                "faces_detected": 1,
                "processing_method": "enhanced",
                "processing_time": 3.45,
                "system_info": {
                    "enhanced_processing": True,
                    "feature_method": "enhanced",
                    "threshold": 0.70
                }
            }
        }


class PersonUpdateFeaturesResponse(BaseModel):
    """Respuesta de actualización de características"""
    message: str = Field(..., description="Mensaje de confirmación")
    person_id: int = Field(..., description="ID de la persona actualizada")
    method: str = Field(..., description="Método usado para nueva extracción")
    features_count: int = Field(..., description="Número de características extraídas")
    faces_detected: int = Field(..., description="Número de rostros detectados")


class PersonListResponse(BaseModel):
    """Respuesta de listado de personas"""
    total: int = Field(..., description="Total de personas")
    persons: List[PersonResponse] = Field(..., description="Lista de personas")
    system_stats: Optional[Dict[str, Any]] = Field(None, description="Estadísticas del sistema")
    pagination: Optional[Dict[str, Any]] = Field(None, description="Información de paginación")

    class Config:
        schema_extra = {
            "example": {
                "total": 2,
                "persons": [
                    {
                        "id": 1,
                        "nombre": "Juan",
                        "apellidos": "Pérez",
                        "correo": "juan@email.com",
                        "id_estudiante": "123456789",
                        "pk": "PK_20250101120000_1234",
                        "fecha_registro": "2025-01-01T12:00:00",
                        "activo": True
                    }
                ],
                "system_stats": {
                    "total_personas": 2,
                    "por_metodo": {
                        "enhanced": 1,
                        "traditional": 1
                    }
                },
                "pagination": {
                    "page": 1,
                    "per_page": 10,
                    "total_pages": 1
                }
            }
        }


class PersonSearchResponse(PersonWithSystemInfo):
    """Respuesta de búsqueda de persona específica"""
    pass


class PersonStats(BaseModel):
    """Estadísticas de personas en el sistema"""
    total_persons: int = Field(..., description="Total de personas registradas")
    active_persons: int = Field(..., description="Personas activas")
    inactive_persons: int = Field(..., description="Personas inactivas")
    by_method: Dict[str, int] = Field(..., description="Distribución por método de procesamiento")
    by_version: Dict[str, int] = Field(..., description="Distribución por versión de algoritmo")
    registration_trend: Optional[Dict[str, int]] = Field(None, description="Tendencia de registros")


class ProcessingStats(BaseModel):
    """Estadísticas de procesamiento"""
    database_stats: PersonStats = Field(..., description="Estadísticas de base de datos")
    current_config: Dict[str, Any] = Field(..., description="Configuración actual")
    system_capabilities: Dict[str, Any] = Field(..., description="Capacidades del sistema")

    class Config:
        schema_extra = {
            "example": {
                "database_stats": {
                    "total_persons": 100,
                    "active_persons": 95,
                    "inactive_persons": 5,
                    "by_method": {
                        "enhanced": 60,
                        "traditional": 40
                    },
                    "by_version": {
                        "2.0": 60,
                        "1.0": 40
                    }
                },
                "current_config": {
                    "enhanced_processing": True,
                    "feature_method": "enhanced",
                    "default_threshold": 0.70,
                    "adaptive_threshold": True,
                    "multiple_detectors": True
                },
                "system_capabilities": {
                    "dlib_available": False,
                    "sklearn_available": True,
                    "migration_enabled": True
                }
            }
        }


class BatchRegistrationRequest(BaseModel):
    """Solicitud de registro en lote"""
    persons_data: List[Dict[str, Any]] = Field(..., description="Lista de datos de personas")
    processing_method: Optional[str] = Field("enhanced", description="Método de procesamiento")
    global_threshold: Optional[float] = Field(None, description="Umbral global")

    class Config:
        schema_extra = {
            "example": {
                "persons_data": [
                    {
                        "nombre": "Juan",
                        "apellidos": "Pérez",
                        "correo": "juan@email.com",
                        "id_estudiante": "123456789"
                    }
                ],
                "processing_method": "enhanced",
                "global_threshold": 0.70
            }
        }


class BatchRegistrationResponse(BaseModel):
    """Respuesta de registro en lote"""
    total: int = Field(..., description="Total de personas procesadas")
    successful: int = Field(..., description="Registros exitosos")
    failed: int = Field(..., description="Registros fallidos")
    errors: List[str] = Field(..., description="Lista de errores")
    processing_time: float = Field(..., description="Tiempo total de procesamiento")

    class Config:
        schema_extra = {
            "example": {
                "total": 10,
                "successful": 8,
                "failed": 2,
                "errors": [
                    "Email duplicado: juan@email.com",
                    "No se detectó rostro en imagen de María"
                ],
                "processing_time": 45.67
            }
        }