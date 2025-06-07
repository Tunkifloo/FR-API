from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from fastapi.responses import JSONResponse
from app.database.models import PersonModel
from app.database.connection import get_db_connection
from app.core.utils import get_system_stats, cleanup_temp_files, verify_system_requirements
from config import settings
import logging

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/stats")
async def get_system_stats_endpoint():
    """Obtener estadísticas completas del sistema"""
    try:
        connection = await get_db_connection()
        if not connection:
            raise HTTPException(status_code=500, detail="Error de conexión a base de datos")

        cursor = connection.cursor()

        # Estadísticas básicas
        cursor.execute("SELECT COUNT(*) FROM PERSONAS WHERE activo = TRUE")
        total_persons = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM CARACTERISTICAS_FACIALES")
        total_models = cursor.fetchone()[0]

        cursor.execute("SELECT MIN(fecha_registro), MAX(fecha_registro) FROM PERSONAS WHERE activo = TRUE")
        dates = cursor.fetchone()

        cursor.execute("SELECT VERSION()")
        mysql_version = cursor.fetchone()[0]

        # Estadísticas por método (con manejo de errores para esquema nuevo/antiguo)
        method_stats = {}
        try:
            cursor.execute("""
                           SELECT COALESCE(metodo, 'opencv_tradicional') as metodo,
                                  COALESCE(version_algoritmo, '1.0')     as version_algoritmo,
                                  COUNT(*)                               as cantidad,
                                  AVG(umbral_similitud)                  as umbral_promedio
                           FROM CARACTERISTICAS_FACIALES cf
                                    JOIN PERSONAS p ON cf.persona_id = p.ID
                           WHERE p.activo = TRUE
                           GROUP BY COALESCE(metodo, 'opencv_tradicional'), COALESCE(version_algoritmo, '1.0')
                           """)
            method_results = cursor.fetchall()
            for row in method_results:
                method_key = f"{row[0]}_v{row[1]}"
                method_stats[method_key] = {
                    "cantidad": row[2],
                    "umbral_promedio": float(row[3]) if row[3] else 0.70
                }
        except Exception as e:
            logger.warning(f"No se pudieron obtener estadísticas por método: {e}")
            # Fallback para esquema básico
            method_stats = {"opencv_tradicional_v1.0": {"cantidad": total_models, "umbral_promedio": 0.70}}

        cursor.close()
        connection.close()

        # Estadísticas del sistema de archivos
        fs_stats = get_system_stats()

        return {
            "system_info": {
                "version": "2.1.0",
                "status": "active",
                "database": "MySQL",
                "enhanced_processing": settings.USE_ENHANCED_PROCESSING,
                "feature_method": settings.FEATURE_METHOD,
                "default_threshold": settings.DEFAULT_SIMILARITY_THRESHOLD
            },
            "database_statistics": {
                "total_persons": total_persons,
                "total_models": total_models,
                "mysql_version": mysql_version,
                "first_register": dates[0] if dates[0] else None,
                "last_register": dates[1] if dates[1] else None,
                "methods": method_stats
            },
            "file_system": fs_stats,
            "configuration": {
                "adaptive_threshold": settings.ADAPTIVE_THRESHOLD,
                "multiple_detectors": settings.USE_MULTIPLE_DETECTORS,
                "use_dlib": settings.USE_DLIB
            }
        }

    except Exception as e:
        logger.error(f"Error obteniendo estadísticas: {e}")
        raise HTTPException(status_code=500, detail="Error interno del servidor")


@router.get("/health")
async def admin_health_check():
    """Verificación de salud del sistema para administradores"""
    try:
        # Verificar base de datos
        connection = await get_db_connection()
        if connection:
            connection.close()
            db_status = "connected"
        else:
            db_status = "disconnected"

        # Verificar procesador facial
        facial_status = "ready"
        try:
            from app.core.facial_processing import FacialProcessor
            processor = FacialProcessor()
            facial_status = "ready"
        except Exception as e:
            facial_status = f"error: {str(e)}"

        # Verificar dependencias
        requirements = verify_system_requirements()

        # Determinar estado general
        critical_components = ["opencv", "numpy", "mysql_connector", "config", "directories"]
        critical_ok = all(requirements.get(comp, False) for comp in critical_components)

        overall_status = "healthy" if (db_status == "connected" and critical_ok) else "degraded"

        return {
            "status": overall_status,
            "components": {
                "database": {"status": db_status, "connection": db_status == "connected"},
                "facial_recognition": facial_status,
                "file_system": "ready",
                "dependencies": requirements
            },
            "system_config": {
                "enhanced_processing": settings.USE_ENHANCED_PROCESSING,
                "feature_method": settings.FEATURE_METHOD,
                "adaptive_threshold": settings.ADAPTIVE_THRESHOLD
            },
            "uptime": "active",
            "last_check": "2025-01-01T00:00:00Z"
        }

    except Exception as e:
        logger.error(f"Error en health check admin: {e}")
        raise HTTPException(status_code=500, detail="Error verificando estado del sistema")


@router.get("/integrity")
async def check_system_integrity():
    """Verificar integridad del sistema"""
    try:
        # Verificar base de datos
        connection = await get_db_connection()
        db_status = "connected" if connection else "disconnected"
        if connection:
            connection.close()

        # Verificar directorios
        import os
        directories = [
            settings.UPLOAD_DIR,
            settings.MODELS_DIR,
            settings.BACKUP_DIR,
            settings.JSON_BACKUP_DIR,
            'logs'
        ]
        dir_status = {}

        for directory in directories:
            if os.path.exists(directory):
                try:
                    files = os.listdir(directory)
                    file_count = len(files)
                    # Calcular tamaño total
                    total_size = sum(
                        os.path.getsize(os.path.join(directory, f))
                        for f in files
                        if os.path.isfile(os.path.join(directory, f))
                    )
                    dir_status[directory] = {
                        "status": "ok",
                        "files": file_count,
                        "size_mb": round(total_size / (1024 * 1024), 2)
                    }
                except Exception as e:
                    dir_status[directory] = {"status": "error", "message": str(e)}
            else:
                dir_status[directory] = {"status": "missing", "files": 0}

        # Verificar consistencia de datos
        data_integrity = await _check_data_integrity()

        # Estado general
        overall_status = "healthy"
        if db_status != "connected":
            overall_status = "critical"
        elif any(d["status"] != "ok" for d in dir_status.values()):
            overall_status = "degraded"

        return {
            "integrity_check": {
                "overall_status": overall_status,
                "database": {"status": db_status, "connection": db_status == "connected"},
                "directories": dir_status,
                "data_integrity": data_integrity
            },
            "recommendations": _get_system_recommendations(overall_status, dir_status)
        }

    except Exception as e:
        logger.error(f"Error verificando integridad: {e}")
        raise HTTPException(status_code=500, detail="Error verificando integridad del sistema")


@router.post("/cleanup")
async def cleanup_system(background_tasks: BackgroundTasks, max_age_hours: int = 24):
    """Limpiar archivos temporales y datos antiguos"""
    try:
        def cleanup_task():
            # Limpiar archivos temporales
            cleanup_temp_files(max_age_hours)
            logger.info(f"Limpieza completada para archivos de más de {max_age_hours} horas")

        background_tasks.add_task(cleanup_task)

        return {
            "message": "Limpieza iniciada en segundo plano",
            "max_age_hours": max_age_hours,
            "status": "started"
        }

    except Exception as e:
        logger.error(f"Error iniciando limpieza: {e}")
        raise HTTPException(status_code=500, detail="Error iniciando limpieza del sistema")


@router.get("/config")
async def get_system_config():
    """Obtener configuración actual del sistema"""
    try:
        return {
            "facial_recognition": {
                "enhanced_processing": settings.USE_ENHANCED_PROCESSING,
                "feature_method": settings.FEATURE_METHOD,
                "default_threshold": settings.DEFAULT_SIMILARITY_THRESHOLD,
                "adaptive_threshold": settings.ADAPTIVE_THRESHOLD,
                "multiple_detectors": settings.USE_MULTIPLE_DETECTORS,
                "use_dlib": settings.USE_DLIB
            },
            "thresholds": {
                "default": settings.DEFAULT_SIMILARITY_THRESHOLD,
                "min": settings.MIN_THRESHOLD,
                "max": settings.MAX_THRESHOLD
            },
            "comparison_weights": settings.COMPARISON_WEIGHTS if settings.USE_ENHANCED_PROCESSING else None,
            "system": {
                "debug": settings.DEBUG,
                "log_level": getattr(settings, 'LOG_LEVEL', 'INFO')
            },
            "directories": {
                "upload": settings.UPLOAD_DIR,
                "models": settings.MODELS_DIR,
                "backup": settings.BACKUP_DIR,
                "json_backup": settings.JSON_BACKUP_DIR
            }
        }
    except Exception as e:
        logger.error(f"Error obteniendo configuración: {e}")
        raise HTTPException(status_code=500, detail="Error obteniendo configuración del sistema")


@router.get("/performance")
async def get_performance_metrics():
    """Obtener métricas de rendimiento del sistema"""
    try:
        # Estadísticas básicas de rendimiento
        stats = await PersonModel.get_system_stats()
        requirements = verify_system_requirements()
        fs_stats = get_system_stats()

        return {
            "database_performance": {
                "total_persons": stats.get('total_personas', 0),
                "total_features": stats.get('total_features', 0)
            },
            "system_resources": {
                "dependencies_ok": all(requirements.get(dep, False) for dep in ["opencv", "numpy", "mysql_connector"]),
                "disk_usage": fs_stats.get('disk_usage', {}),
                "file_counts": fs_stats.get('directories', {})
            },
            "configuration_status": {
                "enhanced_processing": settings.USE_ENHANCED_PROCESSING,
                "adaptive_thresholds": settings.ADAPTIVE_THRESHOLD,
                "multiple_detectors": settings.USE_MULTIPLE_DETECTORS
            }
        }
    except Exception as e:
        logger.error(f"Error obteniendo métricas de rendimiento: {e}")
        raise HTTPException(status_code=500, detail="Error obteniendo métricas de rendimiento")


async def _check_data_integrity():
    """Verificar integridad de datos"""
    integrity_issues = []

    try:
        connection = await get_db_connection()
        if not connection:
            return {"status": "error", "message": "No hay conexión a base de datos"}

        cursor = connection.cursor()

        # Verificar personas sin características
        cursor.execute("""
                       SELECT COUNT(*)
                       FROM PERSONAS p
                                LEFT JOIN CARACTERISTICAS_FACIALES cf ON p.ID = cf.persona_id
                       WHERE p.activo = TRUE
                         AND cf.ID IS NULL
                       """)
        persons_without_features = cursor.fetchone()[0]
        if persons_without_features > 0:
            integrity_issues.append(f"{persons_without_features} personas sin características faciales")

        # Verificar características huérfanas
        cursor.execute("""
                       SELECT COUNT(*)
                       FROM CARACTERISTICAS_FACIALES cf
                                LEFT JOIN PERSONAS p ON cf.persona_id = p.ID
                       WHERE p.ID IS NULL
                       """)
        orphan_features = cursor.fetchone()[0]
        if orphan_features > 0:
            integrity_issues.append(f"{orphan_features} características sin persona asociada")

        # Verificar umbrales fuera de rango
        cursor.execute("""
                       SELECT COUNT(*)
                       FROM CARACTERISTICAS_FACIALES
                       WHERE umbral_similitud < 0.30
                          OR umbral_similitud > 1.00
                       """)
        invalid_thresholds = cursor.fetchone()[0]
        if invalid_thresholds > 0:
            integrity_issues.append(f"{invalid_thresholds} umbrales de similitud inválidos")

        cursor.close()
        connection.close()

        return {
            "status": "ok" if not integrity_issues else "issues_found",
            "issues": integrity_issues,
            "persons_without_features": persons_without_features,
            "orphan_features": orphan_features,
            "invalid_thresholds": invalid_thresholds
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}


def _get_system_recommendations(overall_status, dir_status):
    """Generar recomendaciones basadas en el estado del sistema"""
    recommendations = []

    missing_dirs = [d for d, info in dir_status.items() if info["status"] == "missing"]
    if missing_dirs:
        recommendations.append(f"Crear directorios faltantes: {', '.join(missing_dirs)}")

    if not settings.USE_ENHANCED_PROCESSING:
        recommendations.append("Considerar habilitar procesamiento mejorado para mayor precisión")

    large_dirs = [d for d, info in dir_status.items()
                  if info.get("size_mb", 0) > 1000]  # > 1GB
    if large_dirs:
        recommendations.append(f"Limpiar directorios grandes: {', '.join(large_dirs)}")

    if overall_status == "critical":
        recommendations.append("Verificar conexión a base de datos")

    return recommendations