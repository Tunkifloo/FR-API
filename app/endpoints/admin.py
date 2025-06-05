from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import JSONResponse
from app.database.models import PersonModel
from app.database.connection import get_db_connection
import logging

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/stats")
async def get_system_stats():
    """Obtener estadísticas del sistema"""
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

        cursor.close()
        connection.close()

        return {
            "system_info": {
                "version": "2.0.0",
                "status": "active",
                "database": "MySQL"
            },
            "statistics": {
                "total_persons": total_persons,
                "total_models": total_models,
                "mysql_version": mysql_version,
                "first_register": dates[0] if dates[0] else None,
                "last_register": dates[1] if dates[1] else None
            }
        }

    except Exception as e:
        logger.error(f"Error obteniendo estadísticas: {e}")
        raise HTTPException(status_code=500, detail="Error interno del servidor")


@router.get("/health")
async def admin_health_check():
    """Verificación de salud del sistema para administradores"""
    try:
        connection = await get_db_connection()
        if connection:
            connection.close()
            db_status = "connected"
        else:
            db_status = "disconnected"

        return {
            "status": "healthy",
            "components": {
                "database": db_status,
                "facial_recognition": "active",
                "file_system": "ready"
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
        # Verificar conexión a base de datos
        connection = await get_db_connection()
        db_status = "ok" if connection else "error"
        if connection:
            connection.close()

        # Verificar directorios
        import os
        directories = ['uploads', 'models', 'exports', 'json_backup']
        dir_status = {}

        for directory in directories:
            if os.path.exists(directory):
                file_count = len(os.listdir(directory))
                dir_status[directory] = {"status": "ok", "files": file_count}
            else:
                dir_status[directory] = {"status": "missing", "files": 0}

        return {
            "integrity_check": {
                "database": db_status,
                "directories": dir_status,
                "overall_status": "healthy" if db_status == "ok" else "degraded"
            }
        }

    except Exception as e:
        logger.error(f"Error verificando integridad: {e}")
        raise HTTPException(status_code=500, detail="Error verificando integridad del sistema")
