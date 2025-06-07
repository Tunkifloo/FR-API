from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from app.database.connection import get_db_connection, init_database
from app.endpoints import admin, data_management, persons, recognition
from app.core.utils import configure_logging, create_directories
from config import settings
import os
import logging

# Configurar logging
configure_logging()
logger = logging.getLogger(__name__)

# Crear aplicación FastAPI
app = FastAPI(
    title="Sistema de Reconocimiento Facial Mejorado",
    description="API completa para reconocimiento facial con OpenCV, características avanzadas y comparación multi-métrica",
    version="2.1.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción, especificar dominios exactos
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Incluir rutas
app.include_router(admin.router, prefix="/api/admin", tags=["Administración"])
app.include_router(persons.router, prefix="/api/persons", tags=["Personas"])
app.include_router(recognition.router, prefix="/api/recognition", tags=["Reconocimiento"])
app.include_router(data_management.router, prefix="/api/data", tags=["Gestión de Datos"])


@app.on_event("startup")
async def startup_event():
    """Inicializar sistema al arrancar"""
    logger.info("🚀 Iniciando Sistema de Reconocimiento Facial Mejorado v2.1.0")

    # Crear directorios necesarios
    create_directories()
    logger.info("📁 Directorios de trabajo creados")

    # Inicializar base de datos
    if await init_database():
        logger.info("✅ Base de datos MySQL inicializada correctamente")
    else:
        logger.error("❌ Error inicializando base de datos MySQL")

    # Mostrar configuración actual
    logger.info(f"⚙️ Configuración del sistema:")
    logger.info(f"   - Procesamiento mejorado: {settings.USE_ENHANCED_PROCESSING}")
    logger.info(f"   - Método de características: {settings.FEATURE_METHOD}")
    logger.info(f"   - Umbral por defecto: {settings.DEFAULT_SIMILARITY_THRESHOLD}")
    logger.info(f"   - Detección múltiple: {settings.USE_MULTIPLE_DETECTORS}")
    logger.info(f"   - Umbrales adaptativos: {settings.ADAPTIVE_THRESHOLD}")

    # Verificar dependencias
    try:
        import cv2
        logger.info("✅ OpenCV disponible")
    except ImportError:
        logger.error("❌ OpenCV no disponible")

    try:
        import numpy as np
        logger.info("✅ NumPy disponible")
    except ImportError:
        logger.error("❌ NumPy no disponible")

    try:
        from sklearn.preprocessing import StandardScaler
        logger.info("✅ Scikit-learn disponible para procesamiento mejorado")
    except ImportError:
        logger.warning("⚠️ Scikit-learn no disponible - instalar con: pip install scikit-learn")
        if settings.USE_ENHANCED_PROCESSING:
            logger.warning("⚠️ Procesamiento mejorado requiere scikit-learn")

    try:
        import dlib
        logger.info("✅ dlib disponible para detección avanzada")
    except ImportError:
        logger.info("ℹ️ dlib no disponible (opcional)")

    logger.info("🎯 Sistema listo para recibir peticiones")
    logger.info("📚 Documentación disponible en: http://localhost:8000/docs")


@app.on_event("shutdown")
async def shutdown_event():
    """Limpiar recursos al cerrar"""
    logger.info("🔽 Cerrando Sistema de Reconocimiento Facial")


@app.get("/")
async def root():
    """Endpoint raíz con información del sistema"""
    return {
        "message": "Sistema de Reconocimiento Facial Mejorado - FastAPI",
        "version": "2.1.0",
        "status": "activo",
        "features": {
            "enhanced_processing": settings.USE_ENHANCED_PROCESSING,
            "adaptive_thresholds": settings.ADAPTIVE_THRESHOLD,
            "multiple_detectors": settings.USE_MULTIPLE_DETECTORS,
            "feature_method": settings.FEATURE_METHOD
        },
        "endpoints": {
            "docs": "/docs",
            "admin_panel": "/api/admin/stats",
            "health_check": "/health",
            "system_info": "/info"
        }
    }


@app.get("/health")
async def health_check():
    """Verificar estado del sistema"""
    try:
        # Verificar conexión a base de datos
        connection = await get_db_connection()
        if connection:
            connection.close()
            db_status = "connected"
        else:
            db_status = "disconnected"

        # Verificar procesador facial
        try:
            from app.core.facial_processing import FacialProcessor
            processor = FacialProcessor()
            processing_status = "ready"
        except Exception as e:
            processing_status = f"error: {str(e)}"

        # Verificar dependencias críticas
        dependencies = {}
        try:
            import cv2
            dependencies["opencv"] = "available"
        except ImportError:
            dependencies["opencv"] = "missing"

        try:
            import numpy as np
            dependencies["numpy"] = "available"
        except ImportError:
            dependencies["numpy"] = "missing"

        try:
            from sklearn.preprocessing import StandardScaler
            dependencies["sklearn"] = "available"
        except ImportError:
            dependencies["sklearn"] = "missing"

        # Determinar estado general
        critical_missing = any(status == "missing" for key, status in dependencies.items()
                               if key in ["opencv", "numpy"])

        overall_status = "healthy"
        if db_status == "disconnected":
            overall_status = "degraded"
        elif critical_missing:
            overall_status = "error"
        elif processing_status.startswith("error"):
            overall_status = "degraded"

        return {
            "status": overall_status,
            "database": db_status,
            "facial_processing": processing_status,
            "dependencies": dependencies,
            "configuration": {
                "enhanced_mode": settings.USE_ENHANCED_PROCESSING,
                "feature_method": settings.FEATURE_METHOD,
                "threshold": settings.DEFAULT_SIMILARITY_THRESHOLD
            },
            "timestamp": "2025-01-01T00:00:00Z"
        }
    except Exception as e:
        logger.error(f"Error en health check: {e}")
        raise HTTPException(status_code=500, detail="Sistema no disponible")


@app.get("/info")
async def system_info():
    """Información detallada del sistema"""
    try:
        # Información de configuración
        config_info = {
            "processing": {
                "enhanced_enabled": settings.USE_ENHANCED_PROCESSING,
                "feature_method": settings.FEATURE_METHOD,
                "adaptive_thresholds": settings.ADAPTIVE_THRESHOLD,
                "multiple_detectors": settings.USE_MULTIPLE_DETECTORS,
                "use_dlib": settings.USE_DLIB
            },
            "thresholds": {
                "default": settings.DEFAULT_SIMILARITY_THRESHOLD,
                "min": settings.MIN_THRESHOLD,
                "max": settings.MAX_THRESHOLD
            },
            "comparison_weights": settings.COMPARISON_WEIGHTS if settings.USE_ENHANCED_PROCESSING else None
        }

        # Verificar estadísticas de base de datos
        try:
            from app.database.models import PersonModel
            stats = await PersonModel.get_system_stats()
            db_info = stats
        except Exception as e:
            db_info = {"error": str(e)}

        # Información de dependencias
        dependencies_info = {}
        try:
            import cv2
            dependencies_info["opencv"] = cv2.__version__
        except ImportError:
            dependencies_info["opencv"] = "not_installed"

        try:
            import numpy as np
            dependencies_info["numpy"] = np.__version__
        except ImportError:
            dependencies_info["numpy"] = "not_installed"

        try:
            import sklearn
            dependencies_info["sklearn"] = sklearn.__version__
        except ImportError:
            dependencies_info["sklearn"] = "not_installed"

        try:
            import dlib
            dependencies_info["dlib"] = "available"
        except ImportError:
            dependencies_info["dlib"] = "not_installed"

        return {
            "system": {
                "name": "Sistema de Reconocimiento Facial",
                "version": "2.1.0",
                "mode": "enhanced" if settings.USE_ENHANCED_PROCESSING else "traditional",
                "environment": "development" if settings.DEBUG else "production"
            },
            "configuration": config_info,
            "database": db_info,
            "dependencies": dependencies_info,
            "capabilities": {
                "facial_recognition": True,
                "multiple_detectors": settings.USE_MULTIPLE_DETECTORS,
                "adaptive_thresholds": settings.ADAPTIVE_THRESHOLD,
                "enhanced_features": settings.USE_ENHANCED_PROCESSING,
                "real_time_processing": True
            }
        }

    except Exception as e:
        logger.error(f"Error obteniendo información del sistema: {e}")
        raise HTTPException(status_code=500, detail="Error obteniendo información del sistema")


if __name__ == "__main__":
    print("🚀 Iniciando Sistema de Reconocimiento Facial Mejorado")
    print("📚 Documentación: http://localhost:8000/docs")
    print("🏥 Health Check: http://localhost:8000/health")
    print("ℹ️ Información: http://localhost:8000/info")
    print("⚙️ Admin Panel: http://localhost:8000/api/admin/stats")

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )