from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from app.database.connection import get_db_connection, init_database
from app.endpoints import admin, data_management, persons, recognition
from app.core.utils import configure_logging
import logging

# Configurar logging
configure_logging()
logger = logging.getLogger(__name__)

# Crear aplicación FastAPI
app = FastAPI(
    title="Sistema de Reconocimiento Facial",
    description="API completa para reconocimiento facial con OpenCV y MySQL",
    version="2.0.0",
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
    logger.info("🚀 Iniciando Sistema de Reconocimiento Facial")

    # Inicializar base de datos
    if await init_database():
        logger.info("✅ Base de datos MySQL inicializada correctamente")
    else:
        logger.error("❌ Error inicializando base de datos MySQL")

    logger.info("🎯 Sistema listo para recibir peticiones")


@app.on_event("shutdown")
async def shutdown_event():
    """Limpiar recursos al cerrar"""
    logger.info("🔽 Cerrando Sistema de Reconocimiento Facial")


@app.get("/")
async def root():
    """Endpoint raíz con información del sistema"""
    return {
        "message": "Sistema de Reconocimiento Facial - FastAPI",
        "version": "2.0.0",
        "status": "activo",
        "docs": "/docs",
        "admin_panel": "/api/admin/stats"
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

        return {
            "status": "healthy",
            "database": db_status,
            "timestamp": "2025-01-01T00:00:00Z"
        }
    except Exception as e:
        logger.error(f"Error en health check: {e}")
        raise HTTPException(status_code=500, detail="Sistema no disponible")


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
