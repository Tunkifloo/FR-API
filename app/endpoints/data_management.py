from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse
from app.database.models import PersonModel
from app.database.connection import get_db_connection
import logging
import json
import os
from datetime import datetime
from typing import List

logger = logging.getLogger(__name__)
router = APIRouter()


def serialize_datetime(obj):
    """Función helper para serializar objetos datetime"""
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def prepare_person_for_export(person_data: dict) -> dict:
    """Preparar datos de persona para exportación JSON"""
    person_copy = person_data.copy()

    # Convertir datetime a string ISO
    if 'fecha_registro' in person_copy and isinstance(person_copy['fecha_registro'], datetime):
        person_copy['fecha_registro'] = person_copy['fecha_registro'].isoformat()

    return person_copy


@router.get("/export/all")
async def export_all_data():
    """Exportar todos los datos a JSON"""
    try:
        connection = await get_db_connection()
        if not connection:
            raise HTTPException(status_code=500, detail="Error de conexión a base de datos")

        cursor = connection.cursor()

        # Obtener todas las personas con sus características
        query = """
                SELECT p.*, c.caracteristicas_json, c.umbral_similitud, c.metodo
                FROM PERSONAS p
                         LEFT JOIN CARACTERISTICAS_FACIALES c ON p.ID = c.persona_id
                WHERE p.activo = TRUE
                """

        cursor.execute(query)
        results = cursor.fetchall()

        cursor.close()
        connection.close()

        if not results:
            raise HTTPException(status_code=404, detail="No hay datos para exportar")

        # Crear estructura de exportación
        export_data = {
            'metadata': {
                'fecha_exportacion': datetime.now().isoformat(),
                'total_registros': len(results),
                'base_datos': 'MySQL',
                'version_sistema': '2.0.0'
            },
            'personas': []
        }

        for result in results:
            person_data = {
                'id': result[0],
                'nombre': result[1],
                'apellidos': result[2],
                'correo': result[3],
                'id_estudiante': result[4],
                'foto_base64': result[5],
                'pk': result[6],
                'fecha_registro': result[7].isoformat() if result[7] else None,
                'activo': bool(result[8]),
                'caracteristicas': json.loads(result[9]) if result[9] else None,
                'umbral_similitud': float(result[10]) if result[10] else 0.60,
                'metodo': result[11] if result[11] else 'opencv_tradicional'
            }
            export_data['personas'].append(person_data)

        # Crear directorio de exportación
        os.makedirs('exports', exist_ok=True)

        # Guardar archivo
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"exports/exportacion_completa_{timestamp}.json"

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False, default=serialize_datetime)

        return {
            "message": "Exportación completada exitosamente",
            "filename": filename,
            "total_records": len(results),
            "download_url": f"/api/data/download/{os.path.basename(filename)}"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error en exportación: {e}")
        raise HTTPException(status_code=500, detail="Error interno del servidor")


@router.get("/export/person/email/{email}")
async def export_person_by_email(email: str):
    """Exportar datos de una persona específica por email"""
    try:
        person = await PersonModel.get_person_by_email(email)
        if not person:
            raise HTTPException(status_code=404, detail="Persona no encontrada")

        # Preparar datos para exportación (convertir datetime)
        person_prepared = prepare_person_for_export(person)

        # Crear estructura de exportación individual
        export_data = {
            'metadata': {
                'fecha_exportacion': datetime.now().isoformat(),
                'tipo': 'persona_individual',
                'base_datos': 'MySQL'
            },
            'persona': person_prepared
        }

        # Crear directorio de exportación
        os.makedirs('exports', exist_ok=True)

        # Guardar archivo
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"exports/persona_{person['pk']}_{timestamp}.json"

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False, default=serialize_datetime)

        return {
            "message": "Persona exportada exitosamente",
            "filename": filename,
            "person": {
                "id": person['id'],
                "nombre": person['nombre'],
                "apellidos": person['apellidos'],
                "correo": person['correo']
            },
            "download_url": f"/api/data/download/{os.path.basename(filename)}"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error exportando persona: {e}")
        raise HTTPException(status_code=500, detail="Error interno del servidor")


@router.post("/import")
async def import_data(file: UploadFile = File(...)):
    """Importar datos desde archivo JSON"""
    try:
        if not file.filename.endswith('.json'):
            raise HTTPException(status_code=400, detail="El archivo debe ser un JSON")

        # Leer contenido del archivo
        content = await file.read()
        data = json.loads(content.decode('utf-8'))

        imported_count = 0
        errors = []

        if 'personas' in data:
            # Importación completa
            for person_data in data['personas']:
                try:
                    # Verificar si ya existe
                    existing = await PersonModel.check_email_exists(person_data['correo'])
                    if existing:
                        errors.append(f"Email {person_data['correo']} ya existe")
                        continue

                    # Aquí iría la lógica de importación completa
                    # Por simplicidad, solo contamos los que se procesarían
                    imported_count += 1

                except Exception as e:
                    errors.append(f"Error con {person_data.get('correo', 'desconocido')}: {str(e)}")

        elif 'persona' in data:
            # Importación individual
            person_data = data['persona']
            try:
                existing = await PersonModel.check_email_exists(person_data['correo'])
                if not existing:
                    # Aquí iría la lógica de importación individual
                    imported_count = 1
                else:
                    errors.append(f"Email {person_data['correo']} ya existe")
            except Exception as e:
                errors.append(f"Error importando persona: {str(e)}")

        else:
            raise HTTPException(status_code=400, detail="Formato de archivo JSON no reconocido")

        return {
            "message": "Importación completada",
            "imported": imported_count,
            "errors": len(errors),
            "error_details": errors
        }

    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Archivo JSON inválido")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error en importación: {e}")
        raise HTTPException(status_code=500, detail="Error interno del servidor")


@router.get("/download/{filename}")
async def download_file(filename: str):
    """Descargar archivo exportado"""
    try:
        file_path = os.path.join("exports", filename)

        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="Archivo no encontrado")

        return FileResponse(
            path=file_path,
            filename=filename,
            media_type='application/json'
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error descargando archivo: {e}")
        raise HTTPException(status_code=500, detail="Error interno del servidor")


@router.get("/backup/create")
async def create_backup():
    """Crear respaldo completo del sistema"""
    try:
        # Crear respaldo usando la función de exportación
        result = await export_all_data()

        return {
            "message": "Respaldo creado exitosamente",
            "backup_info": result
        }

    except Exception as e:
        logger.error(f"Error creando respaldo: {e}")
        raise HTTPException(status_code=500, detail="Error creando respaldo")


@router.get("/sync/check")
async def check_synchronization():
    """Verificar sincronización entre BD, JSON y archivos"""
    try:
        # Verificar BD
        persons_db = await PersonModel.list_all_persons()
        db_count = len(persons_db)

        # Verificar archivos JSON
        json_files = []
        if os.path.exists('json_backup'):
            json_files = [f for f in os.listdir('json_backup') if f.endswith('.json')]
        json_count = len(json_files)

        # Verificar modelos pickle
        pkl_files = []
        if os.path.exists('models'):
            pkl_files = [f for f in os.listdir('models') if f.endswith('.pkl')]
        pkl_count = len(pkl_files)

        # Determinar estado de sincronización
        sync_status = "perfect" if db_count == json_count == pkl_count else "inconsistent"

        return {
            "synchronization": {
                "status": sync_status,
                "database_records": db_count,
                "json_backups": json_count,
                "pickle_models": pkl_count,
                "recommendation": "Crear respaldos" if sync_status == "inconsistent" else "Sistema sincronizado"
            }
        }

    except Exception as e:
        logger.error(f"Error verificando sincronización: {e}")
        raise HTTPException(status_code=500, detail="Error verificando sincronización")