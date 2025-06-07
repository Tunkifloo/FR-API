from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse
from typing import Optional, List
from app.database.models import PersonModel
from app.core.utils import (
    generate_unique_pk, validate_student_id, save_uploaded_file,
    validate_email, validate_image_file, log_function_call
)
from app.core.facial_processing import FacialProcessor
from config import settings
import logging
import os
import json
import time
from datetime import datetime

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/register")
async def register_person(
        nombre: str = Form(...),
        apellidos: str = Form(...),
        correo: str = Form(...),
        id_estudiante: Optional[str] = Form(None),
        foto: UploadFile = File(...)
):
    """Registrar nueva persona en el sistema con procesamiento mejorado"""
    start_time = time.time()
    person_id = None  # Inicializar al principio
    image_path = None  # Inicializar al principio

    try:
        # Log de la función
        log_function_call("register_person", {
            "nombre": nombre,
            "correo": correo,
            "has_photo": foto is not None,
            "enhanced_processing": settings.USE_ENHANCED_PROCESSING
        })

        # Validaciones básicas mejoradas
        if not nombre.strip() or not apellidos.strip():
            raise HTTPException(status_code=400, detail="Nombre y apellidos son requeridos")

        if not validate_email(correo):
            raise HTTPException(status_code=400, detail="Correo electrónico inválido")

        # Verificar si el correo ya existe
        existing_person = await PersonModel.check_email_exists(correo.lower())
        if existing_person:
            raise HTTPException(status_code=400, detail="El correo electrónico ya está registrado")

        # Validar ID de estudiante si se proporciona
        if id_estudiante and not validate_student_id(id_estudiante):
            raise HTTPException(status_code=400, detail="ID de estudiante inválido (6-20 dígitos)")

        if id_estudiante and await PersonModel.check_student_id_exists(id_estudiante):
            raise HTTPException(status_code=400, detail="ID de estudiante ya registrado")

        # Validar archivo de imagen
        if not foto.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="El archivo debe ser una imagen")

        # Leer y validar contenido de la imagen
        image_content = await foto.read()
        is_valid, validation_message = validate_image_file(image_content, foto.filename)
        if not is_valid:
            raise HTTPException(status_code=400, detail=f"Imagen inválida: {validation_message}")

        # Guardar imagen
        filename = f"{generate_unique_pk()}_{foto.filename}"
        image_path = save_uploaded_file(image_content, filename)
        logger.info(f"Imagen guardada en: {image_path}")

        # Procesar imagen facial con sistema mejorado o tradicional
        processor = FacialProcessor()

        if settings.USE_ENHANCED_PROCESSING:
            # Procesamiento mejorado con detección híbrida
            logger.info("Usando procesamiento mejorado con detección híbrida")
            processed_image = processor.preprocess_image(image_path)
            if processed_image is None:
                raise HTTPException(status_code=400, detail="Error en preprocesamiento de la imagen")

            # Usar detección híbrida (DNN + Cascadas)
            faces = processor.detect_faces_hybrid(processed_image)
            if not faces:
                raise HTTPException(status_code=400, detail="No se detectaron rostros en la imagen")

            # Log de método de detección usado
            logger.info(f"Rostros detectados: {len(faces)}")

            # Extracción mejorada de características
            features = processor.extract_enhanced_features(processed_image, faces[0])
            if features is None:
                raise HTTPException(status_code=400, detail="Error extrayendo características faciales")

            method = "enhanced_v2"  # Nueva versión con características avanzadas
            logger.info(f"Características avanzadas extraídas: {len(features)} valores")

        else:
            # Procesamiento tradicional (mantener compatibilidad)
            logger.info("Usando procesamiento tradicional")
            processed_image = processor.process_image(image_path)
            if processed_image is None:
                raise HTTPException(status_code=400, detail="Error procesando la imagen")

            faces = processor.detect_faces(processed_image)
            if not faces:
                raise HTTPException(status_code=400, detail="No se detectaron rostros en la imagen")

            features = processor.extract_face_features(processed_image, faces[0])
            if features is None:
                raise HTTPException(status_code=400, detail="Error extrayendo características faciales")

            method = "opencv_tradicional"
            logger.info(f"Características tradicionales extraídas: {len(features)} valores")

        # Preparar datos de la persona
        person_data = {
            'nombre': nombre.strip(),
            'apellidos': apellidos.strip(),
            'correo': correo.strip().lower(),
            'id_estudiante': id_estudiante.strip() if id_estudiante else None,
            'pk': generate_unique_pk()
        }

        # Insertar en base de datos con método específico
        person_id = await PersonModel.insert_person(person_data, image_path, features.tolist(), method)

        if not person_id:
            raise HTTPException(status_code=500, detail="Error guardando persona en base de datos")

        # Crear backup JSON con información del método usado
        await _create_json_backup_enhanced(person_data, features.tolist(), image_path, method)

        # Calcular tiempo de procesamiento
        processing_time = time.time() - start_time

        log_function_call("register_person", execution_time=processing_time)
        logger.info(f"Persona registrada exitosamente: ID {person_id} con método {method}")

        return {
            "message": "Persona registrada exitosamente",
            "person_id": person_id,
            "pk": person_data['pk'],
            "features_count": len(features),
            "faces_detected": len(faces),
            "processing_method": method,
            "processing_time": round(processing_time, 2),
            "system_info": {
                "enhanced_processing": settings.USE_ENHANCED_PROCESSING,
                "feature_method": settings.FEATURE_METHOD,
                "threshold": settings.DEFAULT_SIMILARITY_THRESHOLD,
                "dnn_detection": settings.USE_DNN_DETECTION,
                "face_embeddings": settings.USE_FACE_EMBEDDINGS,
                "voting_system": settings.USE_VOTING_SYSTEM
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error registrando persona: {e}")
        raise HTTPException(status_code=500, detail=f"Error interno del servidor: {str(e)}")

    finally:
        # Limpiar archivo temporal si hay error y no se guardó la persona
        if image_path and os.path.exists(image_path) and person_id is None:
            try:
                os.remove(image_path)
                logger.info(f"Archivo temporal eliminado: {image_path}")
            except Exception as e:
                logger.warning(f"No se pudo eliminar archivo temporal {image_path}: {e}")


@router.post("/register/batch")
async def register_multiple_persons(
        persons_data: List[dict]
):
    """Registrar múltiples personas en lote (para migración masiva)"""
    if not settings.MIGRATION_ENABLED:
        raise HTTPException(status_code=403, detail="Registro en lote no habilitado")

    results = {
        "total": len(persons_data),
        "successful": 0,
        "failed": 0,
        "errors": []
    }

    for person_data in persons_data:
        try:
            # Aquí implementar lógica de registro en lote
            # Por ahora solo retornar estructura
            results["successful"] += 1
        except Exception as e:
            results["failed"] += 1
            results["errors"].append(str(e))

    return results


@router.get("/list")
async def list_persons():
    """Listar todas las personas registradas con información mejorada"""
    try:
        persons = await PersonModel.list_all_persons()

        # Obtener estadísticas adicionales
        stats = await PersonModel.get_system_stats()

        return {
            "total": len(persons),
            "persons": persons,
            "system_stats": stats,
            "pagination": {
                "page": 1,
                "per_page": len(persons),
                "total_pages": 1
            }
        }
    except Exception as e:
        logger.error(f"Error listando personas: {e}")
        raise HTTPException(status_code=500, detail="Error obteniendo lista de personas")


@router.get("/search/email/{email}")
async def search_by_email(email: str):
    """Buscar persona por correo electrónico con información mejorada"""
    try:
        if not validate_email(email):
            raise HTTPException(status_code=400, detail="Formato de email inválido")

        person = await PersonModel.get_person_by_email(email)
        if not person:
            raise HTTPException(status_code=404, detail="Persona no encontrada")

        # Remover datos sensibles de la respuesta
        safe_person = person.copy()
        safe_person.pop('foto', None)  # No incluir imagen en base64
        safe_person.pop('caracteristicas', None)  # No incluir características por seguridad

        # Agregar información del sistema
        safe_person['system_info'] = {
            'processing_method': person.get('metodo', 'opencv_tradicional'),
            'algorithm_version': person.get('version', '1.0'),
            'extraction_date': person.get('fecha_extraccion'),
            'features_count': len(person.get('caracteristicas', [])) if person.get('caracteristicas') else 0
        }

        return safe_person

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error buscando persona por email: {e}")
        raise HTTPException(status_code=500, detail="Error interno del servidor")


@router.get("/search/student/{student_id}")
async def search_by_student_id(student_id: str):
    """Buscar persona por ID de estudiante con información mejorada"""
    try:
        if not validate_student_id(student_id):
            raise HTTPException(status_code=400, detail="Formato de ID de estudiante inválido")

        person = await PersonModel.get_person_by_student_id(student_id)
        if not person:
            raise HTTPException(status_code=404, detail="Persona no encontrada")

        # Remover datos sensibles de la respuesta
        safe_person = person.copy()
        safe_person.pop('foto', None)  # No incluir imagen en base64
        safe_person.pop('caracteristicas', None)  # No incluir características por seguridad

        # Agregar información del sistema
        safe_person['system_info'] = {
            'processing_method': person.get('metodo', 'opencv_tradicional'),
            'algorithm_version': person.get('version', '1.0'),
            'extraction_date': person.get('fecha_extraccion'),
            'features_count': len(person.get('caracteristicas', [])) if person.get('caracteristicas') else 0
        }

        return safe_person

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error buscando persona por ID estudiante: {e}")
        raise HTTPException(status_code=500, detail="Error interno del servidor")


@router.get("/{person_id}")
async def get_person_by_id(person_id: int):
    """Obtener persona por ID"""
    try:
        # Esta función requeriría implementar get_person_by_id en PersonModel
        raise HTTPException(status_code=501, detail="Función no implementada aún")
    except Exception as e:
        logger.error(f"Error obteniendo persona por ID: {e}")
        raise HTTPException(status_code=500, detail="Error interno del servidor")


@router.put("/{person_id}/update-features")
async def update_person_features(person_id: int, foto: UploadFile = File(...)):
    """Actualizar características faciales de una persona existente"""
    if not settings.MIGRATION_ENABLED:
        raise HTTPException(status_code=403, detail="Actualización de características no habilitada")

    image_path = None

    try:
        # Validar imagen
        if not foto.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="El archivo debe ser una imagen")

        image_content = await foto.read()
        is_valid, validation_message = validate_image_file(image_content, foto.filename)
        if not is_valid:
            raise HTTPException(status_code=400, detail=f"Imagen inválida: {validation_message}")

        # Guardar imagen temporal
        filename = f"update_{person_id}_{generate_unique_pk()}_{foto.filename}"
        image_path = save_uploaded_file(image_content, filename)

        # Procesar con sistema mejorado
        processor = FacialProcessor()

        if settings.USE_ENHANCED_PROCESSING:
            processed_image = processor.preprocess_image(image_path)
            faces = processor.detect_faces_robust(processed_image)
            if not faces:
                raise HTTPException(status_code=400, detail="No se detectaron rostros")
            features = processor.extract_enhanced_features(processed_image, faces[0])
            method = "enhanced"
        else:
            processed_image = processor.process_image(image_path)
            faces = processor.detect_faces(processed_image)
            if not faces:
                raise HTTPException(status_code=400, detail="No se detectaron rostros")
            features = processor.extract_face_features(processed_image, faces[0])
            method = "opencv_tradicional"

        if features is None:
            raise HTTPException(status_code=400, detail="Error extrayendo características")

        # Actualizar en base de datos
        success = await PersonModel.update_person_features(person_id, features.tolist(), method)

        if not success:
            raise HTTPException(status_code=500, detail="Error actualizando características")

        return {
            "message": "Características actualizadas exitosamente",
            "person_id": person_id,
            "method": method,
            "features_count": len(features),
            "faces_detected": len(faces)
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error actualizando características de persona {person_id}: {e}")
        raise HTTPException(status_code=500, detail="Error interno del servidor")

    finally:
        # Limpiar archivo temporal
        if image_path and os.path.exists(image_path):
            try:
                os.remove(image_path)
            except Exception as e:
                logger.warning(f"No se pudo eliminar archivo temporal {image_path}: {e}")


@router.get("/stats/processing")
async def get_processing_stats():
    """Obtener estadísticas de procesamiento"""
    try:
        stats = await PersonModel.get_system_stats()

        # Agregar información de configuración actual
        processing_stats = {
            "database_stats": stats,
            "current_config": {
                "enhanced_processing": settings.USE_ENHANCED_PROCESSING,
                "feature_method": settings.FEATURE_METHOD,
                "default_threshold": settings.DEFAULT_SIMILARITY_THRESHOLD,
                "adaptive_threshold": settings.ADAPTIVE_THRESHOLD,
                "multiple_detectors": settings.USE_MULTIPLE_DETECTORS
            },
            "system_capabilities": {
                "dlib_available": settings.USE_DLIB,
                "sklearn_available": True,  # Requerido para sistema mejorado
                "migration_enabled": settings.MIGRATION_ENABLED
            }
        }

        return processing_stats

    except Exception as e:
        logger.error(f"Error obteniendo estadísticas de procesamiento: {e}")
        raise HTTPException(status_code=500, detail="Error obteniendo estadísticas")


async def _create_json_backup_enhanced(person_data: dict, features: list, image_path: str, method: str):
    """Crear backup JSON de la persona con información del método mejorado"""
    try:
        # Crear directorio si no existe
        os.makedirs(settings.JSON_BACKUP_DIR, exist_ok=True)

        backup_data = {
            'datos_personales': person_data,
            'caracteristicas_faciales': features,
            'metadata': {
                'fecha_creacion': datetime.now().isoformat(),
                'archivo_foto_original': os.path.basename(image_path),
                'total_caracteristicas': len(features),
                'metodo_extraccion': method,
                'umbral_similitud': settings.DEFAULT_SIMILARITY_THRESHOLD,
                'base_datos': 'MySQL',
                'version_algoritmo': "2.0" if method == "enhanced" else "1.0",
                'sistema_mejorado': settings.USE_ENHANCED_PROCESSING,
                'configuracion': {
                    'adaptive_threshold': settings.ADAPTIVE_THRESHOLD,
                    'multiple_detectors': settings.USE_MULTIPLE_DETECTORS,
                    'feature_method': settings.FEATURE_METHOD
                }
            }
        }

        filename = f"{settings.JSON_BACKUP_DIR}/persona_{person_data['pk']}.json"
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(backup_data, f, indent=2, ensure_ascii=False)

        logger.info(f"Backup JSON mejorado creado: {filename}")

    except Exception as e:
        logger.error(f"Error creando backup JSON mejorado: {e}")


# Mantener función original para compatibilidad
async def _create_json_backup(person_data: dict, features: list, image_path: str):
    """Función original mantenida para compatibilidad"""
    await _create_json_backup_enhanced(person_data, features, image_path, "opencv_tradicional")