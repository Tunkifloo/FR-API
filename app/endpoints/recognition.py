from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse
from app.database.models import PersonModel
from app.core.facial_processing import FacialProcessor
from app.core.utils import save_uploaded_file, generate_unique_pk
from config import settings
import logging
import os
import numpy as np

logger = logging.getLogger(__name__)
router = APIRouter()


def convert_numpy_types(obj):
    """Convertir tipos de NumPy a tipos nativos de Python"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    return obj


def get_processor_instance():
    """Obtener instancia del procesador según configuración"""
    if settings.USE_ENHANCED_PROCESSING:
        logger.info("Usando procesador mejorado")
        return FacialProcessor()  # Tu clase mejorada
    else:
        logger.info("Usando procesador tradicional")
        # Aquí podrías importar tu clase original si la mantienes separada
        return FacialProcessor()


@router.post("/compare/email")
async def recognize_by_email(
        email: str = Form(...),
        test_image: UploadFile = File(...)
):
    """Realizar reconocimiento facial comparando con persona por email"""
    try:
        # Buscar persona en base de datos
        person = await PersonModel.get_person_by_email(email)
        if not person:
            raise HTTPException(status_code=404, detail="Persona no encontrada")

        if not person['caracteristicas']:
            raise HTTPException(status_code=400, detail="No hay características faciales para esta persona")

        # Procesar imagen de prueba con sistema mejorado o tradicional
        result = await _process_recognition_image_enhanced(test_image, person)

        # Convertir tipos NumPy antes de devolver
        response = {
            "person": {
                "id": int(person['id']),
                "nombre": str(person['nombre']),
                "apellidos": str(person['apellidos']),
                "correo": str(person['correo']),
                "id_estudiante": str(person['id_estudiante']) if person['id_estudiante'] else None,
                "metodo_usado": person.get('metodo', 'opencv_tradicional'),
                "version_algoritmo": person.get('version', '1.0')
            },
            "recognition_result": convert_numpy_types(result),
            "system_info": {
                "enhanced_processing": settings.USE_ENHANCED_PROCESSING,
                "feature_method": settings.FEATURE_METHOD,
                "comparison_method": "multi_metric" if settings.USE_ENHANCED_PROCESSING else "traditional"
            }
        }

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error en reconocimiento por email: {e}")
        raise HTTPException(status_code=500, detail="Error interno del servidor")


@router.post("/compare/student")
async def recognize_by_student_id(
        student_id: str = Form(...),
        test_image: UploadFile = File(...)
):
    """Realizar reconocimiento facial comparando con persona por ID de estudiante"""
    try:
        # Buscar persona en base de datos
        person = await PersonModel.get_person_by_student_id(student_id)
        if not person:
            raise HTTPException(status_code=404, detail="Persona no encontrada")

        if not person['caracteristicas']:
            raise HTTPException(status_code=400, detail="No hay características faciales para esta persona")

        # Procesar imagen de prueba
        result = await _process_recognition_image_enhanced(test_image, person)

        # Convertir tipos NumPy antes de devolver
        response = {
            "person": {
                "id": int(person['id']),
                "nombre": str(person['nombre']),
                "apellidos": str(person['apellidos']),
                "correo": str(person['correo']),
                "id_estudiante": str(person['id_estudiante']) if person['id_estudiante'] else None,
                "metodo_usado": person.get('metodo', 'opencv_tradicional'),
                "version_algoritmo": person.get('version', '1.0')
            },
            "recognition_result": convert_numpy_types(result),
            "system_info": {
                "enhanced_processing": settings.USE_ENHANCED_PROCESSING,
                "feature_method": settings.FEATURE_METHOD,
                "comparison_method": "multi_metric" if settings.USE_ENHANCED_PROCESSING else "traditional"
            }
        }

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error en reconocimiento por ID estudiante: {e}")
        raise HTTPException(status_code=500, detail="Error interno del servidor")


@router.post("/compare/id/{person_id}")
async def recognize_by_person_id(
        person_id: int,
        test_image: UploadFile = File(...)
):
    """Realizar reconocimiento facial comparando con persona por ID"""
    try:
        # Esta función requeriría implementar get_person_by_id en PersonModel
        raise HTTPException(status_code=501, detail="Función no implementada aún")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error en reconocimiento por ID: {e}")
        raise HTTPException(status_code=500, detail="Error interno del servidor")


@router.post("/identify")
async def identify_person(test_image: UploadFile = File(...)):
    """Identificar persona comparando contra toda la base de datos con sistema mejorado"""
    try:
        # Validar archivo de imagen
        if not test_image.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="El archivo debe ser una imagen")

        # Guardar imagen temporal
        image_content = await test_image.read()
        filename = f"temp_{generate_unique_pk()}_{test_image.filename}"
        image_path = save_uploaded_file(image_content, filename)

        try:
            # Obtener procesador según configuración
            processor = get_processor_instance()

            # Procesar imagen con método mejorado o tradicional
            if settings.USE_ENHANCED_PROCESSING:
                processed_image = processor.preprocess_image(image_path)
                if processed_image is None:
                    raise HTTPException(status_code=400, detail="Error procesando la imagen")

                # Detección robusta de rostros
                faces = processor.detect_faces_robust(processed_image)
                if not faces:
                    raise HTTPException(status_code=400, detail="No se detectaron rostros en la imagen")

                # Extracción mejorada de características
                test_features = processor.extract_enhanced_features(processed_image, faces[0])
                if test_features is None:
                    raise HTTPException(status_code=400, detail="Error extrayendo características faciales")
            else:
                # Método tradicional
                processed_image = processor.process_image(image_path)
                if processed_image is None:
                    raise HTTPException(status_code=400, detail="Error procesando la imagen")

                faces = processor.detect_faces(processed_image)
                if not faces:
                    raise HTTPException(status_code=400, detail="No se detectaron rostros en la imagen")

                test_features = processor.extract_face_features(processed_image, faces[0])
                if test_features is None:
                    raise HTTPException(status_code=400, detail="Error extrayendo características faciales")

            # Obtener todas las personas de la base de datos
            all_persons = await PersonModel.list_all_persons()

            best_match = None
            best_similarity = 0.0
            matches = []
            comparison_details = []

            # Comparar contra cada persona
            for person_summary in all_persons:
                # Obtener datos completos de la persona
                person = await PersonModel.get_person_by_email(person_summary['correo'])

                if not person or not person['caracteristicas']:
                    continue

                # Comparar características usando método apropiado
                stored_features = np.array(person['caracteristicas'])

                # Usar umbral específico de la persona o el de configuración
                threshold = settings.get_threshold_for_person(person)

                if settings.USE_ENHANCED_PROCESSING:
                    comparison = processor.compare_features_enhanced(
                        stored_features,
                        test_features,
                        threshold
                    )
                else:
                    comparison = processor.compare_features(
                        stored_features,
                        test_features,
                        threshold
                    )

                # Guardar resultado de comparación (convertir tipos NumPy)
                match_result = {
                    "person": {
                        "id": int(person['id']),
                        "nombre": str(person['nombre']),
                        "apellidos": str(person['apellidos']),
                        "correo": str(person['correo']),
                        "id_estudiante": str(person['id_estudiante']) if person['id_estudiante'] else None,
                        "metodo_caracteristicas": person.get('metodo', 'opencv_tradicional'),
                        "version_algoritmo": person.get('version', '1.0')
                    },
                    "similarity": float(comparison['similarity']),
                    "is_match": bool(comparison['is_match']),
                    "threshold": float(comparison['threshold']),
                    "confidence": float(comparison.get('confidence', comparison['similarity']))
                }

                # Agregar detalles adicionales si está disponible (sistema mejorado)
                if settings.USE_ENHANCED_PROCESSING and 'cosine_similarity' in comparison:
                    match_result["detailed_metrics"] = {
                        "cosine_similarity": float(comparison['cosine_similarity']),
                        "correlation": float(comparison['correlation']),
                        "euclidean_similarity": float(comparison['euclidean_similarity']),
                        "manhattan_similarity": float(comparison['manhattan_similarity']),
                        "consistency": float(comparison.get('consistency', 0.0)),
                        "adjusted_threshold": float(comparison.get('adjusted_threshold', threshold))
                    }

                matches.append(match_result)

                # Actualizar mejor coincidencia
                if comparison['similarity'] > best_similarity:
                    best_similarity = float(comparison['similarity'])
                    best_match = match_result

            # Ordenar matches por similitud
            matches.sort(key=lambda x: x['similarity'], reverse=True)

            response = {
                "identification_result": {
                    "best_match": best_match,
                    "confidence": float(best_similarity),
                    "total_comparisons": int(len(matches)),
                    "faces_detected": int(len(faces)),
                    "processing_method": "enhanced" if settings.USE_ENHANCED_PROCESSING else "traditional",
                    "feature_extraction_method": settings.FEATURE_METHOD
                },
                "all_matches": matches[:5],  # Top 5 coincidencias
                "system_info": {
                    "enhanced_processing": settings.USE_ENHANCED_PROCESSING,
                    "default_threshold": settings.DEFAULT_SIMILARITY_THRESHOLD,
                    "adaptive_threshold": settings.ADAPTIVE_THRESHOLD,
                    "use_multiple_detectors": settings.USE_MULTIPLE_DETECTORS
                }
            }

            return response

        finally:
            # Limpiar archivo temporal
            if os.path.exists(image_path):
                os.remove(image_path)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error en identificación: {e}")
        raise HTTPException(status_code=500, detail="Error interno del servidor")


@router.get("/stats")
async def get_recognition_stats():
    """Obtener estadísticas del sistema de reconocimiento"""
    try:
        stats = await PersonModel.get_system_stats()

        response = {
            "system_stats": stats,
            "configuration": {
                "enhanced_processing": settings.USE_ENHANCED_PROCESSING,
                "feature_method": settings.FEATURE_METHOD,
                "default_threshold": settings.DEFAULT_SIMILARITY_THRESHOLD,
                "adaptive_threshold": settings.ADAPTIVE_THRESHOLD,
                "use_multiple_detectors": settings.USE_MULTIPLE_DETECTORS,
                "use_dlib": settings.USE_DLIB,
                "comparison_weights": settings.COMPARISON_WEIGHTS if settings.USE_ENHANCED_PROCESSING else None
            },
            "feature_config": settings.get_feature_config(),
            "detection_config": settings.get_detection_config()
        }

        return response

    except Exception as e:
        logger.error(f"Error obteniendo estadísticas: {e}")
        raise HTTPException(status_code=500, detail="Error obteniendo estadísticas del sistema")


async def _process_recognition_image_enhanced(test_image: UploadFile, person: dict) -> dict:
    """Procesar imagen para reconocimiento facial con sistema mejorado o tradicional"""
    # Validar archivo de imagen
    if not test_image.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="El archivo debe ser una imagen")

    # Guardar imagen temporal
    image_content = await test_image.read()
    filename = f"temp_{generate_unique_pk()}_{test_image.filename}"
    image_path = save_uploaded_file(image_content, filename)

    try:
        # Obtener procesador según configuración
        processor = get_processor_instance()

        # Procesar según el método configurado
        if settings.USE_ENHANCED_PROCESSING:
            # Procesamiento mejorado
            processed_image = processor.preprocess_image(image_path)
            if processed_image is None:
                raise HTTPException(status_code=400, detail="Error procesando la imagen de prueba")

            # Detección robusta de rostros
            faces = processor.detect_faces_robust(processed_image)
            if not faces:
                raise HTTPException(status_code=400, detail="No se detectaron rostros en la imagen de prueba")

            # Extracción mejorada de características
            test_features = processor.extract_enhanced_features(processed_image, faces[0])
            if test_features is None:
                raise HTTPException(status_code=400, detail="Error extrayendo características de la imagen de prueba")

            # Comparación mejorada con características almacenadas
            stored_features = np.array(person['caracteristicas'])
            threshold = settings.get_threshold_for_person(person)

            comparison_result = processor.compare_features_enhanced(
                stored_features,
                test_features,
                threshold
            )

            # Resultado detallado del sistema mejorado
            result = {
                "similarity": float(comparison_result['similarity']),
                "cosine_similarity": float(comparison_result['cosine_similarity']),
                "correlation": float(comparison_result['correlation']),
                "euclidean_similarity": float(comparison_result['euclidean_similarity']),
                "manhattan_similarity": float(comparison_result['manhattan_similarity']),
                "segment_similarity": float(comparison_result['segment_similarity']),
                "consistency": float(comparison_result['consistency']),
                "euclidean_distance": float(comparison_result['euclidean_distance']),
                "manhattan_distance": float(comparison_result['manhattan_distance']),
                "is_match": bool(comparison_result['is_match']),
                "threshold": float(comparison_result['threshold']),
                "adjusted_threshold": float(comparison_result['adjusted_threshold']),
                "confidence": float(comparison_result['confidence']),
                "faces_detected": int(len(faces)),
                "features_compared": int(len(test_features)),
                "processing_method": "enhanced",
                "metrics_used": comparison_result['metrics_used']
            }

        else:
            # Procesamiento tradicional (mantener compatibilidad)
            processed_image = processor.process_image(image_path)
            if processed_image is None:
                raise HTTPException(status_code=400, detail="Error procesando la imagen de prueba")

            faces = processor.detect_faces(processed_image)
            if not faces:
                raise HTTPException(status_code=400, detail="No se detectaron rostros en la imagen de prueba")

            test_features = processor.extract_face_features(processed_image, faces[0])
            if test_features is None:
                raise HTTPException(status_code=400, detail="Error extrayendo características de la imagen de prueba")

            # Comparación tradicional
            stored_features = np.array(person['caracteristicas'])
            threshold = settings.get_threshold_for_person(person)

            comparison_result = processor.compare_features(
                stored_features,
                test_features,
                threshold
            )

            # Resultado tradicional
            result = {
                "similarity": float(comparison_result['similarity']),
                "correlation": float(comparison_result['correlation']),
                "distance": float(comparison_result['distance']),
                "distance_similarity": float(comparison_result['distance_similarity']),
                "is_match": bool(comparison_result['is_match']),
                "threshold": float(comparison_result['threshold']),
                "faces_detected": int(len(faces)),
                "features_compared": int(len(test_features)),
                "processing_method": "traditional"
            }

        return result

    finally:
        # Limpiar archivo temporal
        if os.path.exists(image_path):
            os.remove(image_path)


# Mantener función original para compatibilidad hacia atrás
async def _process_recognition_image(test_image: UploadFile, person: dict) -> dict:
    """Función original mantenida para compatibilidad"""
    return await _process_recognition_image_enhanced(test_image, person)