from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse
from app.database.models import PersonModel
from app.core.facial_processing import FacialProcessor
from app.core.utils import save_uploaded_file, generate_unique_pk
import logging
import os
import numpy as np

logger = logging.getLogger(__name__)
router = APIRouter()


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

        # Procesar imagen de prueba
        result = await _process_recognition_image(test_image, person)

        return {
            "person": {
                "id": person['id'],
                "nombre": person['nombre'],
                "apellidos": person['apellidos'],
                "correo": person['correo'],
                "id_estudiante": person['id_estudiante']
            },
            "recognition_result": result
        }

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
        result = await _process_recognition_image(test_image, person)

        return {
            "person": {
                "id": person['id'],
                "nombre": person['nombre'],
                "apellidos": person['apellidos'],
                "correo": person['correo'],
                "id_estudiante": person['id_estudiante']
            },
            "recognition_result": result
        }

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
    """Identificar persona comparando contra toda la base de datos"""
    try:
        # Validar archivo de imagen
        if not test_image.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="El archivo debe ser una imagen")

        # Guardar imagen temporal
        image_content = await test_image.read()
        filename = f"temp_{generate_unique_pk()}_{test_image.filename}"
        image_path = save_uploaded_file(image_content, filename)

        try:
            # Procesar imagen
            processor = FacialProcessor()
            processed_image = processor.process_image(image_path)

            if processed_image is None:
                raise HTTPException(status_code=400, detail="Error procesando la imagen")

            # Detectar rostros
            faces = processor.detect_faces(processed_image)
            if not faces:
                raise HTTPException(status_code=400, detail="No se detectaron rostros en la imagen")

            # Extraer características
            test_features = processor.extract_face_features(processed_image, faces[0])
            if test_features is None:
                raise HTTPException(status_code=400, detail="Error extrayendo características faciales")

            # Obtener todas las personas de la base de datos
            all_persons = await PersonModel.list_all_persons()

            best_match = None
            best_similarity = 0.0
            matches = []

            # Comparar contra cada persona
            for person_summary in all_persons:
                # Obtener datos completos de la persona
                person = await PersonModel.get_person_by_email(person_summary['correo'])

                if not person or not person['caracteristicas']:
                    continue

                # Comparar características
                stored_features = np.array(person['caracteristicas'])
                comparison = processor.compare_features(
                    stored_features,
                    test_features,
                    person['umbral']
                )

                # Guardar resultado de comparación
                match_result = {
                    "person": {
                        "id": person['id'],
                        "nombre": person['nombre'],
                        "apellidos": person['apellidos'],
                        "correo": person['correo'],
                        "id_estudiante": person['id_estudiante']
                    },
                    "similarity": comparison['similarity'],
                    "is_match": comparison['is_match'],
                    "threshold": comparison['threshold']
                }

                matches.append(match_result)

                # Actualizar mejor coincidencia
                if comparison['similarity'] > best_similarity:
                    best_similarity = comparison['similarity']
                    best_match = match_result

            # Ordenar matches por similitud
            matches.sort(key=lambda x: x['similarity'], reverse=True)

            return {
                "identification_result": {
                    "best_match": best_match,
                    "confidence": best_similarity,
                    "total_comparisons": len(matches),
                    "faces_detected": len(faces)
                },
                "all_matches": matches[:5]  # Top 5 coincidencias
            }

        finally:
            # Limpiar archivo temporal
            if os.path.exists(image_path):
                os.remove(image_path)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error en identificación: {e}")
        raise HTTPException(status_code=500, detail="Error interno del servidor")


async def _process_recognition_image(test_image: UploadFile, person: dict) -> dict:
    """Procesar imagen para reconocimiento facial"""
    # Validar archivo de imagen
    if not test_image.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="El archivo debe ser una imagen")

    # Guardar imagen temporal
    image_content = await test_image.read()
    filename = f"temp_{generate_unique_pk()}_{test_image.filename}"
    image_path = save_uploaded_file(image_content, filename)

    try:
        # Procesar imagen
        processor = FacialProcessor()
        processed_image = processor.process_image(image_path)

        if processed_image is None:
            raise HTTPException(status_code=400, detail="Error procesando la imagen de prueba")

        # Detectar rostros
        faces = processor.detect_faces(processed_image)
        if not faces:
            raise HTTPException(status_code=400, detail="No se detectaron rostros en la imagen de prueba")

        # Extraer características
        test_features = processor.extract_face_features(processed_image, faces[0])
        if test_features is None:
            raise HTTPException(status_code=400, detail="Error extrayendo características de la imagen de prueba")

        # Comparar con características almacenadas
        stored_features = np.array(person['caracteristicas'])
        comparison_result = processor.compare_features(
            stored_features,
            test_features,
            person['umbral']
        )

        return {
            "similarity": comparison_result['similarity'],
            "correlation": comparison_result['correlation'],
            "distance": comparison_result['distance'],
            "distance_similarity": comparison_result['distance_similarity'],
            "is_match": comparison_result['is_match'],
            "threshold": comparison_result['threshold'],
            "faces_detected": len(faces),
            "features_compared": len(test_features)
        }

    finally:
        # Limpiar archivo temporal
        if os.path.exists(image_path):
            os.remove(image_path)