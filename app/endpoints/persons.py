from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse
from typing import Optional, List
from app.database.models import PersonModel
from app.core.utils import generate_unique_pk, validate_student_id, save_uploaded_file
from app.core.facial_processing import FacialProcessor
from app.core.feature_extraction import FeatureExtractor
import logging
import os
import json
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
    """Registrar nueva persona en el sistema"""
    try:
        # Validaciones básicas
        if not nombre.strip() or not apellidos.strip():
            raise HTTPException(status_code=400, detail="Nombre y apellidos son requeridos")

        if "@" not in correo:
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

        # Guardar imagen
        image_content = await foto.read()
        filename = f"{generate_unique_pk()}_{foto.filename}"
        image_path = save_uploaded_file(image_content, filename)

        # Procesar imagen facial
        processor = FacialProcessor()
        processed_image = processor.process_image(image_path)

        if processed_image is None:
            os.remove(image_path)  # Limpiar archivo
            raise HTTPException(status_code=400, detail="Error procesando la imagen")

        # Detectar rostros
        faces = processor.detect_faces(processed_image)
        if not faces:
            os.remove(image_path)  # Limpiar archivo
            raise HTTPException(status_code=400, detail="No se detectaron rostros en la imagen")

        # Extraer características del primer rostro detectado
        features = processor.extract_face_features(processed_image, faces[0])
        if features is None:
            os.remove(image_path)  # Limpiar archivo
            raise HTTPException(status_code=400, detail="Error extrayendo características faciales")

        # Preparar datos de la persona
        person_data = {
            'nombre': nombre.strip(),
            'apellidos': apellidos.strip(),
            'correo': correo.strip().lower(),
            'id_estudiante': id_estudiante.strip() if id_estudiante else None,
            'pk': generate_unique_pk()
        }

        # Insertar en base de datos
        person_id = await PersonModel.insert_person(person_data, image_path, features.tolist())

        if not person_id:
            os.remove(image_path)  # Limpiar archivo
            raise HTTPException(status_code=500, detail="Error guardando persona en base de datos")

        # Crear backup JSON
        await _create_json_backup(person_data, features.tolist(), image_path)

        logger.info(f"Persona registrada exitosamente: ID {person_id}")

        return {
            "message": "Persona registrada exitosamente",
            "person_id": person_id,
            "pk": person_data['pk'],
            "features_count": len(features),
            "faces_detected": len(faces)
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error registrando persona: {e}")
        raise HTTPException(status_code=500, detail="Error interno del servidor")


@router.get("/list")
async def list_persons():
    """Listar todas las personas registradas"""
    try:
        persons = await PersonModel.list_all_persons()
        return {
            "total": len(persons),
            "persons": persons
        }
    except Exception as e:
        logger.error(f"Error listando personas: {e}")
        raise HTTPException(status_code=500, detail="Error obteniendo lista de personas")


@router.get("/search/email/{email}")
async def search_by_email(email: str):
    """Buscar persona por correo electrónico"""
    try:
        person = await PersonModel.get_person_by_email(email)
        if not person:
            raise HTTPException(status_code=404, detail="Persona no encontrada")

        # Remover datos sensibles de la respuesta
        safe_person = person.copy()
        safe_person.pop('foto', None)  # No incluir imagen en base64

        return safe_person

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error buscando persona por email: {e}")
        raise HTTPException(status_code=500, detail="Error interno del servidor")


@router.get("/search/student/{student_id}")
async def search_by_student_id(student_id: str):
    """Buscar persona por ID de estudiante"""
    try:
        person = await PersonModel.get_person_by_student_id(student_id)
        if not person:
            raise HTTPException(status_code=404, detail="Persona no encontrada")

        # Remover datos sensibles de la respuesta
        safe_person = person.copy()
        safe_person.pop('foto', None)  # No incluir imagen en base64

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


async def _create_json_backup(person_data: dict, features: list, image_path: str):
    """Crear backup JSON de la persona"""
    try:
        # Crear directorio si no existe
        os.makedirs('json_backup', exist_ok=True)

        backup_data = {
            'datos_personales': person_data,
            'caracteristicas_faciales': features,
            'metadata': {
                'fecha_creacion': datetime.now().isoformat(),
                'archivo_foto_original': os.path.basename(image_path),
                'total_caracteristicas': len(features),
                'metodo_extraccion': 'opencv_tradicional',
                'umbral_similitud': 0.75,
                'base_datos': 'MySQL'
            }
        }

        filename = f"json_backup/persona_{person_data['pk']}.json"
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(backup_data, f, indent=2, ensure_ascii=False)

        logger.info(f"Backup JSON creado: {filename}")

    except Exception as e:
        logger.error(f"Error creando backup JSON: {e}")