import logging
import os
import random
from datetime import datetime
from typing import Optional
import base64
import cv2
import numpy as np


def configure_logging():
    """Configurar sistema de logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('facial_recognition.log'),
            logging.StreamHandler()
        ]
    )


def create_directories():
    """Crear directorios necesarios"""
    dirs = ['uploads', 'models', 'exports', 'json_backup']
    for directory in dirs:
        os.makedirs(directory, exist_ok=True)


def generate_unique_pk() -> str:
    """Generar clave primaria única"""
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    random_num = random.randint(1000, 9999)
    return f"PK_{timestamp}_{random_num}"


def validate_student_id(student_id: str) -> bool:
    """Validar formato de ID de estudiante"""
    if not student_id:
        return False

    student_id = str(student_id).strip()
    return student_id.isdigit() and 6 <= len(student_id) <= 20


def image_to_base64(image_path: str) -> Optional[str]:
    """Convertir imagen a base64"""
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode('utf-8')
    except Exception as e:
        logging.error(f"Error convirtiendo imagen a base64: {e}")
        return None


def base64_to_image(base64_string: str, output_path: str) -> bool:
    """Convertir base64 a imagen"""
    try:
        img_data = base64.b64decode(base64_string)
        with open(output_path, 'wb') as img_file:
            img_file.write(img_data)
        return True
    except Exception as e:
        logging.error(f"Error convirtiendo base64 a imagen: {e}")
        return False


def save_uploaded_file(file: bytes, filename: str) -> str:
    """Guardar archivo subido"""
    create_directories()
    file_path = os.path.join("uploads", filename)

    with open(file_path, "wb") as buffer:
        buffer.write(file)

    return file_path