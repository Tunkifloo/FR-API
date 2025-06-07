import logging
import os
import random
from datetime import datetime
from typing import Optional, List, Tuple
import base64
import cv2
import numpy as np
from config import settings


def configure_logging():
    """Configurar sistema de logging mejorado"""
    # Crear directorio de logs si no existe
    os.makedirs('logs', exist_ok=True)

    # Configurar formato de logging
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'

    # Configurar handlers
    handlers = [
        logging.FileHandler('logs/facial_recognition.log', encoding='utf-8'),
        logging.StreamHandler()
    ]

    # Configurar logging
    logging.basicConfig(
        level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
        format=log_format,
        handlers=handlers,
        force=True  # Forzar reconfiguración
    )

    # Configurar loggers específicos
    logging.getLogger('mysql.connector').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)


def create_directories():
    """Crear directorios necesarios para el sistema"""
    directories = [
        settings.UPLOAD_DIR,
        settings.MODELS_DIR,
        settings.BACKUP_DIR,
        settings.JSON_BACKUP_DIR,
        'logs',
        'temp',
        'test_images'
    ]

    for directory in directories:
        try:
            os.makedirs(directory, exist_ok=True)
            logging.debug(f"Directorio creado/verificado: {directory}")
        except Exception as e:
            logging.error(f"Error creando directorio {directory}: {e}")


def generate_unique_pk() -> str:
    """Generar clave primaria única con mayor entropía"""
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")[:-3]  # Incluir microsegundos
    random_num = random.randint(10000, 99999)
    return f"PK_{timestamp}_{random_num}"


def validate_student_id(student_id: str) -> bool:
    """Validar formato de ID de estudiante con reglas mejoradas"""
    if not student_id:
        return False

    student_id = str(student_id).strip()

    # Verificar que solo contenga dígitos
    if not student_id.isdigit():
        return False

    # Verificar longitud según configuración
    min_length = getattr(settings, 'MIN_STUDENT_ID_LENGTH', 6)
    max_length = getattr(settings, 'MAX_STUDENT_ID_LENGTH', 20)

    return min_length <= len(student_id) <= max_length


def validate_email(email: str) -> bool:
    """Validar formato de correo electrónico"""
    import re

    if not email or not isinstance(email, str):
        return False

    # Patrón básico de validación de email
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email.strip()) is not None


def validate_image_file(file_content: bytes, filename: str) -> Tuple[bool, str]:
    """Validar archivo de imagen con verificaciones mejoradas"""
    try:
        # Verificar tamaño del archivo
        if len(file_content) == 0:
            return False, "Archivo vacío"

        if len(file_content) > settings.MAX_FILE_SIZE:
            return False, f"Archivo muy grande (máximo {settings.MAX_FILE_SIZE // (1024 * 1024)}MB)"

        # Verificar extensión
        if filename:
            extension = filename.lower().split('.')[-1]
            if extension not in settings.ALLOWED_EXTENSIONS:
                return False, f"Extensión no permitida. Usar: {', '.join(settings.ALLOWED_EXTENSIONS)}"

        # Verificar que sea una imagen válida
        try:
            # Convertir bytes a numpy array
            nparr = np.frombuffer(file_content, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if img is None:
                return False, "No se pudo decodificar la imagen"

            # Verificar dimensiones mínimas
            height, width = img.shape[:2]
            min_size = getattr(settings, 'MIN_IMAGE_SIZE', (100, 100))

            if width < min_size[0] or height < min_size[1]:
                return False, f"Imagen muy pequeña (mínimo {min_size[0]}x{min_size[1]})"

            return True, "Imagen válida"

        except Exception as e:
            return False, f"Error procesando imagen: {str(e)}"

    except Exception as e:
        return False, f"Error validando archivo: {str(e)}"


def image_to_base64(image_path: str) -> Optional[str]:
    """Convertir imagen a base64 con manejo mejorado de errores"""
    try:
        if not os.path.exists(image_path):
            logging.error(f"Archivo no encontrado: {image_path}")
            return None

        # Verificar que sea un archivo de imagen válido
        try:
            img = cv2.imread(image_path)
            if img is None:
                logging.error(f"No se pudo leer la imagen: {image_path}")
                return None
        except Exception as e:
            logging.error(f"Error verificando imagen {image_path}: {e}")
            return None

        with open(image_path, "rb") as img_file:
            encoded = base64.b64encode(img_file.read()).decode('utf-8')
            logging.debug(f"Imagen convertida a base64: {image_path}")
            return encoded

    except Exception as e:
        logging.error(f"Error convirtiendo imagen a base64: {e}")
        return None


def base64_to_image(base64_string: str, output_path: str) -> bool:
    """Convertir base64 a imagen con validaciones"""
    try:
        if not base64_string:
            logging.error("String base64 vacío")
            return False

        # Crear directorio si no existe
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        # Decodificar y guardar
        img_data = base64.b64decode(base64_string)

        # Validar que sea una imagen válida
        is_valid, message = validate_image_file(img_data, output_path)
        if not is_valid:
            logging.error(f"Imagen base64 inválida: {message}")
            return False

        with open(output_path, 'wb') as img_file:
            img_file.write(img_data)

        logging.debug(f"Imagen base64 guardada: {output_path}")
        return True

    except Exception as e:
        logging.error(f"Error convirtiendo base64 a imagen: {e}")
        return False


def save_uploaded_file(file_content: bytes, filename: str) -> str:
    """Guardar archivo subido con validaciones y organización mejorada"""
    try:
        # Validar archivo
        is_valid, message = validate_image_file(file_content, filename)
        if not is_valid:
            raise ValueError(f"Archivo inválido: {message}")

        # Crear directorios
        create_directories()

        # Generar nombre único para evitar colisiones
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name, ext = os.path.splitext(filename) if filename else ("image", ".jpg")
        unique_filename = f"{timestamp}_{generate_unique_pk()}_{name}{ext}"

        # Construir ruta completa
        file_path = os.path.join(settings.UPLOAD_DIR, unique_filename)

        # Guardar archivo
        with open(file_path, "wb") as buffer:
            buffer.write(file_content)

        logging.info(f"Archivo guardado: {file_path}")
        return file_path

    except Exception as e:
        logging.error(f"Error guardando archivo: {e}")
        raise


def cleanup_temp_files(max_age_hours: int = 24):
    """Limpiar archivos temporales antiguos"""
    try:
        temp_dirs = [settings.UPLOAD_DIR, 'temp']
        current_time = datetime.now()
        cleaned_count = 0

        for temp_dir in temp_dirs:
            if not os.path.exists(temp_dir):
                continue

            for filename in os.listdir(temp_dir):
                if filename.startswith('temp_') or filename.startswith('test_'):
                    file_path = os.path.join(temp_dir, filename)
                    try:
                        # Verificar edad del archivo
                        file_time = datetime.fromtimestamp(os.path.getctime(file_path))
                        age_hours = (current_time - file_time).total_seconds() / 3600

                        if age_hours > max_age_hours:
                            os.remove(file_path)
                            cleaned_count += 1
                            logging.debug(f"Archivo temporal eliminado: {file_path}")

                    except Exception as e:
                        logging.warning(f"Error eliminando archivo temporal {file_path}: {e}")

        if cleaned_count > 0:
            logging.info(f"Limpieza completada: {cleaned_count} archivos temporales eliminados")

    except Exception as e:
        logging.error(f"Error en limpieza de archivos temporales: {e}")


def get_file_info(file_path: str) -> dict:
    """Obtener información detallada de un archivo"""
    info = {
        "exists": False,
        "size": 0,
        "modified": None,
        "is_image": False,
        "dimensions": None,
        "format": None
    }

    try:
        if not os.path.exists(file_path):
            return info

        info["exists"] = True
        info["size"] = os.path.getsize(file_path)
        info["modified"] = datetime.fromtimestamp(os.path.getmtime(file_path)).isoformat()

        # Verificar si es imagen
        try:
            img = cv2.imread(file_path)
            if img is not None:
                info["is_image"] = True
                info["dimensions"] = {"width": img.shape[1], "height": img.shape[0]}

                # Detectar formato por extensión
                ext = os.path.splitext(file_path)[1].lower()
                format_map = {'.jpg': 'JPEG', '.jpeg': 'JPEG', '.png': 'PNG',
                              '.bmp': 'BMP', '.gif': 'GIF', '.tiff': 'TIFF'}
                info["format"] = format_map.get(ext, "UNKNOWN")
        except:
            pass

    except Exception as e:
        logging.error(f"Error obteniendo info del archivo {file_path}: {e}")

    return info


def sanitize_filename(filename: str) -> str:
    """Sanitizar nombre de archivo para prevenir problemas de seguridad"""
    import re

    if not filename:
        return "unnamed_file"

    # Remover caracteres peligrosos
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)

    # Limitar longitud
    if len(filename) > 100:
        name, ext = os.path.splitext(filename)
        filename = name[:95] + ext

    # Prevenir nombres reservados en Windows
    reserved_names = ['CON', 'PRN', 'AUX', 'NUL'] + [f'COM{i}' for i in range(1, 10)] + [f'LPT{i}' for i in
                                                                                         range(1, 10)]
    if filename.upper() in reserved_names:
        filename = f"file_{filename}"

    return filename


def format_file_size(size_bytes: int) -> str:
    """Formatear tamaño de archivo en formato legible"""
    if size_bytes == 0:
        return "0 B"

    size_names = ["B", "KB", "MB", "GB", "TB"]
    size_index = 0
    size = float(size_bytes)

    while size >= 1024.0 and size_index < len(size_names) - 1:
        size /= 1024.0
        size_index += 1

    return f"{size:.1f} {size_names[size_index]}"


def create_backup_filename(prefix: str = "backup", extension: str = "json") -> str:
    """Crear nombre de archivo de backup único"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{timestamp}.{extension}"


def verify_system_requirements() -> dict:
    """Verificar que el sistema tenga todos los requisitos"""
    requirements = {
        "opencv": False,
        "numpy": False,
        "mysql_connector": False,
        "sklearn": False,
        "dlib": False,
        "directories": False,
        "config": False
    }

    # Verificar importaciones
    try:
        import cv2
        requirements["opencv"] = True
    except ImportError:
        pass

    try:
        import numpy
        requirements["numpy"] = True
    except ImportError:
        pass

    try:
        import mysql.connector
        requirements["mysql_connector"] = True
    except ImportError:
        pass

    try:
        from sklearn.preprocessing import StandardScaler
        requirements["sklearn"] = True
    except ImportError:
        pass

    try:
        import dlib
        requirements["dlib"] = True
    except ImportError:
        pass

    # Verificar directorios
    try:
        create_directories()
        requirements["directories"] = True
    except:
        pass

    # Verificar configuración
    try:
        from config import settings
        requirements["config"] = True
    except:
        pass

    return requirements


def get_system_stats() -> dict:
    """Obtener estadísticas del sistema de archivos"""
    stats = {
        "directories": {},
        "total_files": 0,
        "total_size": 0,
        "disk_usage": {}
    }

    try:
        # Estadísticas por directorio
        directories = [settings.UPLOAD_DIR, settings.MODELS_DIR, settings.BACKUP_DIR, settings.JSON_BACKUP_DIR]

        for directory in directories:
            if os.path.exists(directory):
                files = os.listdir(directory)
                total_size = sum(os.path.getsize(os.path.join(directory, f))
                                 for f in files if os.path.isfile(os.path.join(directory, f)))

                stats["directories"][directory] = {
                    "files": len(files),
                    "size": total_size,
                    "size_formatted": format_file_size(total_size)
                }

                stats["total_files"] += len(files)
                stats["total_size"] += total_size

        stats["total_size_formatted"] = format_file_size(stats["total_size"])

        # Uso de disco
        try:
            import shutil
            total, used, free = shutil.disk_usage('.')
            stats["disk_usage"] = {
                "total": format_file_size(total),
                "used": format_file_size(used),
                "free": format_file_size(free),
                "usage_percent": round((used / total) * 100, 1)
            }
        except:
            pass

    except Exception as e:
        logging.error(f"Error obteniendo estadísticas del sistema: {e}")

    return stats


def log_function_call(func_name: str, args: dict = None, execution_time: float = None):
    """Loggear llamadas a funciones importantes para debugging"""
    if settings.DEBUG:
        message = f"Función: {func_name}"
        if args:
            # Sanitizar argumentos sensibles
            safe_args = args.copy()
            sensitive_keys = ['password', 'token', 'key', 'secret']
            for key in safe_args:
                if any(sensitive in key.lower() for sensitive in sensitive_keys):
                    safe_args[key] = "***"
            message += f" | Args: {safe_args}"
        if execution_time:
            message += f" | Tiempo: {execution_time:.3f}s"

        logging.debug(message)


def create_test_image(width: int = 200, height: int = 200, output_path: str = None) -> str:
    """Crear imagen de prueba para testing"""
    try:
        # Crear imagen sintética con un "rostro" básico
        img = np.zeros((height, width, 3), dtype=np.uint8)
        img.fill(128)  # Fondo gris

        # "Rostro" básico centrado
        center_x, center_y = width // 2, height // 2
        face_radius = min(width, height) // 3

        # Cara
        cv2.circle(img, (center_x, center_y), face_radius, (200, 200, 200), -1)

        # Ojos
        eye_offset = face_radius // 3
        eye_radius = face_radius // 8
        cv2.circle(img, (center_x - eye_offset, center_y - eye_offset), eye_radius, (0, 0, 0), -1)
        cv2.circle(img, (center_x + eye_offset, center_y - eye_offset), eye_radius, (0, 0, 0), -1)

        # Boca
        mouth_width = face_radius // 2
        mouth_height = face_radius // 4
        cv2.ellipse(img, (center_x, center_y + eye_offset),
                    (mouth_width, mouth_height), 0, 0, 180, (0, 0, 0), 2)

        # Guardar imagen
        if not output_path:
            os.makedirs('test_images', exist_ok=True)
            output_path = f'test_images/test_face_{datetime.now().strftime("%Y%m%d_%H%M%S")}.jpg'

        cv2.imwrite(output_path, img)
        logging.info(f"Imagen de prueba creada: {output_path}")
        return output_path

    except Exception as e:
        logging.error(f"Error creando imagen de prueba: {e}")
        raise


# Funciones de compatibilidad hacia atrás
def save_uploaded_file_legacy(file: bytes, filename: str) -> str:
    """Función legacy para compatibilidad"""
    return save_uploaded_file(file, filename)