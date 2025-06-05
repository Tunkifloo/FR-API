from datetime import datetime
from typing import Optional, List
import json
import logging

logger = logging.getLogger(__name__)


class PersonModel:
    """Modelo para manejar datos de personas"""

    @staticmethod
    async def insert_person(person_data: dict, photo_path: str, features: list) -> Optional[int]:
        """Insertar nueva persona en la base de datos"""
        from app.database.connection import get_db_connection
        from app.core.utils import image_to_base64

        connection = None
        try:
            connection = await get_db_connection()
            if not connection:
                return None

            cursor = connection.cursor()

            # Convertir foto a base64
            photo_base64 = image_to_base64(photo_path)
            if not photo_base64:
                logger.error("Error procesando la foto")
                return None

            # Insertar persona
            query_person = """
                           INSERT INTO PERSONAS (Nombre, Apellidos, Correo, ID_Estudiante, Foto, PK)
                           VALUES (%s, %s, %s, %s, %s, %s) \
                           """

            cursor.execute(query_person, (
                person_data['nombre'],
                person_data['apellidos'],
                person_data['correo'],
                person_data.get('id_estudiante'),
                photo_base64,
                person_data['pk']
            ))

            person_id = cursor.lastrowid

            # Insertar características faciales
            features_json = json.dumps(features)
            query_features = """
                             INSERT INTO CARACTERISTICAS_FACIALES (persona_id, caracteristicas_json)
                             VALUES (%s, %s) \
                             """

            cursor.execute(query_features, (person_id, features_json))

            connection.commit()
            cursor.close()
            connection.close()

            logger.info(f"Persona insertada con ID: {person_id}")
            return person_id

        except Exception as e:
            logger.error(f"Error insertando persona: {e}")
            if connection and connection.is_connected():
                connection.rollback()
            return None
        finally:
            if connection and connection.is_connected():
                connection.close()

    @staticmethod
    async def get_person_by_email(email: str) -> Optional[dict]:
        """Buscar persona por correo electrónico"""
        from app.database.connection import get_db_connection

        connection = None
        try:
            connection = await get_db_connection()
            if not connection:
                return None

            cursor = connection.cursor()

            query = """
                    SELECT p.*, c.caracteristicas_json, c.umbral_similitud, c.metodo
                    FROM PERSONAS p
                             LEFT JOIN CARACTERISTICAS_FACIALES c ON p.ID = c.persona_id
                    WHERE p.Correo = %s \
                      AND p.activo = TRUE \
                    """

            cursor.execute(query, (email.lower(),))
            result = cursor.fetchone()

            cursor.close()
            connection.close()

            if result:
                return {
                    'id': result[0],
                    'nombre': result[1],
                    'apellidos': result[2],
                    'correo': result[3],
                    'id_estudiante': result[4],
                    'foto': result[5],
                    'pk': result[6],
                    'fecha_registro': result[7],
                    'activo': result[8],
                    'caracteristicas': json.loads(result[9]) if result[9] else None,
                    'umbral': float(result[10]) if result[10] else 0.75,
                    'metodo': result[11] if result[11] else 'opencv_tradicional'
                }
            return None

        except Exception as e:
            logger.error(f"Error buscando persona por email: {e}")
            return None
        finally:
            if connection and connection.is_connected():
                connection.close()

    @staticmethod
    async def get_person_by_student_id(student_id: str) -> Optional[dict]:
        """Buscar persona por ID de estudiante"""
        from app.database.connection import get_db_connection

        connection = None
        try:
            connection = await get_db_connection()
            if not connection:
                return None

            cursor = connection.cursor()

            query = """
                    SELECT p.*, c.caracteristicas_json, c.umbral_similitud, c.metodo
                    FROM PERSONAS p
                             LEFT JOIN CARACTERISTICAS_FACIALES c ON p.ID = c.persona_id
                    WHERE p.ID_Estudiante = %s \
                      AND p.activo = TRUE \
                    """

            cursor.execute(query, (student_id,))
            result = cursor.fetchone()

            cursor.close()
            connection.close()

            if result:
                return {
                    'id': result[0],
                    'nombre': result[1],
                    'apellidos': result[2],
                    'correo': result[3],
                    'id_estudiante': result[4],
                    'foto': result[5],
                    'pk': result[6],
                    'fecha_registro': result[7],
                    'activo': result[8],
                    'caracteristicas': json.loads(result[9]) if result[9] else None,
                    'umbral': float(result[10]) if result[10] else 0.75,
                    'metodo': result[11] if result[11] else 'opencv_tradicional'
                }
            return None

        except Exception as e:
            logger.error(f"Error buscando persona por ID estudiante: {e}")
            return None
        finally:
            if connection and connection.is_connected():
                connection.close()

    @staticmethod
    async def list_all_persons() -> List[dict]:
        """Listar todas las personas registradas"""
        from app.database.connection import get_db_connection

        connection = None
        try:
            connection = await get_db_connection()
            if not connection:
                return []

            cursor = connection.cursor()

            query = """
                    SELECT ID, Nombre, Apellidos, Correo, ID_Estudiante, PK, fecha_registro
                    FROM PERSONAS
                    WHERE activo = TRUE
                    ORDER BY fecha_registro DESC \
                    """

            cursor.execute(query)
            results = cursor.fetchall()

            cursor.close()
            connection.close()

            persons = []
            for result in results:
                persons.append({
                    'id': result[0],
                    'nombre': result[1],
                    'apellidos': result[2],
                    'correo': result[3],
                    'id_estudiante': result[4],
                    'pk': result[5],
                    'fecha_registro': result[6]
                })

            return persons

        except Exception as e:
            logger.error(f"Error listando personas: {e}")
            return []
        finally:
            if connection and connection.is_connected():
                connection.close()

    @staticmethod
    async def check_email_exists(email: str) -> bool:
        """Verificar si un correo ya existe"""
        from app.database.connection import get_db_connection

        connection = None
        try:
            connection = await get_db_connection()
            if not connection:
                return False

            cursor = connection.cursor()

            query = "SELECT COUNT(*) FROM PERSONAS WHERE Correo = %s AND activo = TRUE"
            cursor.execute(query, (email.lower(),))

            count = cursor.fetchone()[0]

            cursor.close()
            connection.close()

            return count > 0

        except Exception as e:
            logger.error(f"Error verificando email: {e}")
            return False
        finally:
            if connection and connection.is_connected():
                connection.close()

    @staticmethod
    async def check_student_id_exists(student_id: str) -> bool:
        """Verificar si un ID de estudiante ya existe"""
        from app.database.connection import get_db_connection

        connection = None
        try:
            connection = await get_db_connection()
            if not connection:
                return False

            cursor = connection.cursor()

            query = "SELECT COUNT(*) FROM PERSONAS WHERE ID_Estudiante = %s AND activo = TRUE"
            cursor.execute(query, (student_id,))

            count = cursor.fetchone()[0]

            cursor.close()
            connection.close()

            return count > 0

        except Exception as e:
            logger.error(f"Error verificando ID estudiante: {e}")
            return False
        finally:
            if connection and connection.is_connected():
                connection.close()