from datetime import datetime
from typing import Optional, List
import json
import logging

logger = logging.getLogger(__name__)


class PersonModel:
    """Modelo mejorado para manejar datos de personas"""

    @staticmethod
    async def insert_person(person_data: dict, photo_path: str, features: list, method: str = "enhanced") -> Optional[
        int]:
        """Insertar nueva persona en la base de datos con soporte para método mejorado"""
        from app.database.connection import get_db_connection
        from app.core.utils import image_to_base64
        from config import settings

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

            # Determinar umbral basado en método y configuración
            if method == "enhanced":
                threshold = settings.DEFAULT_SIMILARITY_THRESHOLD
                version = "2.0"
            else:
                threshold = 0.60  # Mantener umbral anterior para compatibilidad
                version = "1.0"

            # Insertar características faciales
            features_json = json.dumps(features)

            # Verificar si la tabla tiene las nuevas columnas
            try:
                query_features = """
                                 INSERT INTO CARACTERISTICAS_FACIALES
                                 (persona_id, caracteristicas_json, umbral_similitud, metodo, version_algoritmo, \
                                  fecha_extraccion)
                                 VALUES (%s, %s, %s, %s, %s, NOW()) \
                                 """
                cursor.execute(query_features, (person_id, features_json, threshold, method, version))
            except Exception as e:
                # Fallback para esquema antiguo
                logger.warning(f"Usando esquema antiguo para características: {e}")
                query_features = """
                                 INSERT INTO CARACTERISTICAS_FACIALES (persona_id, caracteristicas_json, umbral_similitud)
                                 VALUES (%s, %s, %s) \
                                 """
                cursor.execute(query_features, (person_id, features_json, threshold))

            connection.commit()
            cursor.close()
            connection.close()

            logger.info(f"Persona insertada con ID: {person_id}, método: {method}")
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
        """Buscar persona por correo electrónico con soporte para nuevos campos"""
        from app.database.connection import get_db_connection

        connection = None
        try:
            connection = await get_db_connection()
            if not connection:
                return None

            cursor = connection.cursor()

            # Intentar consulta con nuevos campos primero
            try:
                query = """
                        SELECT p.*, \
                               c.caracteristicas_json, \
                               c.umbral_similitud, \
                               c.metodo, \
                               c.version_algoritmo, \
                               c.fecha_extraccion
                        FROM PERSONAS p
                                 LEFT JOIN CARACTERISTICAS_FACIALES c ON p.ID = c.persona_id
                        WHERE p.Correo = %s \
                          AND p.activo = TRUE \
                        """
                cursor.execute(query, (email.lower(),))
                result = cursor.fetchone()

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
                        'umbral': float(result[10]) if result[10] else 0.70,
                        'metodo': result[11] if result[11] else 'opencv_tradicional',
                        'version': result[12] if result[12] else '1.0',
                        'fecha_extraccion': result[13] if result[13] else None
                    }

            except Exception as e:
                # Fallback para esquema antiguo
                logger.warning(f"Usando esquema antiguo: {e}")
                query = """
                        SELECT p.*, c.caracteristicas_json, c.umbral_similitud, c.metodo
                        FROM PERSONAS p
                                 LEFT JOIN CARACTERISTICAS_FACIALES c ON p.ID = c.persona_id
                        WHERE p.Correo = %s \
                          AND p.activo = TRUE \
                        """
                cursor.execute(query, (email.lower(),))
                result = cursor.fetchone()

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
                        'umbral': float(result[10]) if result[10] else 0.70,
                        'metodo': result[11] if result[11] else 'opencv_tradicional',
                        'version': '1.0',
                        'fecha_extraccion': None
                    }

            cursor.close()
            connection.close()
            return None

        except Exception as e:
            logger.error(f"Error buscando persona por email: {e}")
            return None
        finally:
            if connection and connection.is_connected():
                connection.close()

    @staticmethod
    async def get_person_by_student_id(student_id: str) -> Optional[dict]:
        """Buscar persona por ID de estudiante con soporte para nuevos campos"""
        from app.database.connection import get_db_connection

        connection = None
        try:
            connection = await get_db_connection()
            if not connection:
                return None

            cursor = connection.cursor()

            # Intentar consulta con nuevos campos primero
            try:
                query = """
                        SELECT p.*, \
                               c.caracteristicas_json, \
                               c.umbral_similitud, \
                               c.metodo, \
                               c.version_algoritmo, \
                               c.fecha_extraccion
                        FROM PERSONAS p
                                 LEFT JOIN CARACTERISTICAS_FACIALES c ON p.ID = c.persona_id
                        WHERE p.ID_Estudiante = %s \
                          AND p.activo = TRUE \
                        """
                cursor.execute(query, (student_id,))
                result = cursor.fetchone()

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
                        'umbral': float(result[10]) if result[10] else 0.70,
                        'metodo': result[11] if result[11] else 'opencv_tradicional',
                        'version': result[12] if result[12] else '1.0',
                        'fecha_extraccion': result[13] if result[13] else None
                    }

            except Exception as e:
                # Fallback para esquema antiguo
                logger.warning(f"Usando esquema antiguo: {e}")
                query = """
                        SELECT p.*, c.caracteristicas_json, c.umbral_similitud, c.metodo
                        FROM PERSONAS p
                                 LEFT JOIN CARACTERISTICAS_FACIALES c ON p.ID = c.persona_id
                        WHERE p.ID_Estudiante = %s \
                          AND p.activo = TRUE \
                        """
                cursor.execute(query, (student_id,))
                result = cursor.fetchone()

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
                        'umbral': float(result[10]) if result[10] else 0.70,
                        'metodo': result[11] if result[11] else 'opencv_tradicional',
                        'version': '1.0',
                        'fecha_extraccion': None
                    }

            cursor.close()
            connection.close()
            return None

        except Exception as e:
            logger.error(f"Error buscando persona por ID estudiante: {e}")
            return None
        finally:
            if connection and connection.is_connected():
                connection.close()

    @staticmethod
    async def update_person_features(person_id: int, new_features: list, method: str = "enhanced") -> bool:
        """Actualizar características de una persona existente"""
        from app.database.connection import get_db_connection
        from config import settings

        connection = None
        try:
            connection = await get_db_connection()
            if not connection:
                return False

            cursor = connection.cursor()

            features_json = json.dumps(new_features)
            threshold = settings.DEFAULT_SIMILARITY_THRESHOLD if method == "enhanced" else 0.60
            version = "2.0" if method == "enhanced" else "1.0"

            # Intentar actualización con nuevos campos
            try:
                query = """
                        UPDATE CARACTERISTICAS_FACIALES
                        SET caracteristicas_json = %s, \
                            metodo               = %s, \
                            version_algoritmo    = %s,
                            umbral_similitud     = %s, \
                            fecha_actualizacion  = NOW()
                        WHERE persona_id = %s \
                        """
                cursor.execute(query, (features_json, method, version, threshold, person_id))
            except Exception as e:
                # Fallback para esquema antiguo
                logger.warning(f"Usando actualización de esquema antiguo: {e}")
                query = """
                        UPDATE CARACTERISTICAS_FACIALES
                        SET caracteristicas_json = %s, \
                            umbral_similitud     = %s
                        WHERE persona_id = %s \
                        """
                cursor.execute(query, (features_json, threshold, person_id))

            connection.commit()
            cursor.close()
            connection.close()

            logger.info(f"Características actualizadas para persona ID: {person_id}")
            return True

        except Exception as e:
            logger.error(f"Error actualizando características: {e}")
            return False
        finally:
            if connection and connection.is_connected():
                connection.close()

    @staticmethod
    async def get_persons_for_migration() -> List[dict]:
        """Obtener personas que necesitan migración al sistema mejorado"""
        from app.database.connection import get_db_connection

        connection = None
        try:
            connection = await get_db_connection()
            if not connection:
                return []

            cursor = connection.cursor()

            # Intentar consulta con nuevos campos
            try:
                query = """
                        SELECT p.ID, \
                               p.Nombre, \
                               p.Apellidos, \
                               p.Correo, \
                               p.Foto,
                               COALESCE(c.metodo, 'opencv_tradicional') as metodo,
                               COALESCE(c.version_algoritmo, '1.0')     as version
                        FROM PERSONAS p
                                 LEFT JOIN CARACTERISTICAS_FACIALES c ON p.ID = c.persona_id
                        WHERE p.activo = TRUE
                          AND (c.metodo IS NULL OR c.metodo != 'enhanced' OR c.version_algoritmo != '2.0') \
                        """
                cursor.execute(query)
                results = cursor.fetchall()

            except Exception as e:
                # Fallback para esquema antiguo - migrar todas las personas
                logger.warning(f"Esquema antiguo detectado, migrar todas las personas: {e}")
                query = """
                        SELECT p.ID, \
                               p.Nombre, \
                               p.Apellidos, \
                               p.Correo, \
                               p.Foto,
                               'opencv_tradicional' as metodo, \
                               '1.0'                as version
                        FROM PERSONAS p
                        WHERE p.activo = TRUE \
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
                    'foto': result[4],
                    'metodo_actual': result[5],
                    'version_actual': result[6]
                })

            return persons

        except Exception as e:
            logger.error(f"Error obteniendo personas para migración: {e}")
            return []
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

    @staticmethod
    async def get_system_stats() -> dict:
        """Obtener estadísticas del sistema"""
        from app.database.connection import get_db_connection

        connection = None
        try:
            connection = await get_db_connection()
            if not connection:
                return {}

            cursor = connection.cursor()

            stats = {}

            # Total de personas activas
            cursor.execute("SELECT COUNT(*) FROM PERSONAS WHERE activo = TRUE")
            stats['total_personas'] = cursor.fetchone()[0]

            # Intentar obtener estadísticas por método
            try:
                cursor.execute("""
                               SELECT COALESCE(c.metodo, 'sin_metodo') as metodo, COUNT(*) as cantidad
                               FROM PERSONAS p
                                        LEFT JOIN CARACTERISTICAS_FACIALES c ON p.ID = c.persona_id
                               WHERE p.activo = TRUE
                               GROUP BY COALESCE(c.metodo, 'sin_metodo')
                               """)
                method_stats = cursor.fetchall()
                stats['por_metodo'] = {row[0]: row[1] for row in method_stats}

                cursor.execute("""
                               SELECT COALESCE(c.version_algoritmo, 'sin_version') as version, COUNT(*) as cantidad
                               FROM PERSONAS p
                                        LEFT JOIN CARACTERISTICAS_FACIALES c ON p.ID = c.persona_id
                               WHERE p.activo = TRUE
                               GROUP BY COALESCE(c.version_algoritmo, 'sin_version')
                               """)
                version_stats = cursor.fetchall()
                stats['por_version'] = {row[0]: row[1] for row in version_stats}

            except Exception as e:
                logger.warning(f"No se pudieron obtener estadísticas avanzadas: {e}")
                stats['por_metodo'] = {'traditional': stats['total_personas']}
                stats['por_version'] = {'1.0': stats['total_personas']}

            cursor.close()
            connection.close()

            return stats

        except Exception as e:
            logger.error(f"Error obteniendo estadísticas: {e}")
            return {}
        finally:
            if connection and connection.is_connected():
                connection.close()