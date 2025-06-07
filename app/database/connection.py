import mysql.connector
from mysql.connector import Error as MySQLError, errorcode
import logging
from config import settings
from typing import Optional

logger = logging.getLogger(__name__)

# Configuración de conexión MySQL
MYSQL_CONFIG = {
    'host': settings.MYSQL_HOST,
    'port': settings.MYSQL_PORT,
    'user': settings.MYSQL_USER,
    'password': settings.MYSQL_PASSWORD,
    'database': settings.MYSQL_DATABASE,
    'charset': settings.MYSQL_CHARSET,
    'collation': settings.MYSQL_COLLATION,
    'autocommit': False,  # Para mejor control de transacciones
    'connection_timeout': 10  # Timeout de conexión
}


async def get_db_connection() -> Optional[mysql.connector.MySQLConnection]:
    """Crear conexión a MySQL con manejo mejorado de errores"""
    try:
        connection = mysql.connector.connect(**MYSQL_CONFIG)
        if connection.is_connected():
            return connection
    except MySQLError as e:
        logger.error(f"Error conectando a MySQL: {e}")
        return None


async def init_database() -> bool:
    """Inicializar base de datos y tablas con esquema mejorado"""
    try:
        # Crear base de datos si no existe
        config_without_db = MYSQL_CONFIG.copy()
        del config_without_db['database']

        connection = mysql.connector.connect(**config_without_db)
        cursor = connection.cursor()

        # Crear base de datos
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS {MYSQL_CONFIG['database']} "
                       f"CHARACTER SET {MYSQL_CONFIG['charset']} "
                       f"COLLATE {MYSQL_CONFIG['collation']}")

        cursor.close()
        connection.close()

        # Conectar a la base de datos específica
        connection = await get_db_connection()
        if not connection:
            return False

        cursor = connection.cursor()

        # Crear tabla PERSONAS
        create_persons_table = """
            CREATE TABLE IF NOT EXISTS PERSONAS (
                ID INT AUTO_INCREMENT PRIMARY KEY,
                Nombre VARCHAR(100) NOT NULL,
                Apellidos VARCHAR(100) NOT NULL,
                Correo VARCHAR(255) UNIQUE NOT NULL,
                ID_Estudiante VARCHAR(20) UNIQUE,
                Foto LONGTEXT,
                PK VARCHAR(50) UNIQUE NOT NULL,
                fecha_registro TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                activo BOOLEAN DEFAULT TRUE,
                INDEX idx_correo (Correo),
                INDEX idx_id_estudiante (ID_Estudiante),
                INDEX idx_pk (PK),
                INDEX idx_activo (activo),
                INDEX idx_fecha_registro (fecha_registro)
            ) ENGINE=InnoDB CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci
        """

        cursor.execute(create_persons_table)

        # Crear tabla CARACTERISTICAS_FACIALES con soporte para sistema mejorado
        create_features_table = """
            CREATE TABLE IF NOT EXISTS CARACTERISTICAS_FACIALES (
                ID INT AUTO_INCREMENT PRIMARY KEY,
                persona_id INT NOT NULL,
                caracteristicas_json LONGTEXT NOT NULL,
                umbral_similitud DECIMAL(3,2) DEFAULT 0.70,
                metodo VARCHAR(50) DEFAULT 'opencv_tradicional',
                version_algoritmo VARCHAR(10) DEFAULT '1.0',
                fecha_extraccion TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                fecha_actualizacion TIMESTAMP NULL ON UPDATE CURRENT_TIMESTAMP,
                activo BOOLEAN DEFAULT TRUE,
                FOREIGN KEY (persona_id) REFERENCES PERSONAS(ID) ON DELETE CASCADE,
                INDEX idx_persona_id (persona_id),
                INDEX idx_metodo (metodo),
                INDEX idx_version (version_algoritmo),
                INDEX idx_activo (activo),
                INDEX idx_fecha_extraccion (fecha_extraccion)
            ) ENGINE=InnoDB CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci
        """

        cursor.execute(create_features_table)

        connection.commit()
        cursor.close()
        connection.close()

        logger.info("Base de datos inicializada correctamente")
        return True

    except MySQLError as e:
        logger.error(f"Error inicializando base de datos: {e}")
        return False