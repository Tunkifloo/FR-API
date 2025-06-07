import mysql.connector
from mysql.connector import Error as MySQLError
from urllib.parse import urlparse
import asyncio

# URL completa de Railway (de tus variables)
RAILWAY_URL = "mysql://root:MxxNXhHaMcnyldErVBfzAoOKxlyVhblM@shuttle.proxy.rlwy.net:26047/railway"


def parse_mysql_url(url):
    """Parsear URL de MySQL"""
    parsed = urlparse(url)
    return {
        'host': parsed.hostname,
        'port': parsed.port,
        'user': parsed.username,
        'password': parsed.password,
        'database': parsed.path[1:]  # Remover el '/' inicial
    }


def test_connection():
    """Probar conexión usando URL de Railway"""
    print("🔗 CONECTANDO A RAILWAY USANDO URL COMPLETA")
    print("=" * 50)
    print(f"🌐 URL: {RAILWAY_URL}")

    try:
        # Parsear URL
        config = parse_mysql_url(RAILWAY_URL)
        print(f"🔌 Host: {config['host']}:{config['port']}")
        print(f"👤 Usuario: {config['user']}")
        print(f"📂 Base de datos: {config['database']}")
        print("🔐 Password: [HIDDEN]")

        print("\n🔄 Intentando conectar...")

        # Intentar conexión
        connection = mysql.connector.connect(
            host=config['host'],
            port=config['port'],
            user=config['user'],
            password=config['password'],
            database=config['database'],
            connection_timeout=15,
            autocommit=True
        )

        if connection.is_connected():
            print("✅ ¡CONEXIÓN EXITOSA!")

            cursor = connection.cursor()

            # Información del servidor
            cursor.execute("SELECT VERSION()")
            version = cursor.fetchone()[0]
            print(f"📊 MySQL Version: {version}")

            # Verificar base de datos actual
            cursor.execute("SELECT DATABASE()")
            current_db = cursor.fetchone()[0]
            print(f"📂 Base de datos actual: {current_db}")

            # Listar tablas
            cursor.execute("SHOW TABLES")
            tables = [row[0] for row in cursor.fetchall()]
            print(f"📋 Tablas existentes: {tables}")

            # Si no hay tablas, crear estructura
            if not tables:
                print("\n🔧 No hay tablas, creando estructura...")
                create_tables(cursor)
            else:
                print(f"\n📊 Base de datos ya tiene {len(tables)} tablas")
                for table in tables:
                    cursor.execute(f"SELECT COUNT(*) FROM {table}")
                    count = cursor.fetchone()[0]
                    print(f"   - {table}: {count} registros")

            cursor.close()
            connection.close()
            print("\n🎉 ¡Todo listo! Tu base de datos está funcionando.")
            return True

    except MySQLError as e:
        print(f"❌ Error MySQL: {e}")
        print(f"🔍 Error Code: {e.errno}")
        if e.errno == 1045:
            print("🔐 Error de autenticación - verificar credenciales")
        elif e.errno == 2003:
            print("🌐 Error de conexión - verificar host y puerto")
        return False

    except Exception as e:
        print(f"❌ Error general: {e}")
        return False


def create_tables(cursor):
    """Crear tablas básicas"""
    try:
        # Tabla PERSONAS
        persons_sql = """
                      CREATE TABLE IF NOT EXISTS PERSONAS \
                      ( \
                          ID \
                          INT \
                          AUTO_INCREMENT \
                          PRIMARY \
                          KEY, \
                          Nombre \
                          VARCHAR \
                      ( \
                          100 \
                      ) NOT NULL,
                          Apellidos VARCHAR \
                      ( \
                          100 \
                      ) NOT NULL,
                          Correo VARCHAR \
                      ( \
                          255 \
                      ) UNIQUE NOT NULL,
                          ID_Estudiante VARCHAR \
                      ( \
                          20 \
                      ) UNIQUE,
                          Foto LONGTEXT,
                          PK VARCHAR \
                      ( \
                          50 \
                      ) UNIQUE NOT NULL,
                          fecha_registro TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                          activo BOOLEAN DEFAULT TRUE,
                          INDEX idx_correo \
                      ( \
                          Correo \
                      ),
                          INDEX idx_id_estudiante \
                      ( \
                          ID_Estudiante \
                      ),
                          INDEX idx_pk \
                      ( \
                          PK \
                      )
                          ) ENGINE=InnoDB CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci \
                      """

        cursor.execute(persons_sql)
        print("✅ Tabla PERSONAS creada")

        # Tabla CARACTERISTICAS_FACIALES
        features_sql = """
                       CREATE TABLE IF NOT EXISTS CARACTERISTICAS_FACIALES \
                       ( \
                           ID \
                           INT \
                           AUTO_INCREMENT \
                           PRIMARY \
                           KEY, \
                           persona_id \
                           INT \
                           NOT \
                           NULL, \
                           caracteristicas_json \
                           LONGTEXT \
                           NOT \
                           NULL, \
                           umbral_similitud \
                           DECIMAL \
                       ( \
                           3, \
                           2 \
                       ) DEFAULT 0.70,
                           metodo VARCHAR \
                       ( \
                           50 \
                       ) DEFAULT 'opencv_tradicional',
                           version_algoritmo VARCHAR \
                       ( \
                           10 \
                       ) DEFAULT '1.0',
                           fecha_extraccion TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                           fecha_actualizacion TIMESTAMP NULL ON UPDATE CURRENT_TIMESTAMP,
                           activo BOOLEAN DEFAULT TRUE,
                           FOREIGN KEY \
                       ( \
                           persona_id \
                       ) REFERENCES PERSONAS \
                       ( \
                           ID \
                       ) \
                                                              ON DELETE CASCADE,
                           INDEX idx_persona_id \
                       ( \
                           persona_id \
                       ),
                           INDEX idx_metodo \
                       ( \
                           metodo \
                       )
                           ) ENGINE=InnoDB CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci \
                       """

        cursor.execute(features_sql)
        print("✅ Tabla CARACTERISTICAS_FACIALES creada")

        print("🎯 Estructura de base de datos completada")

    except Exception as e:
        print(f"❌ Error creando tablas: {e}")


async def test_with_fastapi_style():
    """Probar usando el mismo estilo que FastAPI"""
    print("\n" + "=" * 50)
    print("🧪 PROBANDO ESTILO FASTAPI")

    try:
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))

        # Configurar variables de entorno temporalmente
        os.environ['MYSQL_HOST'] = 'shuttle.proxy.rlwy.net'
        os.environ['MYSQL_PORT'] = '26047'
        os.environ['MYSQL_USER'] = 'root'
        os.environ['MYSQL_PASSWORD'] = 'MxxNXhHaMcnyldErVBfzAoOKxlyVhblM'
        os.environ['MYSQL_DATABASE'] = 'railway'

        # Intentar importar y usar el módulo de conexión
        from app.database.connection import get_db_connection, init_database

        print("🔗 Probando get_db_connection()...")
        connection = await get_db_connection()

        if connection:
            print("✅ get_db_connection() funciona!")
            connection.close()

            print("🔧 Probando init_database()...")
            success = await init_database()
            if success:
                print("✅ init_database() funciona!")
                return True
            else:
                print("❌ init_database() falló")
                return False
        else:
            print("❌ get_db_connection() falló")
            return False

    except Exception as e:
        print(f"❌ Error en prueba FastAPI: {e}")
        return False


def main():
    """Función principal"""
    print("🚀 INICIALIZADOR DE RAILWAY DATABASE")
    print("📡 Usando credenciales oficiales de Railway")

    # Paso 1: Probar conexión directa
    success = test_connection()

    if success:
        # Paso 2: Probar con módulos de FastAPI
        asyncio.run(test_with_fastapi_style())

        print("\n🎉 ¡ÉXITO COMPLETO!")
        print("✅ Tu base de datos Railway está lista")
        print("✅ FastAPI puede conectarse correctamente")
        print("\n📝 Actualiza tu .env.production.local con:")
        print("MYSQL_HOST=shuttle.proxy.rlwy.net")
        print("MYSQL_PORT=26047")
        print("MYSQL_PASSWORD=MxxNXhHaMcnyldErVBfzAoOKxlyVhblM")
        print("MYSQL_DATABASE=railway")
    else:
        print("\n❌ Error en conexión")
        print("🔍 Verifica las credenciales en Railway")


if __name__ == "__main__":
    main()