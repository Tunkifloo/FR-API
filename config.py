import os
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Configuración MySQL
    MYSQL_HOST: str = "localhost"
    MYSQL_PORT: int = 3306
    MYSQL_USER: str = "root"
    MYSQL_PASSWORD: str = "@dmin"
    MYSQL_DATABASE: str = "reconocimiento_facial"
    MYSQL_CHARSET: str = "utf8mb4"
    MYSQL_COLLATION: str = "utf8mb4_unicode_ci"

    # Configuración de la aplicación
    DEBUG: bool = True
    SECRET_KEY: str = "your-secret-key-here"

    # Configuración de archivos
    UPLOAD_DIR: str = "uploads"
    MODELS_DIR: str = "models"
    BACKUP_DIR: str = "exports"
    JSON_BACKUP_DIR: str = "json_backup"

    # Configuración de reconocimiento
    DEFAULT_SIMILARITY_THRESHOLD: float = 0.75
    FACE_SIZE: tuple = (128, 128)

    class Config:
        env_file = ".env"


settings = Settings()