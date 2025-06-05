-- Script de inicialización de base de datos
CREATE DATABASE IF NOT EXISTS reconocimiento_facial
CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

USE reconocimiento_facial;

-- Tabla de personas
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
    INDEX idx_pk (PK)
) ENGINE=InnoDB CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

-- Tabla de características faciales
CREATE TABLE IF NOT EXISTS CARACTERISTICAS_FACIALES (
    ID INT AUTO_INCREMENT PRIMARY KEY,
    persona_id INT NOT NULL,
    caracteristicas_json LONGTEXT NOT NULL,
    umbral_similitud DECIMAL(3,2) DEFAULT 0.75,
    metodo VARCHAR(50) DEFAULT 'opencv_tradicional',
    fecha_creacion TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (persona_id) REFERENCES PERSONAS(ID) ON DELETE CASCADE,
    INDEX idx_persona_id (persona_id)
) ENGINE=InnoDB CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;