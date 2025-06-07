-- Script de inicialización de base de datos para Sistema de Reconocimiento Facial Mejorado
-- Versión: 2.1.0
-- Fecha: 2025-01-01

CREATE DATABASE IF NOT EXISTS reconocimiento_facial
CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

USE reconocimiento_facial;

-- ===== TABLA DE PERSONAS =====
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

    -- Índices para optimización
    INDEX idx_correo (Correo),
    INDEX idx_id_estudiante (ID_Estudiante),
    INDEX idx_pk (PK),
    INDEX idx_activo (activo),
    INDEX idx_fecha_registro (fecha_registro),
    INDEX idx_nombres (Nombre, Apellidos)
) ENGINE=InnoDB CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

-- ===== TABLA DE CARACTERÍSTICAS FACIALES MEJORADA =====
CREATE TABLE IF NOT EXISTS CARACTERISTICAS_FACIALES (
    ID INT AUTO_INCREMENT PRIMARY KEY,
    persona_id INT NOT NULL,
    caracteristicas_json LONGTEXT NOT NULL,

    -- Configuración de umbrales y métodos
    umbral_similitud DECIMAL(3,2) DEFAULT 0.70,
    metodo VARCHAR(50) DEFAULT 'opencv_tradicional',
    version_algoritmo VARCHAR(10) DEFAULT '1.0',

    -- Timestamps para auditoría
    fecha_extraccion TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    fecha_actualizacion TIMESTAMP NULL ON UPDATE CURRENT_TIMESTAMP,

    -- Estado y metadatos
    activo BOOLEAN DEFAULT TRUE,
    calidad_imagen DECIMAL(3,2) DEFAULT NULL,
    num_rostros_detectados INT DEFAULT 1,
    confianza_extraccion DECIMAL(3,2) DEFAULT NULL,

    -- Claves foráneas y restricciones
    FOREIGN KEY (persona_id) REFERENCES PERSONAS(ID) ON DELETE CASCADE,

    -- Índices para optimización de consultas
    INDEX idx_persona_id (persona_id),
    INDEX idx_metodo (metodo),
    INDEX idx_version (version_algoritmo),
    INDEX idx_activo (activo),
    INDEX idx_fecha_extraccion (fecha_extraccion),
    INDEX idx_umbral (umbral_similitud),
    INDEX idx_metodo_version (metodo, version_algoritmo),

    -- Índice compuesto para búsquedas frecuentes
    INDEX idx_persona_activo (persona_id, activo)
) ENGINE=InnoDB CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

-- ===== TABLA DE HISTORIAL DE RECONOCIMIENTOS (OPCIONAL) =====
CREATE TABLE IF NOT EXISTS HISTORIAL_RECONOCIMIENTOS (
    ID INT AUTO_INCREMENT PRIMARY KEY,
    persona_id INT,
    similitud_obtenida DECIMAL(5,4),
    umbral_usado DECIMAL(3,2),
    metodo_comparacion VARCHAR(50),
    resultado_exitoso BOOLEAN,
    tiempo_procesamiento_ms INT,
    fecha_reconocimiento TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    ip_origen VARCHAR(45),
    user_agent TEXT,

    FOREIGN KEY (persona_id) REFERENCES PERSONAS(ID) ON DELETE SET NULL,
    INDEX idx_persona_historial (persona_id),
    INDEX idx_fecha_reconocimiento (fecha_reconocimiento),
    INDEX idx_resultado (resultado_exitoso),
    INDEX idx_metodo_hist (metodo_comparacion)
) ENGINE=InnoDB CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

-- ===== TABLA DE CONFIGURACIÓN DEL SISTEMA =====
CREATE TABLE IF NOT EXISTS CONFIGURACION_SISTEMA (
    ID INT AUTO_INCREMENT PRIMARY KEY,
    clave VARCHAR(100) UNIQUE NOT NULL,
    valor TEXT,
    tipo VARCHAR(20) DEFAULT 'string',
    descripcion TEXT,
    fecha_modificacion TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    usuario_modificacion VARCHAR(100),

    INDEX idx_clave (clave)
) ENGINE=InnoDB CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

-- ===== INSERTAR CONFIGURACIONES POR DEFECTO =====
INSERT IGNORE INTO CONFIGURACION_SISTEMA (clave, valor, tipo, descripcion) VALUES
('sistema_version', '2.1.0', 'string', 'Versión del sistema de reconocimiento facial'),
('umbral_por_defecto', '0.70', 'decimal', 'Umbral de similitud por defecto'),
('metodo_por_defecto', 'enhanced', 'string', 'Método de procesamiento por defecto'),
('max_intentos_reconocimiento', '3', 'integer', 'Máximo número de intentos de reconocimiento'),
('tiempo_cache_caracteristicas', '3600', 'integer', 'Tiempo en segundos para cache de características'),
('habilitar_historial', 'true', 'boolean', 'Habilitar registro de historial de reconocimientos'),
('limpiar_archivos_temp', 'true', 'boolean', 'Habilitar limpieza automática de archivos temporales'),
('backup_automatico', 'true', 'boolean', 'Habilitar backup automático de datos');

-- ===== VISTAS ÚTILES =====

-- Vista de personas con estadísticas de reconocimiento
CREATE OR REPLACE VIEW vista_personas_stats AS
SELECT
    p.ID,
    p.Nombre,
    p.Apellidos,
    p.Correo,
    p.ID_Estudiante,
    p.fecha_registro,
    cf.metodo,
    cf.version_algoritmo,
    cf.umbral_similitud,
    cf.fecha_extraccion,
    cf.calidad_imagen,
    COALESCE(hr.total_reconocimientos, 0) as total_reconocimientos,
    COALESCE(hr.reconocimientos_exitosos, 0) as reconocimientos_exitosos,
    CASE
        WHEN hr.total_reconocimientos > 0
        THEN ROUND((hr.reconocimientos_exitosos / hr.total_reconocimientos) * 100, 2)
        ELSE NULL
    END as tasa_exito_porcentaje
FROM PERSONAS p
LEFT JOIN CARACTERISTICAS_FACIALES cf ON p.ID = cf.persona_id AND cf.activo = TRUE
LEFT JOIN (
    SELECT
        persona_id,
        COUNT(*) as total_reconocimientos,
        SUM(CASE WHEN resultado_exitoso = TRUE THEN 1 ELSE 0 END) as reconocimientos_exitosos
    FROM HISTORIAL_RECONOCIMIENTOS
    GROUP BY persona_id
) hr ON p.ID = hr.persona_id
WHERE p.activo = TRUE;

-- Vista de estadísticas por método de procesamiento
CREATE OR REPLACE VIEW vista_stats_metodos AS
SELECT
    metodo,
    version_algoritmo,
    COUNT(*) as total_personas,
    AVG(umbral_similitud) as umbral_promedio,
    MIN(fecha_extraccion) as primera_extraccion,
    MAX(fecha_extraccion) as ultima_extraccion
FROM CARACTERISTICAS_FACIALES cf
JOIN PERSONAS p ON cf.persona_id = p.ID
WHERE cf.activo = TRUE AND p.activo = TRUE
GROUP BY metodo, version_algoritmo
ORDER BY total_personas DESC;

-- ===== PROCEDIMIENTOS ALMACENADOS =====

-- Procedimiento para limpiar datos antiguos
DELIMITER //
CREATE PROCEDURE IF NOT EXISTS LimpiarDatosAntiguos(IN dias_historial INT)
BEGIN
    DECLARE EXIT HANDLER FOR SQLEXCEPTION
    BEGIN
        ROLLBACK;
        RESIGNAL;
    END;

    START TRANSACTION;

    -- Limpiar historial de reconocimientos antiguo
    DELETE FROM HISTORIAL_RECONOCIMIENTOS
    WHERE fecha_reconocimiento < DATE_SUB(NOW(), INTERVAL dias_historial DAY);

    -- Limpiar características inactivas antiguas
    DELETE FROM CARACTERISTICAS_FACIALES
    WHERE activo = FALSE
    AND fecha_actualizacion < DATE_SUB(NOW(), INTERVAL dias_historial DAY);

    COMMIT;
END //
DELIMITER ;

-- Procedimiento para obtener estadísticas completas
DELIMITER //
CREATE PROCEDURE IF NOT EXISTS ObtenerEstadisticasCompletas()
BEGIN
    SELECT
        'Personas Activas' as metrica,
        COUNT(*) as valor
    FROM PERSONAS WHERE activo = TRUE

    UNION ALL

    SELECT
        'Total Características',
        COUNT(*)
    FROM CARACTERISTICAS_FACIALES WHERE activo = TRUE

    UNION ALL

    SELECT
        CONCAT('Método: ', metodo),
        COUNT(*)
    FROM CARACTERISTICAS_FACIALES cf
    JOIN PERSONAS p ON cf.persona_id = p.ID
    WHERE cf.activo = TRUE AND p.activo = TRUE
    GROUP BY metodo

    UNION ALL

    SELECT
        'Reconocimientos Hoy',
        COUNT(*)
    FROM HISTORIAL_RECONOCIMIENTOS
    WHERE DATE(fecha_reconocimiento) = CURDATE();
END //
DELIMITER ;

-- ===== TRIGGERS PARA AUDITORÍA =====

-- Trigger para actualizar fecha de modificación en características
DELIMITER //
CREATE TRIGGER IF NOT EXISTS tr_caracteristicas_update
    BEFORE UPDATE ON CARACTERISTICAS_FACIALES
    FOR EACH ROW
BEGIN
    SET NEW.fecha_actualizacion = CURRENT_TIMESTAMP;
END //
DELIMITER ;

-- ===== COMENTARIOS EN TABLAS =====
ALTER TABLE PERSONAS COMMENT = 'Tabla principal de personas registradas en el sistema';
ALTER TABLE CARACTERISTICAS_FACIALES COMMENT = 'Características faciales extraídas con diferentes métodos y versiones';
ALTER TABLE HISTORIAL_RECONOCIMIENTOS COMMENT = 'Historial de intentos de reconocimiento facial';
ALTER TABLE CONFIGURACION_SISTEMA COMMENT = 'Configuraciones del sistema de reconocimiento facial';

-- ===== VERIFICACIÓN DE INSTALACIÓN =====
SELECT
    'Base de datos inicializada correctamente' as status,
    VERSION() as mysql_version,
    DATABASE() as database_name,
    NOW() as fecha_inicializacion;