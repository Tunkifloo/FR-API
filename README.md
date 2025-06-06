# Sistema de Reconocimiento Facial - FastAPI

## 🚀 Migración Completa de Jupyter Notebook a FastAPI

Este proyecto migra completamente el sistema de reconocimiento facial desde Jupyter Notebook a una API robusta desarrollada con FastAPI, manteniendo toda la funcionalidad original mientras añade nuevas capacidades empresariales.

## 📋 Características Principales

### ✅ **API RESTful Completa**
- Endpoints organizados por funcionalidad
- Documentación automática con Swagger/OpenAPI
- Validación de datos con Pydantic
- Manejo robusto de errores
- Logging estructurado

### ✅ **Arquitectura Modular**
```
app/
├── core/           # Utilidades centrales y procesamiento facial
├── database/       # Modelos y conexión a MySQL
├── endpoints/      # Rutas de la API organizadas
└── schemas/        # Validación de datos con Pydantic
```

### ✅ **Funcionalidades Migradas**
- ✅ Registro completo de personas con validación
- ✅ Reconocimiento facial por email/ID estudiante
- ✅ Identificación contra toda la base de datos
- ✅ Exportación/importación de datos
- ✅ Administración del sistema
- ✅ Procesamiento avanzado de imágenes
- ✅ Extracción de 1024 características faciales

## 🛠️ Tecnologías Utilizadas

- **FastAPI 0.104.1** - Framework web moderno y rápido
- **Uvicorn** - Servidor ASGI de alta performance
- **Pydantic** - Validación de datos y serialización
- **OpenCV 4.8.1** - Procesamiento de imágenes
- **MySQL 8.0** - Base de datos relacional
- **Docker & Docker Compose** - Containerización
- **pytest** - Testing automatizado

## 🚀 Instalación y Configuración

### **Opción 1: Instalación Local**

1. **Clonar repositorio**
```bash
git clone <repository-url>
cd facial-recognition-fastapi
```

2. **Crear entorno virtual**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows
```

3. **Instalar dependencias**
```bash
pip install -r requirements.txt
```

4. **Configurar variables de entorno**
```bash
cp .env.example .env
# Editar .env con tus configuraciones
```

5. **Inicializar base de datos MySQL**
```bash
# Asegúrate de que MySQL esté ejecutándose
mysql -u root -p < init.sql
```

6. **Ejecutar aplicación**
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### **Opción 2: Docker Compose (Recomendado)**

```bash
# Clonar y ejecutar
git clone <repository-url>
cd facial-recognition-fastapi
docker-compose up -d
```

Esto levantará automáticamente:
- **API FastAPI** en `http://localhost:8000`
- **MySQL** en puerto `3306`
- **phpMyAdmin** en `http://localhost:8080`

## 📖 Uso de la API

### **Documentación Interactiva**
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

### **Endpoints Principales**

#### **1. Registrar Nueva Persona**
```bash
curl -X POST "http://localhost:8000/api/persons/register" \
  -H "Content-Type: multipart/form-data" \
  -F "nombre=Juan" \
  -F "apellidos=Pérez" \
  -F "correo=juan.perez@email.com" \
  -F "id_estudiante=123456789" \
  -F "foto=@/path/to/photo.jpg"
```

#### **2. Reconocimiento por Email**
```bash
curl -X POST "http://localhost:8000/api/recognition/compare/email" \
  -H "Content-Type: multipart/form-data" \
  -F "email=juan.perez@email.com" \
  -F "test_image=@/path/to/test_image.jpg"
```

#### **3. Identificar Persona**
```bash
curl -X POST "http://localhost:8000/api/recognition/identify" \
  -H "Content-Type: multipart/form-data" \
  -F "test_image=@/path/to/unknown_person.jpg"
```

#### **4. Listar Personas**
```bash
curl -X GET "http://localhost:8000/api/persons/list"
```

#### **5. Estadísticas del Sistema**
```bash
curl -X GET "http://localhost:8000/api/admin/stats"
```

#### **6. Exportar Datos**
```bash
curl -X GET "http://localhost:8000/api/data/export/all"
```

## 🔧 Configuración Avanzada

### **Variables de Entorno (.env)**
```env
# MySQL
MYSQL_HOST=localhost
MYSQL_PORT=3306
MYSQL_USER=root
MYSQL_PASSWORD=@dmin
MYSQL_DATABASE=reconocimiento_facial

# Aplicación
DEBUG=True
SECRET_KEY=your-secret-key-here
DEFAULT_SIMILARITY_THRESHOLD=0.75

# Directorios
UPLOAD_DIR=uploads
MODELS_DIR=models
BACKUP_DIR=exports
JSON_BACKUP_DIR=json_backup
```

### **Configuración de Umbrales**
El sistema permite ajustar la sensibilidad del reconocimiento:
- **0.60-0.70**: Muy permisivo (más falsos positivos)
- **0.75**: Balanceado (ACTUAL 0.60)
- **0.80-0.90**: Muy estricto (más falsos negativos)

## 📊 Monitoreo y Logging

### **Logs del Sistema**
```bash
# Ver logs en tiempo real
tail -f facial_recognition.log

# Logs con Docker
docker-compose logs -f facial-recognition-api
```

### **Health Checks**
```bash
# Estado general
curl http://localhost:8000/health

# Estado administrativo
curl http://localhost:8000/api/admin/health

# Verificar integridad
curl http://localhost:8000/api/admin/integrity
```

## 🧪 Testing

```bash
# Ejecutar tests
pytest test_main.py -v

# Coverage
pytest --cov=app test_main.py
```

## 📁 Estructura de Archivos

```
facial-recognition-fastapi/
├── main.py                 # Aplicación principal FastAPI
├── config.py              # Configuraciones del sistema
├── requirements.txt       # Dependencias Python
├── Dockerfile             # Imagen Docker
├── docker-compose.yml     # Orquestación de servicios
├── init.sql              # Script inicialización MySQL
├── .env.example          # Variables de entorno ejemplo
├── test_main.py          # Tests automatizados
├── app/
│   ├── __init__.py
│   ├── core/             # Núcleo del sistema
│   │   ├── __init__.py
│   │   ├── utils.py      # Utilidades generales
│   │   ├── facial_processing.py    # Procesamiento facial
│   │   └── feature_extraction.py  # Extracción características
│   ├── database/         # Base de datos
│   │   ├── __init__.py
│   │   ├── connection.py # Conexión MySQL
│   │   └── models.py     # Modelos de datos
│   ├── endpoints/        # Rutas API
│   │   ├── __init__.py
│   │   ├── admin.py      # Administración
│   │   ├── persons.py    # Gestión personas
│   │   ├── recognition.py # Reconocimiento
│   │   └── data_management.py # Gestión datos
│   └── schemas/          # Validación datos
│       ├── __init__.py
│       ├── person.py     # Esquemas persona
│       └── recognition.py # Esquemas reconocimiento
├── uploads/              # Imágenes subidas
├── models/               # Modelos pickle (compatibilidad)
├── exports/              # Exportaciones JSON
└── json_backup/          # Respaldos automáticos
```

## 🔄 Migración desde Jupyter

### **Diferencias Principales**

| Jupyter Notebook | FastAPI |
|------------------|---------|
| Interfaz interactiva manual | API REST automatizada |
| Diálogos tkinter | Formularios HTTP |
| Ejecución secuencial | Peticiones concurrentes |
| Archivos locales | Uploads HTTP |
| Visualización matplotlib | Respuestas JSON |

### **Ventajas de la Migración**

✅ **Escalabilidad**: Múltiples usuarios concurrentes
✅ **Integración**: Fácil integración con otros sistemas
✅ **Automatización**: Procesos automatizados
✅ **Monitoreo**: Logs y métricas estructuradas
✅ **Deployment**: Fácil despliegue en producción
✅ **Testing**: Tests automatizados
✅ **Documentación**: API docs automática

## 🔐 Seguridad

- ✅ Validación de tipos de archivo (solo imágenes)
- ✅ Límites de tamaño de archivos
- ✅ Sanitización de datos de entrada
- ✅ Manejo seguro de archivos temporales
- ✅ Logging de operaciones sensibles
- ⚠️ **Producción**: Implementar autenticación JWT
- ⚠️ **Producción**: HTTPS obligatorio
- ⚠️ **Producción**: Rate limiting

## 🚀 Deployment en Producción

### **Docker Swarm**
```bash
docker stack deploy -c docker-compose.yml facial-recognition
```

### **Kubernetes**
```bash
# Ejemplo básico
kubectl apply -f k8s-deployment.yaml
```

### **Variables de Producción**
```env
DEBUG=False
MYSQL_HOST=production-mysql-host
SECRET_KEY=super-secure-random-key
```

## 📈 Performance

### **Métricas Esperadas**
- **Registro**: ~3-5 segundos por persona
- **Reconocimiento**: ~1-2 segundos por imagen
- **Identificación**: ~0.5 segundos por persona en BD
- **Concurrencia**: 10-20 usuarios simultáneos

### **Optimizaciones**
- Cache de características faciales
- Pool de conexiones MySQL
- Procesamiento asíncrono de imágenes
- CDN para archivos estáticos

## 🐛 Troubleshooting

### **Problemas Comunes**

**Error MySQL Connection**
```bash
# Verificar que MySQL esté ejecutándose
docker-compose ps
# Revisar logs
docker-compose logs mysql
```

**Error OpenCV**
```bash
# Instalar dependencias del sistema
apt-get install libgl1-mesa-glx
```

**No se detectan rostros**
- Verificar calidad de imagen (mínimo 200x200px)
- Asegurar buena iluminación
- Rostro frontal y visible

**Error de permisos en directorios**
```bash
chmod -R 755 uploads/ models/ exports/ json_backup/
```

## 🤝 Contribución

1. Fork del repositorio
2. Crear rama feature (`git checkout -b feature/nueva-funcionalidad`)
3. Commit cambios (`git commit -am 'Agregar nueva funcionalidad'`)
4. Push a la rama (`git push origin feature/nueva-funcionalidad`)
5. Crear Pull Request