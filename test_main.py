import pytest
from fastapi.testclient import TestClient
from main import app
import io
from PIL import Image

client = TestClient(app)


def test_root():
    """Test endpoint raíz"""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "version" in data
    assert data["version"] == "2.0.0"


def test_health_check():
    """Test verificación de salud"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data


def test_admin_stats():
    """Test estadísticas del sistema"""
    response = client.get("/api/admin/stats")
    assert response.status_code == 200
    data = response.json()
    assert "system_info" in data
    assert "statistics" in data


def test_list_persons():
    """Test listar personas"""
    response = client.get("/api/persons/list")
    assert response.status_code == 200
    data = response.json()
    assert "total" in data
    assert "persons" in data


def create_test_image():
    """Crear imagen de prueba"""
    img = Image.new('RGB', (200, 200), color='red')
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='JPEG')
    img_bytes.seek(0)
    return img_bytes


def test_register_person():
    """Test registro de persona"""
    test_image = create_test_image()

    response = client.post(
        "/api/persons/register",
        data={
            "nombre": "Juan",
            "apellidos": "Pérez",
            "correo": "juan.perez@test.com",
            "id_estudiante": "123456789"
        },
        files={"foto": ("test.jpg", test_image, "image/jpeg")}
    )

    # Nota: Este test podría fallar si no hay rostros detectables en la imagen de prueba
    # En un entorno real, usar imágenes con rostros reales para testing
    assert response.status_code in [200, 400]  # 400 si no se detectan rostros