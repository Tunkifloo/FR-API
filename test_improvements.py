import requests
import json
import os
import sys
from datetime import datetime
import time

# Configuración
API_URL = "http://localhost:8000"
TEST_EMAIL = f"test.mejoras.{int(time.time())}@test.com"


def check_api_status():
    """Verificar que la API esté funcionando"""
    print("🔍 Verificando estado de la API...")
    try:
        response = requests.get(f"{API_URL}/health")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ API en línea - Estado: {data['status']}")
            return True
        else:
            print(f"❌ API no responde correctamente: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error conectando con la API: {e}")
        print("   Asegúrate de que el servidor esté ejecutándose: uvicorn main:app --reload")
        return False


def check_system_config():
    """Verificar configuración del sistema"""
    print("\n📋 Configuración del sistema:")
    try:
        response = requests.get(f"{API_URL}/api/recognition/stats")
        if response.status_code == 200:
            config = response.json()['configuration']
            print(
                f"  - Procesamiento mejorado: {'✅' if config['enhanced_processing'] else '❌'} {config['enhanced_processing']}")
            print(
                f"  - Detección DNN: {'✅' if config.get('use_dnn_detection', False) else '❌'} {config.get('use_dnn_detection', False)}")
            print(
                f"  - Face Embeddings: {'✅' if config.get('use_face_embeddings', False) else '❌'} {config.get('use_face_embeddings', False)}")
            print(
                f"  - Sistema de votación: {'✅' if config.get('use_voting_system', False) else '❌'} {config.get('use_voting_system', False)}")
            print(f"  - Umbral por defecto: {config['default_threshold']}")
            return config
        else:
            print(f"❌ Error obteniendo configuración: {response.status_code}")
            return None
    except Exception as e:
        print(f"❌ Error: {e}")
        return None


def test_registration():
    """Probar registro con sistema mejorado"""
    print("\n🧪 Probando registro con sistema mejorado...")

    # Verificar que existe imagen de prueba
    test_images = ["test_images/test_person.jpg", "test_images/person1.jpg", "uploads/test_face.jpg"]
    test_image = None

    for img in test_images:
        if os.path.exists(img):
            test_image = img
            break

    if not test_image:
        print("❌ No se encontró imagen de prueba")
        print("   Crea una imagen en: test_images/test_person.jpg")
        return None

    print(f"📷 Usando imagen: {test_image}")

    # Datos de prueba
    person_data = {
        "nombre": "Usuario",
        "apellidos": "Prueba Mejoras",
        "correo": TEST_EMAIL,
        "id_estudiante": "999999999"
    }

    # Enviar solicitud
    print("📤 Enviando solicitud de registro...")
    start_time = time.time()

    try:
        with open(test_image, 'rb') as f:
            files = {'foto': (os.path.basename(test_image), f, 'image/jpeg')}
            response = requests.post(
                f"{API_URL}/api/persons/register",
                data=person_data,
                files=files
            )

        elapsed_time = time.time() - start_time

        if response.status_code == 200:
            result = response.json()
            print(f"✅ Registro exitoso en {elapsed_time:.2f} segundos")
            print(f"   - ID: {result['person_id']}")
            print(f"   - Características extraídas: {result['features_count']}")
            print(f"   - Rostros detectados: {result['faces_detected']}")
            print(f"   - Método: {result['processing_method']}")

            if 'system_info' in result:
                si = result['system_info']
                print("   - Sistema:")
                print(f"     • DNN: {si.get('dnn_detection', False)}")
                print(f"     • Embeddings: {si.get('face_embeddings', False)}")
                print(f"     • Voting: {si.get('voting_system', False)}")

            return result
        else:
            print(f"❌ Error en registro: {response.status_code}")
            print(f"   Detalles: {response.json()}")
            return None

    except Exception as e:
        print(f"❌ Error: {e}")
        return None


def test_recognition(email):
    """Probar reconocimiento con sistema de votación"""
    print(f"\n🔍 Probando reconocimiento para: {email}")

    # Buscar imagen de prueba diferente
    test_images = ["test_images/test_person_2.jpg", "test_images/person2.jpg", "test_images/test_person.jpg"]
    test_image = None

    for img in test_images:
        if os.path.exists(img):
            test_image = img
            break

    if not test_image:
        print("❌ No se encontró imagen de prueba para reconocimiento")
        return None

    print(f"📷 Usando imagen: {test_image}")

    # Enviar solicitud
    print("🔄 Procesando reconocimiento...")
    start_time = time.time()

    try:
        with open(test_image, 'rb') as f:
            files = {'test_image': (os.path.basename(test_image), f, 'image/jpeg')}
            data = {'email': email}
            response = requests.post(
                f"{API_URL}/api/recognition/compare/email",
                data=data,
                files=files
            )

        elapsed_time = time.time() - start_time

        if response.status_code == 200:
            result = response.json()
            recog = result['recognition_result']

            print(f"✅ Reconocimiento completado en {elapsed_time:.2f} segundos")
            print(f"   - Similitud: {recog.get('similarity', 0):.3f}")
            print(f"   - ¿Coincide?: {'✅ SÍ' if recog['is_match'] else '❌ NO'}")
            print(f"   - Umbral: {recog['threshold']}")
            print(f"   - Método: {recog.get('comparison_method', 'standard')}")

            # Si hay sistema de votación
            if 'votes' in recog:
                print("\n   📊 Sistema de Votación:")
                votes = recog['votes']
                scores = recog.get('detailed_scores', {})

                for method, vote in votes.items():
                    score = scores.get(method, 0)
                    print(f"     • {method}: {'✅' if vote else '❌'} ({score:.3f})")

                print(f"   - Votos positivos: {recog['positive_votes']}/{recog['total_votes']}")
                print(f"   - Confianza por votación: {recog.get('voting_confidence', 0):.3f}")

            return result
        else:
            print(f"❌ Error en reconocimiento: {response.status_code}")
            print(f"   Detalles: {response.json()}")
            return None

    except Exception as e:
        print(f"❌ Error: {e}")
        return None


def main():
    """Función principal de pruebas"""
    print("=" * 60)
    print("🚀 PRUEBA DEL SISTEMA DE RECONOCIMIENTO FACIAL MEJORADO")
    print("=" * 60)

    # 1. Verificar API
    if not check_api_status():
        sys.exit(1)

    # 2. Verificar configuración
    config = check_system_config()
    if not config:
        sys.exit(1)

    # 3. Advertencias
    if not config.get('enhanced_processing'):
        print("\n⚠️  ADVERTENCIA: El procesamiento mejorado no está habilitado")
        print("   Actualiza USE_ENHANCED_PROCESSING=true en .env")

    # 4. Probar registro
    reg_result = test_registration()
    if not reg_result:
        print("\n❌ No se pudo completar el registro")
        sys.exit(1)

    # 5. Esperar un momento
    print("\n⏳ Esperando 2 segundos antes del reconocimiento...")
    time.sleep(2)

    # 6. Probar reconocimiento
    test_recognition(TEST_EMAIL)

    print("\n" + "=" * 60)
    print("✅ Pruebas completadas")
    print("=" * 60)


if __name__ == "__main__":
    main()