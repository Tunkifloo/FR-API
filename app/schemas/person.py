from pydantic import BaseModel, EmailStr, validator
from typing import Optional
from datetime import datetime


class PersonCreate(BaseModel):
    nombre: str
    apellidos: str
    correo: EmailStr
    id_estudiante: Optional[str] = None

    @validator('nombre', 'apellidos')
    def validate_names(cls, v):
        if not v.strip():
            raise ValueError('Nombre y apellidos no pueden estar vacíos')
        return v.strip()

    @validator('id_estudiante')
    def validate_student_id(cls, v):
        if v is not None:
            v = v.strip()
            if not v.isdigit() or len(v) < 6 or len(v) > 20:
                raise ValueError('ID de estudiante debe tener entre 6-20 dígitos')
        return v


class PersonResponse(BaseModel):
    id: int
    nombre: str
    apellidos: str
    correo: str
    id_estudiante: Optional[str]
    pk: str
    fecha_registro: datetime
    activo: bool


class PersonWithFeatures(PersonResponse):
    caracteristicas: Optional[list]
    umbral: float
    metodo: str