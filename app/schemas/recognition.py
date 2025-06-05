from pydantic import BaseModel
from typing import Optional

from app.schemas.person import PersonResponse


class RecognitionResult(BaseModel):
    similarity: float
    correlation: float
    distance: float
    distance_similarity: float
    is_match: bool
    threshold: float
    faces_detected: int
    features_compared: int

class RecognitionResponse(BaseModel):
    person: PersonResponse
    recognition_result: RecognitionResult

class IdentificationResult(BaseModel):
    best_match: Optional[dict]
    confidence: float
    total_comparisons: int
    faces_detected: int

class IdentificationResponse(BaseModel):
    identification_result: IdentificationResult
    all_matches: list