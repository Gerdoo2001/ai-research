from pydantic import BaseModel, Field

class PredictionResponse(BaseModel):
    label: str = Field(..., description="Predicted class: NORMAL or PNEUMONIA")
    probability: float = Field(..., description="Sigmoid probability for PNEUMONIA")
    threshold: float = Field(..., description="Decision threshold used")
