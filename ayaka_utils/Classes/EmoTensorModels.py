from pydantic import BaseModel
from typing import Literal, Optional, List

################################################################################
## Configuration

EmoTensorVersion = "v0.0.1-EmoTensor Sliced Contextualized"

################################################################################
## EmoTensor File Types

# EmoTensor files can be:
#  * .etur - EmoTensor Unsliced Raw
#  * .etuu - EmoTensor Unsliced Unicode
#  * .etsu - EmoTensor Sliced Unicode
#  * .etsc - EmoTensor Sliced Contextualized

################################################################################
## Pydandtic Models

class EmoTensor1DSlice_CTXD(BaseModel): # Contextualized 1-dimensional (emotion vector: emotion, intensity, valence, arousal) EmoTensor Slice with Context
    # Emotion: Literal["Joy", "Sadness", "Fear", "Anger", "Disgust", "Surprise", "Shame", "Envy", "Embarrassment", "Pride", "Admiration", "Love", "Gratitude", "Awe"] # Custom
    emotion: Literal[
        "Joy",
        "Trust",
        "Fear",
        "Surprise",
        "Sadness",
        "Disgust",
        "Anger",
        "Anticipation",
    ]  # Plutchik
    context: str
    intensity: float
    valence: float
    arousal: float

class EmoTensor1DSlice_uCTXD(BaseModel): # Uncontextualized 1-dimensional (emotion vector: emotion, intensity, valence, arousal) EmoTensor Slice with Context
    # Emotion: Literal["Joy", "Sadness", "Fear", "Anger", "Disgust", "Surprise", "Shame", "Envy", "Embarrassment", "Pride", "Admiration", "Love", "Gratitude", "Awe"] # Custom
    emotion: Literal[
        "Joy",
        "Trust",
        "Fear",
        "Surprise",
        "Sadness",
        "Disgust",
        "Anger",
        "Anticipation",
    ]  # Plutchik
    intensity: float
    valence: float
    arousal: float

class EmoTensor2DSlice_CTXD(BaseModel): # Contextualized 2-dimensional (emotion state matrix: emotion 1, emotion 2,..., emotion 8) EmoTensor with Context
    this_target: str  # Format: "uid=X-uid=Y"
    scratch_context: Optional[str | None] = None
    scratch_synopsis: str
    emotions: List[EmoTensor1DSlice_CTXD]

class EmoTensor2DSlice_uCTXD(BaseModel): # Uncontextualized 2-dimensional (emotion state matrix: emotion 1, emotion 2,..., emotion 8) EmoTensor with Context
    this_target: str  # Format: "uid=X-uid=Y"
    emotions: List[EmoTensor1DSlice_uCTXD]

class EmoTensor3DSlice_CTXD(BaseModel): # Contextualized 3-dimensional (interpersonal emotion state cuboid: person 1 towards person 2, person 1 towards person 3,..., person 1 towards person N) EmoTensor with Context
    emoter_user: str  # Format: "uid=X"
    external_context: str
    targets: List[EmoTensor2DSlice_CTXD]

class EmoTensor3DSlice_uCTXD(BaseModel): # Uncontextualized 3-dimensional (interpersonal emotion state cuboid: person 1 towards person 2, person 1 towards person 3,..., person 1 towards person N) EmoTensor with Context
    emoter_user: str  # Format: "uid=X"
    targets: List[EmoTensor2DSlice_uCTXD]

class EmoTensor4DSlice_CTXD(BaseModel): # Contextualized 4-dimensional (personal emotion state transient: for one moment in time; person 1, person 2,..., person N) EmoTensor with Context
    message: str
    speaker_user: str  # Format: "uid=X"
    emotional_context: str
    emoters: List[EmoTensor3DSlice_CTXD]

    def validate_structure(self):
        """Validate the hierarchical structure is complete"""
        if not self.emoters:
            raise ValueError("EmoTensor4DSlice must have at least one emoter")
        for emoter in self.emoters:
            if not emoter.targets:
                raise ValueError("EmoTensor3DSlice must have at least one target")
            for target in emoter.targets:
                if not target.emotions:
                    raise ValueError("EmoTensor2DSlice must have emotions")

class EmoTensor4DSlice_uCTXD(BaseModel): # Uncontextualized 4-dimensional (personal emotion state transient: for one moment in time; person 1, person 2,..., person N) EmoTensor with Context
    speaker_user: str  # Format: "uid=X"
    emoters: List[EmoTensor3DSlice_uCTXD]

    def validate_structure(self):
        """Validate the hierarchical structure is complete"""
        if not self.emoters:
            raise ValueError("EmoTensor4DSlice must have at least one emoter")
        for emoter in self.emoters:
            if not emoter.targets:
                raise ValueError("EmoTensor3DSlice must have at least one target")
            for target in emoter.targets:
                if not target.emotions:
                    raise ValueError("EmoTensor2DSlice must have emotions")

class EmoTensorFull_CTXD(BaseModel): # Contextualized Full 5-dimensional (morphing interpersonal emotion state) EmoTensor with Context
    version: str = EmoTensorVersion
    order_1_attributes: List[str]
    order_2_emotions: List[str]
    order_3_target: List[str]
    order_4_emoters: List[str]
    order_5_transients: List[str]
    user_prefnames: List[str]
    transients: List[EmoTensor4DSlice_CTXD]

class EmoTensorFull_uCTXD(BaseModel): # Uncontextualized Full 5-dimensional (morphing interpersonal emotion state) EmoTensor with Context
    version: str = EmoTensorVersion
    order_1_attributes: List[str]
    order_2_emotions: List[str]
    order_3_target: List[str]
    order_4_emoters: List[str]
    order_5_transients: List[str]
    transients: List[EmoTensor4DSlice_uCTXD]

class EmoTensorOutput(BaseModel):
    scratch_context: str
    external_context: str
    happiness: float
    sadness: float
    anger: float
    fear: float