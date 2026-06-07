from pydantic import BaseModel, Field

DEFAULT_CHARACTERS = ["HAMLET", "ROMEO", "JULIET", "MACBETH", "OPHELIA"]


class GenerateRequest(BaseModel):
    prompt: str
    character: str | None = None
    temperature: float = 0.5
    top_k: int = 40
    max_new_tokens: int = Field(default=200, ge=1, le=500)


class RoundtableRequest(BaseModel):
    prompt: str
    characters: list[str] = Field(default_factory=lambda: list(DEFAULT_CHARACTERS), min_length=2, max_length=6)
    turns: int = Field(default=2, ge=1, le=4)
    temperature: float = 0.7
    top_k: int = 40
    max_new_tokens: int = Field(default=120, ge=1, le=300)


class TopTokensRequest(BaseModel):
    prompt: str
    character: str | None = None
    top_k: int = Field(default=10, ge=1, le=50)


class ArenaRequest(BaseModel):
    prompt: str
    characters: list[str] = Field(..., min_length=1, max_length=4)
    temperature: float = 0.5
    top_k: int = 40
    max_new_tokens: int = Field(default=150, ge=1, le=500)


class TokenProb(BaseModel):
    token: str
    prob: float


class TopTokensResponse(BaseModel):
    tokens: list[TokenProb]


class ArenaResult(BaseModel):
    character: str
    text: str
    perplexity: float


class ArenaResponse(BaseModel):
    results: list[ArenaResult]
