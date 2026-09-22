from pydantic import BaseModel


class DecodedResponse(BaseModel):
    decoded: str
