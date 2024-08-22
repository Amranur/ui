from pydantic import BaseModel

class APIDocumentationCreate(BaseModel):
    title: str
    section: str
    content: str
    example_code: str = None

class APIDocumentationUpdate(BaseModel):
    title: str
    section: str
    content: str
    example_code: str = None

class APIDocumentation(BaseModel):
    id: int
    title: str
    section: str
    content: str
    example_code: str = None

    class Config:
        orm_mode = True
