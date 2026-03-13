from pydantic import BaseModel, model_validator, Field


class Node(BaseModel):
    id: int
    file_name: str
    parent_id: int | None
    child_ids: list[int] = []
    description: str | None = None
    keywords: list[str] = []
    page_start: int = Field(ge=1)
    page_end: int = Field(ge=1)

    @model_validator(mode="after")
    def check_page_range(self):
        if self.page_end < self.page_start:
            raise ValueError("page_end must be greater than or equal to page_start")
        return self

    def get_sparse_text(self):
        return f"{self.file_name}, {self.description or 'N/A'}, {', '.join(self.keywords) or 'N/A'}"

    def get_dense_text(self):
        return f"{self.description or 'N/A'}"


class Chunk(BaseModel):
    file_name: str
    page_start: int = Field(ge=1)
    page_end: int = Field(ge=1)
    text: str

    @model_validator(mode="after")
    def check_page_range(self):
        if self.page_end < self.page_start:
            raise ValueError("page_end must be greater than or equal to page_start")
        return self


class DescriptorOutput(BaseModel):
    description: str = Field(
        default="No description available.",
        description="A brief description of the content.",
    )
    keywords: list[str] = Field(
        default_factory=list,
        description="A list of relevant keywords extracted from the content.",
    )
