from pydantic import BaseModel

from .clients import OpenAIClient
from .config import MODEL


def llm_call(
    messages: list[dict], structured_output: None | type[BaseModel] = None
) -> tuple[str, str]:
    with OpenAIClient() as client:
        response = client.client.chat.completions.create(  # type: ignore
            model=MODEL,
            messages=messages,  # type: ignore
            response_format=(
                None
                if structured_output is None
                else {
                    "type": "json_schema",
                    "json_schema": {
                        "name": structured_output.__name__,
                        "schema": structured_output.model_json_schema(),
                    },
                }
            ),  # type: ignore
            extra_body={"reasoning": {"enabled": True}},
        )

    content = getattr(response.choices[0].message, "content", "N/A")
    reasoning = getattr(
        response.choices[0].message,
        "reasoning",
        getattr(response.choices[0].message, "reasoning_details", "N/A"),
    )

    return content, reasoning
