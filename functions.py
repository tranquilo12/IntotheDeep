import os
from pydantic import BaseModel
from typing import List, Optional, Dict, Literal
import json
from models import ModelNames
from enum import Enum


class ParameterType(str, Enum):
    STRING = "string"
    NUMBER = "number"
    BOOLEAN = "boolean"
    OBJECT = "object"
    ARRAY = "array"


class Parameter(BaseModel):
    type: ParameterType
    description: str
    enum: Optional[List[str]] = None


class FunctionParameters(BaseModel):
    type: Literal["object"] = "object"
    properties: Dict[str, Parameter]
    required: List[str]


class Function(BaseModel):
    name: str
    description: str
    parameters: FunctionParameters


class OpenAIFunction(BaseModel):
    name: str
    description: str
    parameters: FunctionParameters


class AnthropicFunction(BaseModel):
    type: Literal["function"] = "function"
    function: Function


class FunctionSet(BaseModel):
    functions: List[Function]

    def to_openai(self) -> List[OpenAIFunction]:
        return [OpenAIFunction(**func.model_dump()) for func in self.functions]

    def to_anthropic(self) -> List[AnthropicFunction]:
        return [AnthropicFunction(function=func) for func in self.functions]


def load_functions(filepath: str) -> FunctionSet:
    with open(filepath, "r") as f:
        data = json.load(f)
    return FunctionSet(functions=[Function(**func) for func in data])


class Payload(BaseModel):
    model: str
    api_key: str
    api_base: str
    messages: List[Dict[str, str]]
    max_tokens: int
    temperature: float = 0.3
    stream: bool = True
    functions: Optional[List[OpenAIFunction]] = None
    function_call: Optional[str] = None
    tools: Optional[List[AnthropicFunction]] = None
    tool_choice: Optional[str] = None

    @classmethod
    def create(
        cls,
        model: str,
        messages: List[Dict[str, str]],
        max_tokens: int,
        functions: FunctionSet,
        stream: bool = True,
        api_base: str = "http://0.0.0.0:4000",
    ):
        model_type = ModelNames.get_model_type(model)
        api_key = os.getenv(f"{model_type.upper()}_API_KEY", "")
        return cls(
            model=model,
            api_key=api_key,
            api_base=api_base,
            messages=messages,
            max_tokens=max_tokens,
            stream=stream,
            functions=functions.to_openai(),
            function_call="auto",
        )
