import enum
import tiktoken
from typing import Union, List


class ModelTypes(enum.Enum):
    OPENAI = "OPENAI"
    ANTHROPIC = "ANTHROPIC"
    LOCAL = "LMSTUDIO"
    UNKNOWN = "UNKNOWN"


class ModelNames(enum.Enum):
    CLAUDE_3_SONNET = "claude-3"
    CLAUDE_3_5_SONNET = "claude-3.5"
    DEEPSEEK_CODER_V2 = "deepseek-coder-v2"
    AZURE_GPT_4O = "azure/PDFS-GPT-4o"
    AZURE_GPT_35_TURBO = "azure/PDFS-GPT-35-turbo-0301"

    @classmethod
    def all_to_list(cls):
        return [model_name.value for model_name in cls]

    @classmethod
    def get_model_type(cls, model_name):
        if "azure" in model_name:
            return ModelTypes.OPENAI.value
        elif "claude" in model_name:
            return ModelTypes.ANTHROPIC.value
        elif "deepseek" in model_name:
            return ModelTypes.LOCAL.value
        else:
            return ModelTypes.UNKNOWN.value

    @classmethod
    def get_api_base(cls):
        return "http://0.0.0.0:4000"

    @classmethod
    def get_encoding_name(cls, model_name: str) -> str:
        model_type = cls.get_model_type(model_name)
        if model_type in [ModelTypes.OPENAI.value, ModelTypes.LOCAL.value]:
            return "gpt-3.5-turbo" if "3.5" in model_name else "gpt-4"
        elif model_type == ModelTypes.ANTHROPIC.value:
            return "gpt-3.5-turbo"  # Using OpenAI's encoding for Anthropic models
        else:
            raise ValueError(f"Unknown model type: {model_name}")

    @classmethod
    def get_encoding(cls, model_name: str) -> tiktoken.Encoding:
        encoding_name = cls.get_encoding_name(model_name)
        return tiktoken.encoding_for_model(encoding_name)

    @classmethod
    def count_tokens(cls, text: str, model_name: str) -> int:
        encoding = cls.get_encoding(model_name)
        return len(encoding.encode(text))


def get_token_count(text: Union[str, List], model_name: str) -> int:
    """
    Parameters:
        text: The list of message objects.
    """
    if isinstance(text, List):
        return sum(ModelNames.count_tokens(t.content.text, model_name) for t in text)
    return ModelNames.count_tokens(text, model_name)
