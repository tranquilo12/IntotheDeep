import enum


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
