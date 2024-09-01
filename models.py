import enum


class ModelTypes(enum.Enum):
    OPENAI = "OPENAI"
    ANTHROPIC = "ANTHROPIC"
    LOCAL = "LMSTUDIO"
    UNKNOWN = "UNKNOWN"


class ModelNames(enum.Enum):
    AZURE_GPT_4O = "azure/PDFS-GPT-4o"
    AZURE_GPT_35_TURBO = "azure/PDFS-GPT-35-turbo-0301"
    CLAUDE_3_SONNET = "claude-3-sonnet-20240229"
    CLAUDE_3_5_SONNET = "claude-3-5-sonnet-20240620"
    DEEPSEEK_CODER_V2 = "DeepSeek-Coder-V2-Lite-Instruct-GGUF"

    @classmethod
    def oai_to_list(cls):
        return [
            model_name.value
            for model_name in cls
            if model_name.value.startswith("gpt")
            and not model_name.value.startswith("azure")
        ]

    @classmethod
    def claude_to_list(cls):
        return [
            model_name.value
            for model_name in cls
            if model_name.value.startswith("claude")
        ]

    @classmethod
    def azure_to_list(cls):
        return [
            model_name.value
            for model_name in cls
            if model_name.value.startswith("azure")
        ]

    @classmethod
    def local_to_list(cls):
        return [model_name.value for model_name in cls if "GGUF" in model_name.value]

    @classmethod
    def all_to_list(cls):
        return [model_name.value for model_name in cls]

    @classmethod
    def get_model_type(cls, model_name):
        if model_name.startswith("azure"):
            return ModelTypes.OPENAI.value
        elif model_name.startswith("claude"):
            return ModelTypes.ANTHROPIC.value
        elif "GGUF" in model_name:
            return ModelTypes.LOCAL.value
        else:
            return ModelTypes.UNKNOWN.value

    @classmethod
    def get_api_base(cls, model_name):
        model_type = cls.get_model_type(model_name)
        if model_type == "azure":
            return "AZURE_API_BASE"
        elif model_type == "claude":
            return "ANTHROPIC_API_BASE"
        elif model_type == "local":
            return "http://localhost:1234/v1"
        else:
            return None
