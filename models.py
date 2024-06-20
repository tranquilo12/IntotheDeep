# Constants for Model names
import enum


class ModelNames(enum.Enum):
    GPT_4o = "gpt-4o"
    GPT_4_0125_PREVIEW = "gpt-4-turbo"
    GPT_3_5_TURBO = "gpt-3.5-turbo-0125"

    @classmethod
    def oai_to_list(cls):
        return [model_name.value for model_name in cls if "gpt" in model_name.value]

    @classmethod
    def claude_to_list(cls):
        return [model_name.value for model_name in cls if "claude" in model_name.value]

    @classmethod
    def all_to_list(cls):
        return [model_name.value for model_name in cls]
