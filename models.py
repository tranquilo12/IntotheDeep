import enum


class ModelNames(enum.Enum):
    AZURE_GPT_4O = "azure/PDFS-GPT-4o"
    AZURE_GPT_35_TURBO = "azure/PDFS-GPT-35-turbo-0301"

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
        return [model_name.value for model_name in cls if "claude" in model_name.value]

    @classmethod
    def azure_to_list(cls):
        return [
            model_name.value
            for model_name in cls
            if model_name.value.startswith("azure")
        ]

    @classmethod
    def all_to_list(cls):
        return [model_name.value for model_name in cls]

    @classmethod
    def is_azure_model(cls, model_name):
        return model_name.startswith("azure/")

    @classmethod
    def get_azure_deployment_name(cls, model_name):
        if cls.is_azure_model(model_name):
            return model_name.split("/")[1]
        return None
