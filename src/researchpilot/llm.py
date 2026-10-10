from langchain_openai import ChatOpenAI


def create_chat_model(api_key: str, model_name: str, base_url: str | None = None) -> ChatOpenAI:
    options: dict[str, object] = {
        "api_key": api_key,
        "model": model_name,
        "temperature": 0,
    }
    if base_url:
        options["base_url"] = base_url
    return ChatOpenAI(**options)

