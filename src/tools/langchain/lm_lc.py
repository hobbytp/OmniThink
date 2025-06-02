import os
from typing import Optional, Any, Dict, List
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatAlibabaTongyi
from langchain_core.language_models.chat_models import BaseChatModel

# Placeholder for future helper functions if needed

def get_langchain_llm(
    model_name: str,
    temperature: float = 0.7,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    **kwargs: Any
) -> BaseChatModel:
    """
    Factory function to get an instance of a LangChain chat model.

    Supports OpenAI, DeepSeek (via OpenAI-compatible endpoint), and Qwen/Dashscope models.
    API keys and base URLs are resolved with the following precedence:
    1. Explicit function parameters (`api_key`, `base_url`).
    2. Environment variables (model-specific, e.g., OPENAI_API_KEY, DEEPSEEK_API_KEY).
    3. Default values (e.g., default base URL for DeepSeek).

    Args:
        model_name (str): The name of the model to instantiate (e.g., "gpt-4o", "deepseek-chat", "qwen-max").
        temperature (float, optional): The sampling temperature for the model. Defaults to 0.7.
        api_key (Optional[str], optional): The API key for the model provider. Defaults to None.
        base_url (Optional[str], optional): The base URL for the model provider's API. Defaults to None.
        **kwargs (Any): Additional keyword arguments to pass to the model constructor.
                         These can override default arguments or add new ones.

    Returns:
        BaseChatModel: An instance of a LangChain compatible chat model.

    Raises:
        ValueError: If the provided model_name is not supported.
    """
    model_name_lower = model_name.lower()
    llm_args: Dict[str, Any] = {"model_name": model_name, "temperature": temperature}

    if "gpt" in model_name_lower or model_name_lower.startswith("openai-"): # Common OpenAI prefixes/names
        resolved_api_key = api_key or os.getenv("OPENAI_API_KEY")
        resolved_base_url = base_url or os.getenv("OPENAI_BASE_URL")

        if resolved_api_key:
            llm_args["api_key"] = resolved_api_key
        if resolved_base_url:
            llm_args["base_url"] = resolved_base_url

        llm_args.update(kwargs) # Apply/override with explicit kwargs
        return ChatOpenAI(**llm_args)

    elif "deepseek" in model_name_lower:
        resolved_api_key = api_key or os.getenv("DEEPSEEK_API_KEY") or os.getenv("LM_KEY")
        resolved_base_url = base_url or os.getenv("DEEPSEEK_BASE_URL") or "https://api.deepseek.com"

        if resolved_api_key:
            llm_args["api_key"] = resolved_api_key # ChatOpenAI uses 'api_key'
        if resolved_base_url:
            llm_args["base_url"] = resolved_base_url

        llm_args.update(kwargs)
        # DeepSeek models are used with ChatOpenAI client pointing to DeepSeek's base URL
        return ChatOpenAI(**llm_args)

    elif "qwen" in model_name_lower or "tongyi" in model_name_lower: # Common Qwen/Tongyi names
        # Note: ChatAlibabaTongyi might read DASHSCOPE_API_KEY from env by default.
        # Explicitly passing it if provided via api_key parameter or QWEN_API_KEY.
        resolved_api_key = api_key or os.getenv("QWEN_API_KEY") or os.getenv("DASHSCOPE_API_KEY")

        # For ChatAlibabaTongyi, the parameter is 'dashscope_api_key'.
        if resolved_api_key:
            llm_args["dashscope_api_key"] = resolved_api_key

        # ChatAlibabaTongyi does not typically take a 'base_url' parameter in the same way as ChatOpenAI.
        # Dashscope SDK's base URL is usually configured globally via dashscope.base_http_api_url
        # or relies on defaults. If a base_url is passed here for Qwen, we'll print a warning.
        if base_url:
            print(f"Warning: Custom 'base_url' ('{base_url}') was provided for Qwen model '{model_name}'. "
                  f"ChatAlibabaTongyi does not directly use a 'base_url' parameter in its constructor. "
                  f"Ensure Dashscope SDK is configured globally if a non-default base URL is needed.")

        # Remove 'api_key' and 'base_url' from llm_args if they were added by mistake from other paths
        # (though the current structure should prevent this for Qwen path)
        llm_args.pop("api_key", None)
        llm_args.pop("base_url", None)

        llm_args.update(kwargs)
        return ChatAlibabaTongyi(**llm_args)

    else:
        raise ValueError(f"Model '{model_name}' not supported by get_langchain_llm factory. "
                         f"Supported model families: OpenAI (gpt-...), DeepSeek, Qwen/Tongyi.")

# Example Usage (can be removed or kept for testing)
if __name__ == '__main__':
    # Ensure API keys are set in your environment for these tests to run
    # export OPENAI_API_KEY="..."
    # export DEEPSEEK_API_KEY="..."
    # export DASHSCOPE_API_KEY="..."

    print("Attempting to get GPT-4o...")
    try:
        gpt_llm = get_langchain_llm("gpt-4o", temperature=0.5)
        print(f"GPT-4o LLM: {gpt_llm.__class__.__name__}, Model: {gpt_llm.model_name}, Temp: {gpt_llm.temperature}")
        # response = gpt_llm.invoke("Hello, who are you?")
        # print(f"GPT-4o Response: {response.content[:100]}...")
    except Exception as e:
        print(f"Error getting/using GPT-4o: {e}")

    print("\nAttempting to get DeepSeek Chat...")
    try:
        ds_llm = get_langchain_llm("deepseek-chat", temperature=0.6)
        print(f"DeepSeek LLM: {ds_llm.__class__.__name__}, Model: {ds_llm.model_name}, Temp: {ds_llm.temperature}")
        # response = ds_llm.invoke("Hello, who are you?")
        # print(f"DeepSeek Response: {response.content[:100]}...")
    except Exception as e:
        print(f"Error getting/using DeepSeek: {e}")

    print("\nAttempting to get Qwen Max...")
    try:
        qwen_llm = get_langchain_llm("qwen-max", temperature=0.8)
        print(f"Qwen LLM: {qwen_llm.__class__.__name__}, Model: {qwen_llm.model_name}, Temp: {qwen_llm.temperature}")
        # response = qwen_llm.invoke("Hello, who are you?")
        # print(f"Qwen Response: {response.content[:100]}...")
    except Exception as e:
        print(f"Error getting/using Qwen: {e}")

    print("\nAttempting to get custom model with kwargs (e.g., max_tokens)...")
    try:
        gpt_custom_llm = get_langchain_llm("gpt-3.5-turbo", temperature=0.5, max_tokens=100)
        print(f"GPT Custom LLM: {gpt_custom_llm.__class__.__name__}, Model: {gpt_custom_llm.model_name}, Temp: {gpt_custom_llm.temperature}, MaxTokens: {gpt_custom_llm.max_tokens}")
    except Exception as e:
        print(f"Error getting/using Custom GPT: {e}")

    print("\nAttempting to get unsupported model...")
    try:
        unsupported_llm = get_langchain_llm("unknown-model-9000")
    except ValueError as e:
        print(f"Correctly caught error for unsupported model: {e}")
    except Exception as e:
        print(f"Unexpected error for unsupported model: {e}")
