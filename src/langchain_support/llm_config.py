# Configuration and utility functions for LangChain LLMs
import os
from typing import Optional, List, Any

from langchain_core.language_models.llms import BaseLLM
from langchain.llms.fake import FakeListLLM

# Attempt to import provider-specific LLMs, but don't fail if not installed
try:
    from langchain_openai import ChatOpenAI
except ImportError:
    ChatOpenAI = None 

try:
    from langchain_community.llms import HuggingFaceHub # Or from langchain.llms for older versions
except ImportError:
    try:
        from langchain.llms import HuggingFaceHub # older import path
    except ImportError:
        HuggingFaceHub = None


DEFAULT_FAKE_RESPONSES = [
    "This is a default fake response.",
    "Another default fake response for testing.",
]

def get_langchain_llm(
    provider: str, 
    api_key: Optional[str] = None, 
    model_name: Optional[str] = None,
    responses: Optional[List[str]] = None, # Specific to FakeListLLM
    **kwargs: Any
) -> BaseLLM:
    """
    Instantiates and returns a LangChain LLM based on the specified provider.

    Args:
        provider: The name of the LLM provider (e.g., 'openai', 'huggingface', 'fake').
        api_key: Optional API key for the provider. If None, attempts to load from environment.
        model_name: Optional model name or repository ID.
        responses: Optional list of responses for FakeListLLM.
        **kwargs: Additional keyword arguments to pass to the LLM constructor.

    Returns:
        An instance of a LangChain BaseLLM.

    Raises:
        ValueError: If the provider is unknown or required components are missing.
        ImportError: If provider-specific libraries are not installed.
    """
    provider = provider.lower()

    if provider == 'openai':
        if ChatOpenAI is None:
            raise ImportError("langchain-openai package not found. Please install it via `pip install langchain-openai`.")
        
        openai_api_key = api_key if api_key else os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OpenAI API key not provided and not found in OPENAI_API_KEY environment variable.")
        
        # Common model names: "gpt-3.5-turbo", "gpt-4", etc.
        # Default to a common one if not specified
        model_to_use = model_name if model_name else "gpt-3.5-turbo"
        
        return ChatOpenAI(model_name=model_to_use, openai_api_key=openai_api_key, **kwargs)

    elif provider == 'huggingface':
        if HuggingFaceHub is None:
            raise ImportError("langchain_community (or langchain) and huggingface_hub packages not found or out of date. Please install them.")

        hf_api_token = api_key if api_key else os.getenv("HUGGINGFACEHUB_API_TOKEN")
        if not hf_api_token:
            raise ValueError("Hugging Face API token not provided and not found in HUGGINGFACEHUB_API_TOKEN environment variable.")
        
        if not model_name:
            raise ValueError("model_name (Hugging Face Hub repository ID) must be provided for HuggingFaceHub.")
        # Example model_name: "google/flan-t5-large"
        
        return HuggingFaceHub(repo_id=model_name, huggingfacehub_api_token=hf_api_token, **kwargs)

    elif provider == 'fake':
        # `responses` kwarg is specific to FakeListLLM
        fake_responses = responses if responses is not None else DEFAULT_FAKE_RESPONSES
        return FakeListLLM(responses=fake_responses, **kwargs)

    else:
        raise ValueError(
            f"Unknown LLM provider: {provider}. Supported providers: 'openai', 'huggingface', 'fake'."
        )

if __name__ == '__main__':
    print("--- Testing LLM Configuration Utility ---")

    # Test FakeListLLM
    print("\n1. Testing FakeListLLM...")
    try:
        fake_llm = get_langchain_llm(provider='fake', responses=["Test response 1", "Test response 2"])
        print(f"FakeListLLM instantiated: {fake_llm}")
        # Test a prediction (invoke expects a string or a list of messages)
        # For basic LLMs (not ChatModels), a string prompt is usually fine.
        print(f"FakeListLLM response 1: {fake_llm.invoke('Any prompt')}")
        print(f"FakeListLLM response 2: {fake_llm.invoke('Another prompt')}")
    except Exception as e:
        print(f"Error testing FakeListLLM: {e}")

    # Test OpenAI (will only work if OPENAI_API_KEY is set in env or passed)
    print("\n2. Testing OpenAI ChatOpenAI...")
    openai_api_key_env = os.getenv("OPENAI_API_KEY")
    if openai_api_key_env:
        try:
            openai_llm = get_langchain_llm(provider='openai', model_name='gpt-3.5-turbo')
            print(f"ChatOpenAI instantiated: {openai_llm}")
            # Test a prediction (invoke expects a string for basic call, or list of messages for chat)
            # For ChatOpenAI, it's better to send a list of messages or use .invoke("prompt")
            # result = openai_llm.invoke([HumanMessage(content="What is the capital of France?")])
            # print(f"ChatOpenAI response (first 50 chars): {str(result.content)[:50]}...")
            print("ChatOpenAI instantiation seems successful (actual call skipped in this test).")
        except Exception as e:
            print(f"Error testing ChatOpenAI: {e}")
    else:
        print("Skipping ChatOpenAI test as OPENAI_API_KEY environment variable is not set.")

    # Test HuggingFaceHub (conceptual, requires token and valid repo_id)
    print("\n3. Testing HuggingFaceHub (conceptual)...")
    huggingface_token_env = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if huggingface_token_env:
        try:
            # Replace with a valid, small model for testing if possible, e.g. "google/flan-t5-small"
            # For this test, we'll just check instantiation.
            hf_llm = get_langchain_llm(provider='huggingface', model_name="google/flan-t5-small", api_key=huggingface_token_env)
            print(f"HuggingFaceHub instantiated: {hf_llm}")
            print("HuggingFaceHub instantiation seems successful (actual call skipped in this test).")
        except Exception as e:
            print(f"Error testing HuggingFaceHub: {e}")
    else:
        print("Skipping HuggingFaceHub test as HUGGINGFACEHUB_API_TOKEN environment variable is not set.")
    
    print("\n--- LLM Configuration Utility Testing Done ---")
