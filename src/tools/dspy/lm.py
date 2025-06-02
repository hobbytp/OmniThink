import random
import threading
import time
import dspy
import os
from openai import OpenAI
from zhipuai import ZhipuAI
from typing import Optional, Literal, Any, List, Dict # Added Dict
import dashscope
from dashscope import Generation

# This code is originally sourced from Repository STORM
# URL: [https://github.com/stanford-oval/storm]


class OpenAIModel_dashscope(dspy.OpenAI):
    """A wrapper class for dspy.OpenAI."""

    def __init__(
            self,
            model: str = "gpt-4o",
            max_tokens: int = 2000,
            api_key: Optional[str] = None,
            api_base_url: Optional[str] = None,
            **kwargs
    ):
        """
        Initializes the OpenAIModel_dashscope.

        Args:
            model (str, optional): The model name to use. Defaults to "gpt-4o".
            max_tokens (int, optional): The maximum number of tokens to generate. Defaults to 2000.
            api_key (Optional[str], optional): The OpenAI API key. If not provided,
                it might be sourced from environment variables by the underlying client.
            api_base_url (Optional[str], optional): The base URL for the OpenAI API.
                If provided, this will be used. Otherwise, the value from the
                OPENAI_BASE_URL environment variable is used. If neither is set,
                a default URL ('https://api.gpts.vin/') is used.
            **kwargs: Additional keyword arguments to pass to the dspy.OpenAI constructor.
        """
        # Determine the base_url
        resolved_base_url = api_base_url  # Prioritize parameter
        if resolved_base_url is None:
            resolved_base_url = os.getenv("OPENAI_BASE_URL")  # Then environment variable
        if resolved_base_url is None:
            resolved_base_url = 'https://api.gpts.vin/'  # Fallback to default

        super().__init__(model=model, api_key=api_key, base_url=resolved_base_url, **kwargs)
        print(model)
        self.model = model
        self._token_usage_lock = threading.Lock()
        self.max_tokens = max_tokens
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def log_usage(self, response):
        """Log the total tokens from the OpenAI API response."""
        usage_data = response.get('usage')
        if usage_data:
            with self._token_usage_lock:
                self.prompt_tokens += usage_data.get('input_tokens', 0)
                self.completion_tokens += usage_data.get('output_tokens', 0)

    def get_usage_and_reset(self):
        """Get the total tokens used and reset the token usage."""
        usage = {
            self.kwargs.get('model') or self.kwargs.get('engine'):
                {'prompt_tokens': self.prompt_tokens, 'completion_tokens': self.completion_tokens}
        }
        self.prompt_tokens = 0
        self.completion_tokens = 0

        return usage

    def __call__(
            self,
            prompt: str,
            only_completed: bool = True,
            return_sorted: bool = False,
            **kwargs,
    ) -> list[dict[str, Any]]: # Retained original type hint for this specific class
        """Copied from dspy/dsp/modules/gpt3.py with the addition of tracking token usage."""

        assert only_completed, "for now"
        assert return_sorted is False, "for now"

        CALL_URL = 'https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions'
        LM_KEY = os.getenv('LM_KEY')
        HEADERS = {
            'Content-Type': 'application/json',
            "Authorization": f"Bearer {LM_KEY}"
        }

        kwargs_call = dict( # Renamed to avoid conflict with method's kwargs
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=self.max_tokens,
            stream=False,
        )
        import requests # Keep import local if only used here
        max_try = 10
        for i in range(max_try):
            try:
                ret = requests.post(CALL_URL, json=kwargs_call, # Use renamed dict
                                    headers=HEADERS, timeout=1000)
                if ret.status_code != 200:
                    raise Exception(f"http status_code: {ret.status_code}\n{ret.content}")
                ret_json = ret.json()
                for output_choice in ret_json['choices']: # Renamed to avoid conflict
                    if output_choice['finish_reason'] not in ['stop', 'function_call']:
                        raise Exception(f'openai finish with error...\n{ret_json}')
                return [ret_json['choices'][0]['message']['content']]
            except Exception as e:
                print(f"请求失败: {e}. 尝试重新请求...")
                time.sleep(1)
        return [] # Ensure return outside loop if all retries fail


class BaseDeepSeekModel(dspy.OpenAI):
    """
    Base class for DeepSeek models providing common initialization for API key,
    base URL, client, and token tracking. It also handles the common API call
    logic, while message preparation is delegated to subclasses.
    """
    def __init__(
            self,
            model: str,
            api_key: Optional[str] = None,
            api_base_url: Optional[str] = None,
            **kwargs
    ):
        """
        Initializes the BaseDeepSeekModel.

        Args:
            model (str): The specific DeepSeek model name (e.g., "deepseek-chat").
            api_key (Optional[str], optional): The DeepSeek API key.
                Resolution priority:
                1. This `api_key` parameter.
                2. `DEEPSEEK_API_KEY` environment variable.
                3. `LM_KEY` environment variable (fallback).
            api_base_url (Optional[str], optional): The base URL for the DeepSeek API.
                Resolution priority:
                1. This `api_base_url` parameter.
                2. `DEEPSEEK_BASE_URL` environment variable.
                3. Defaults to "https://api.deepseek.com".
            **kwargs: Additional keyword arguments to pass to the dspy.OpenAI constructor.
        """
        resolved_api_key = api_key
        if resolved_api_key is None:
            resolved_api_key = os.getenv("DEEPSEEK_API_KEY")
        if resolved_api_key is None:
            resolved_api_key = os.getenv("LM_KEY")

        resolved_base_url = api_base_url
        if resolved_base_url is None:
            resolved_base_url = os.getenv("DEEPSEEK_BASE_URL")
        if resolved_base_url is None:
            resolved_base_url = "https://api.deepseek.com"

        # For dspy.OpenAI, base_url is the correct parameter.
        # If dspy.OpenAI specifically expected api_base, this would need adjustment.
        super().__init__(model=model, api_key=resolved_api_key, base_url=resolved_base_url, **kwargs)

        self.model = model
        self.client = OpenAI(api_key=resolved_api_key, base_url=resolved_base_url)

        self._token_usage_lock = threading.Lock()
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def _prepare_messages(self, prompt: str) -> List[Dict[str, str]]:
        """
        Prepares the list of messages for the API call.
        Subclasses should override this to add model-specific system prompts
        or other message formatting.
        """
        return [{"role": "user", "content": prompt}]

    def __call__(
            self,
            prompt: str,
            only_completed: bool = True,
            return_sorted: bool = False,
            **kwargs,
    ) -> List[str]:
        """
        Makes an API call to the DeepSeek model.
        Handles retries and token counting. Message preparation is done by
        `_prepare_messages`.
        """
        assert only_completed, "for now"
        assert return_sorted is False, "for now"

        messages = self._prepare_messages(prompt)

        max_retries = 3
        attempt = 0
        response_obj = None
        # Pass through relevant kwargs from dspy.OpenAI to the client if any.
        # Common ones like 'temperature', 'max_tokens' (already handled by some models), 'top_p', etc.
        # For this base class, we'll pass all additional kwargs.
        client_kwargs = {k: v for k, v in kwargs.items() if k not in ['only_completed', 'return_sorted']}

        while attempt < max_retries:
            try:
                response_obj = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    stream=False,
                    **client_kwargs
                )

                if response_obj and response_obj.usage:
                    with self._token_usage_lock:
                        self.prompt_tokens += response_obj.usage.prompt_tokens
                        self.completion_tokens += response_obj.usage.completion_tokens

                choices_to_process = response_obj.choices if response_obj else []

                if only_completed:
                    completed_choices = [c for c in choices_to_process if c.finish_reason != "length"]
                    choices_to_process = completed_choices

                completions = [c.message.content for c in choices_to_process if c.message and c.message.content is not None]
                return completions

            except Exception as e:
                print(f"DeepSeek API call failed (model: {self.model}) on attempt {attempt + 1}/{max_retries}: {e}")
                # Consider logging the full error for debugging if necessary
                # import traceback
                # print(traceback.format_exc())
                delay = random.uniform(1, 3)
                time.sleep(delay)
                attempt += 1

        print(f"DeepSeek API call failed after {max_retries} retries for model {self.model}, prompt: {prompt[:100]}...")
        return []

    def get_usage_and_reset(self):
        """Get the total tokens used and reset the token usage."""
        model_key = self.kwargs.get('model_name') or self.kwargs.get('engine') or self.model # dspy.OpenAI uses model_name
        usage = {
            model_key:
                {'prompt_tokens': self.prompt_tokens, 'completion_tokens': self.completion_tokens}
        }
        self.prompt_tokens = 0
        self.completion_tokens = 0
        return usage


class DeepSeekChatModel(BaseDeepSeekModel):
    """
    Specialized DeepSeek model class for chat applications,
    includes a default system prompt for helpful assistance.
    """

    def __init__(
            self,
            model: str = "deepseek-chat",
            api_key: Optional[str] = None,
            api_base_url: Optional[str] = None,
            **kwargs
    ):
        """
        Initializes the DeepSeekChatModel.

        Args:
            model (str, optional): The specific DeepSeek chat model name.
                                   Defaults to "deepseek-chat".
            api_key (Optional[str], optional): The DeepSeek API key.
            api_base_url (Optional[str], optional): The base URL for the DeepSeek API.
            **kwargs: Additional keyword arguments for BaseDeepSeekModel.
        """
        super().__init__(model=model, api_key=api_key, api_base_url=api_base_url, **kwargs)

    def _prepare_messages(self, prompt: str) -> List[Dict[str, str]]:
        """
        Prepares the messages list for the API call, including a system prompt
        for helpful chat behavior.
        """
        return [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": prompt},
        ]


class DeepSeekReasonerModel(BaseDeepSeekModel):
    """
    Specialized DeepSeek model class for reasoner-type applications,
    which does not include a default system prompt.
    """

    def __init__(
            self,
            model: str = "deepseek-reasoner",
            api_key: Optional[str] = None,
            api_base_url: Optional[str] = None,
            **kwargs
    ):
        """
        Initializes the DeepSeekReasonerModel.

        Args:
            model (str, optional): The specific DeepSeek reasoner model name.
                                   Defaults to "deepseek-reasoner".
            api_key (Optional[str], optional): The DeepSeek API key.
            api_base_url (Optional[str], optional): The base URL for the DeepSeek API.
            **kwargs: Additional keyword arguments for BaseDeepSeekModel.
        """
        super().__init__(model=model, api_key=api_key, api_base_url=api_base_url, **kwargs)

    def _prepare_messages(self, prompt: str) -> List[Dict[str, str]]:
        """
        Prepares the messages list for the API call, containing only the user prompt.
        """
        return [
            {"role": "user", "content": prompt},
        ]


class QwenModel(dspy.OpenAI):
    """A wrapper class for Alibaba Qwen models using the Dashscope SDK."""

    def __init__(
            self,
            model: str = "qwen-max",
            api_key: Optional[str] = None,
            api_base_url: Optional[str] = None,
            **kwargs
    ):
        """
        Initializes the QwenModel.

        Note: `dashscope.api_key` and `dashscope.base_http_api_url` are global
        configurations for the Dashscope SDK. Setting them here will affect
        other Dashscope calls.

        Args:
            model (str, optional): The Qwen model name. Defaults to "qwen-max".
            api_key (Optional[str], optional): The Dashscope API key.
                Resolution priority:
                1. This `api_key` parameter.
                2. `QWEN_API_KEY` environment variable.
                3. `DASHSCOPE_API_KEY` environment variable.
                If none are set, Dashscope SDK will look for `DASHSCOPE_API_KEY`
                or raise an error if no key is found.
            api_base_url (Optional[str], optional): The base URL for Dashscope API.
                Resolution priority:
                1. This `api_base_url` parameter.
                2. `QWEN_BASE_URL` environment variable.
                3. `DASHSCOPE_BASE_URL` environment variable.
                If none are set, Dashscope SDK uses its default URL.
            **kwargs: Additional keyword arguments to pass to the dspy.OpenAI constructor.
        """
        resolved_api_key = api_key
        if resolved_api_key is None:
            resolved_api_key = os.getenv("QWEN_API_KEY")
        if resolved_api_key is None:
            resolved_api_key = os.getenv("DASHSCOPE_API_KEY")

        if resolved_api_key:
            dashscope.api_key = resolved_api_key

        resolved_base_url = api_base_url
        if resolved_base_url is None:
            resolved_base_url = os.getenv("QWEN_BASE_URL")
        if resolved_base_url is None:
            resolved_base_url = os.getenv("DASHSCOPE_BASE_URL")

        if resolved_base_url:
            dashscope.base_http_api_url = resolved_base_url

        super().__init__(model=model, api_key=resolved_api_key, api_base=None, **kwargs)

        self.model = model

        self._token_usage_lock = threading.Lock()
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def log_usage(self, usage_dict: dict):
        """Log the total tokens from the Dashscope API response usage dict."""
        usage_data = usage_dict
        if usage_data:
            with self._token_usage_lock:
                self.prompt_tokens += usage_data.get('input_tokens', 0)
                self.completion_tokens += usage_data.get('output_tokens', 0)

    def get_usage_and_reset(self):
        """Get the total tokens used and reset the token usage."""
        usage = {
            (self.kwargs.get('model') or self.kwargs.get('engine') or self.model):
                {'prompt_tokens': self.prompt_tokens, 'completion_tokens': self.completion_tokens}
        }
        self.prompt_tokens = 0
        self.completion_tokens = 0
        return usage

    def __call__(
            self,
            prompt: str,
            only_completed: bool = True,
            return_sorted: bool = False,
            **kwargs,
    ) -> List[str]:
        assert only_completed, "for now"
        assert return_sorted is False, "for now"

        messages = [{'role': 'user', 'content': prompt}]
        max_retries = 3
        attempt = 0
        response = None
        # kwargs from dspy.OpenAI are not directly passed to Generation.call unless supported
        # For now, not passing **kwargs to Generation.call as per previous QwenModel structure
        while attempt < max_retries:
            try:
                response = dashscope.Generation.call(
                    model=self.model,
                    messages=messages,
                    result_format='message',
                )

                if response.status_code == 200 and response.output and response.usage:
                    self.log_usage(response.usage)

                    choices = response.output.get("choices", [])

                    processed_choices = choices
                    if only_completed:
                        completed_choices = [c for c in choices if c.get('finish_reason') != "length"]
                        if completed_choices or not choices:
                            processed_choices = completed_choices
                        else:
                            processed_choices = []

                    completions = [c['message']['content'] for c in processed_choices if c.get('message') and c['message'].get('content')]
                    return completions
                elif response.status_code != 200:
                    print(f"Dashscope API Error: Status {response.status_code}, Code {response.code if hasattr(response, 'code') else 'N/A'}, Message {response.message if hasattr(response, 'message') else 'N/A'}")

            except Exception as e:
                print(f"Dashscope call failed on attempt {attempt + 1}/{max_retries}: {e}")

            delay = random.uniform(1, 5)
            time.sleep(delay)
            attempt += 1

        print(f"Dashscope call failed after {max_retries} retries for model {self.model}.")
        return []