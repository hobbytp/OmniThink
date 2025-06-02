import random
import threading
import time
import dspy
import os
from openai import OpenAI
from zhipuai import ZhipuAI
from typing import Optional, Literal, Any
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
    ) -> list[dict[str, Any]]:
        """Copied from dspy/dsp/modules/gpt3.py with the addition of tracking token usage."""

        assert only_completed, "for now"
        assert return_sorted is False, "for now"

        CALL_URL = 'https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions'
        LM_KEY = os.getenv('LM_KEY')
        HEADERS = {
            'Content-Type': 'application/json',
            "Authorization": f"Bearer {LM_KEY}"
        }

        kwargs = dict(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=self.max_tokens,
            stream=False,
        )
        import requests
        max_try = 10
        for i in range(max_try):
            try:
                ret = requests.post(CALL_URL, json=kwargs,
                                    headers=HEADERS, timeout=1000)
                if ret.status_code != 200:
                    raise Exception(f"http status_code: {ret.status_code}\n{ret.content}")
                ret_json = ret.json()
                for output in ret_json['choices']:
                    if output['finish_reason'] not in ['stop', 'function_call']:
                        raise Exception(f'openai finish with error...\n{ret_json}')
                return [ret_json['choices'][0]['message']['content']]
            except Exception as e:
                print(f"请求失败: {e}. 尝试重新请求...")    
                time.sleep(1)


class DeepSeekModel(dspy.OpenAI):
    """A wrapper class for dspy.OpenAI."""

    def __init__(
            self,
            model: str = "deepseek-chat",
            api_key: Optional[str] = None,
            **kwargs
    ):
        super().__init__(model=model, api_key=api_key, **kwargs)
        self.model = model
        self.api_key = api_key
        self._token_usage_lock = threading.Lock()
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
    ) -> list[dict[str, Any]]:
        """Copied from dspy/dsp/modules/gpt3.py with the addition of tracking token usage."""

        assert only_completed, "for now"
        assert return_sorted is False, "for now"

        LM_KEY = os.getenv('LM_KEY')
        client = OpenAI(api_key=LM_KEY, base_url="https://api.deepseek.com")

        max_retries = 3
        attempt = 0
        messages = []
        if self.model != "deepseek-reasoner":
            messages.append({"role": "system", "content": "You are a helpful assistant"})
        messages.append({"role": "user", "content": prompt})
        print(messages)
        while attempt < max_retries:
            try:
                response = client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    stream=False
                )
                choices = response["output"]["choices"]
                break

            except Exception as e:
                delay = random.uniform(0, 3)
                time.sleep(delay)
                attempt += 1

        self.log_usage(response)

        completed_choices = [c for c in choices if c["finish_reason"] != "length"]

        if only_completed and len(completed_choices):
            choices = completed_choices

        completions = [c['message']['content'] for c in choices]

        return completions


class QwenModel(dspy.OpenAI):
    """A wrapper class for dspy.OpenAI."""

    def __init__(
            self,
            model: str = "qwen-max-allinone",
            api_key: Optional[str] = None,
            **kwargs
    ):
        super().__init__(model=model, api_key=api_key, **kwargs)
        self.model = model
        self.api_key = api_key
        self._token_usage_lock = threading.Lock()
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
    ) -> list[dict[str, Any]]:
        """Copied from dspy/dsp/modules/gpt3.py with the addition of tracking token usage."""

        assert only_completed, "for now"
        assert return_sorted is False, "for now"

        messages = [{'role': 'user', 'content': prompt}]
        max_retries = 3
        attempt = 0
        while attempt < max_retries:
            try:
                response = Generation.call(
                    model=self.model, 
                    messages=messages,
                    result_format='message',
                ) 
                choices = response["output"]["choices"]
                break

            except Exception as e:
                delay = random.uniform(0, 10)
                time.sleep(delay)
                attempt += 1

        self.log_usage(response)

        completed_choices = [c for c in choices if c["finish_reason"] != "length"]

        if only_completed and len(completed_choices):
            choices = completed_choices

        completions = [c['message']['content'] for c in choices]

        return completions