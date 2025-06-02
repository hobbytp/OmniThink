import unittest
from unittest.mock import patch, MagicMock
import os

# Make sure to adjust the import path if your structure is different
from src.langchain_support.llm_config import get_langchain_llm
from langchain.llms.fake import FakeListLLM
from langchain_core.language_models.llms import BaseLLM

# Conditional imports for actual LLM classes for type checking
try:
    from langchain_openai import ChatOpenAI
except ImportError:
    ChatOpenAI = None

try:
    from langchain_community.llms import HuggingFaceHub
except ImportError:
    try:
        from langchain.llms import HuggingFaceHub
    except ImportError:
        HuggingFaceHub = None


class TestGetLangchainLLM(unittest.TestCase):

    def test_get_fake_llm(self):
        """Test successful instantiation of FakeListLLM."""
        responses = ["fake response 1", "fake response 2"]
        llm = get_langchain_llm(provider='fake', responses=responses)
        self.assertIsInstance(llm, FakeListLLM)
        self.assertEqual(llm.responses, responses)

    def test_get_fake_llm_default_responses(self):
        """Test FakeListLLM with default responses."""
        llm = get_langchain_llm(provider='fake')
        self.assertIsInstance(llm, FakeListLLM)
        # Check against the default responses defined in llm_config.py
        from src.langchain_support.llm_config import DEFAULT_FAKE_RESPONSES
        self.assertEqual(llm.responses, DEFAULT_FAKE_RESPONSES)

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test_api_key_env"})
    @patch('src.langchain_support.llm_config.ChatOpenAI') # Mock the class
    def test_get_openai_llm_from_env_key(self, MockChatOpenAI):
        """Test ChatOpenAI instantiation with API key from environment."""
        if ChatOpenAI is None: # If langchain_openai is not installed
            self.skipTest("langchain-openai not installed, skipping ChatOpenAI test.")
            
        # Configure the mock constructor to return a MagicMock instance
        mock_openai_instance = MagicMock(spec=ChatOpenAI)
        MockChatOpenAI.return_value = mock_openai_instance
        
        llm = get_langchain_llm(provider='openai', model_name='gpt-3.5-turbo')
        
        self.assertIsInstance(llm, MagicMock) # Check if it's the mocked instance
        MockChatOpenAI.assert_called_once_with(
            model_name='gpt-3.5-turbo', 
            openai_api_key="test_api_key_env"
        )

    @patch('src.langchain_support.llm_config.ChatOpenAI')
    def test_get_openai_llm_from_passed_key(self, MockChatOpenAI):
        """Test ChatOpenAI instantiation with passed API key."""
        if ChatOpenAI is None:
            self.skipTest("langchain-openai not installed, skipping ChatOpenAI test.")

        mock_openai_instance = MagicMock(spec=ChatOpenAI)
        MockChatOpenAI.return_value = mock_openai_instance
        
        llm = get_langchain_llm(provider='openai', api_key="test_api_key_passed", model_name='gpt-4')
        
        self.assertIsInstance(llm, MagicMock)
        MockChatOpenAI.assert_called_once_with(
            model_name='gpt-4', 
            openai_api_key="test_api_key_passed"
        )

    @patch.dict(os.environ, clear=True) # Ensure OPENAI_API_KEY is not set
    @patch('src.langchain_support.llm_config.ChatOpenAI', new_callable=MagicMock) # Ensure it's a mock even if None
    def test_get_openai_llm_missing_key(self, MockChatOpenAI_class):
        """Test ChatOpenAI error with missing API key."""
        if ChatOpenAI is None: # If actual ChatOpenAI is None due to import error
             # Simulate the class being None in get_langchain_llm by making the mock behave so
            with patch('src.langchain_support.llm_config.ChatOpenAI', None):
                 with self.assertRaises(ImportError): # It should raise ImportError first
                    get_langchain_llm(provider='openai')
            return # End test if module not installed

        # If ChatOpenAI is available but key is missing
        with self.assertRaises(ValueError) as context:
            get_langchain_llm(provider='openai')
        self.assertIn("OpenAI API key not provided", str(context.exception))


    @patch('src.langchain_support.llm_config.ChatOpenAI', None) # Simulate not installed
    def test_get_openai_llm_import_error(self):
        """Test ChatOpenAI raises ImportError if langchain_openai is not installed."""
        with self.assertRaises(ImportError) as context:
            get_langchain_llm(provider='openai')
        self.assertIn("langchain-openai package not found", str(context.exception))
        

    @patch.dict(os.environ, {"HUGGINGFACEHUB_API_TOKEN": "test_hf_token_env"})
    @patch('src.langchain_support.llm_config.HuggingFaceHub') # Mock the class
    def test_get_huggingface_llm_from_env_key(self, MockHuggingFaceHub):
        """Test HuggingFaceHub instantiation with token from environment."""
        if HuggingFaceHub is None:
            self.skipTest("langchain_community or huggingface_hub not installed, skipping HuggingFaceHub test.")

        mock_hf_instance = MagicMock(spec=HuggingFaceHub)
        MockHuggingFaceHub.return_value = mock_hf_instance
        
        llm = get_langchain_llm(provider='huggingface', model_name='google/flan-t5-small')
        
        self.assertIsInstance(llm, MagicMock)
        MockHuggingFaceHub.assert_called_once_with(
            repo_id='google/flan-t5-small', 
            huggingfacehub_api_token="test_hf_token_env"
        )

    @patch('src.langchain_support.llm_config.HuggingFaceHub')
    def test_get_huggingface_llm_from_passed_key(self, MockHuggingFaceHub):
        """Test HuggingFaceHub instantiation with passed token."""
        if HuggingFaceHub is None:
            self.skipTest("langchain_community or huggingface_hub not installed, skipping HuggingFaceHub test.")
            
        mock_hf_instance = MagicMock(spec=HuggingFaceHub)
        MockHuggingFaceHub.return_value = mock_hf_instance

        llm = get_langchain_llm(provider='huggingface', api_key="test_hf_token_passed", model_name='distilbert-base-uncased')
        
        self.assertIsInstance(llm, MagicMock)
        MockHuggingFaceHub.assert_called_once_with(
            repo_id='distilbert-base-uncased', 
            huggingfacehub_api_token="test_hf_token_passed"
        )

    @patch.dict(os.environ, clear=True)
    @patch('src.langchain_support.llm_config.HuggingFaceHub', new_callable=MagicMock)
    def test_get_huggingface_llm_missing_token(self, MockHuggingFaceHub_class):
        """Test HuggingFaceHub error with missing token."""
        if HuggingFaceHub is None:
            with patch('src.langchain_support.llm_config.HuggingFaceHub', None):
                with self.assertRaises(ImportError):
                    get_langchain_llm(provider='huggingface', model_name='any/model')
            return

        with self.assertRaises(ValueError) as context:
            get_langchain_llm(provider='huggingface', model_name='google/flan-t5-small')
        self.assertIn("Hugging Face API token not provided", str(context.exception))

    @patch.dict(os.environ, {"HUGGINGFACEHUB_API_TOKEN": "test_hf_token_env"})
    @patch('src.langchain_support.llm_config.HuggingFaceHub', new_callable=MagicMock)
    def test_get_huggingface_llm_missing_model_name(self, MockHuggingFaceHub_class):
        """Test HuggingFaceHub error with missing model_name."""
        if HuggingFaceHub is None: # Should not happen if token is set and mock is active, but for safety
             self.skipTest("HuggingFaceHub not available for testing.")

        with self.assertRaises(ValueError) as context:
            get_langchain_llm(provider='huggingface')
        self.assertIn("model_name (Hugging Face Hub repository ID) must be provided", str(context.exception))

    @patch('src.langchain_support.llm_config.HuggingFaceHub', None) # Simulate not installed
    def test_get_huggingface_llm_import_error(self):
        """Test HuggingFaceHub raises ImportError if not installed."""
        with self.assertRaises(ImportError) as context:
            get_langchain_llm(provider='huggingface', model_name='any/model')
        self.assertIn("langchain_community (or langchain) and huggingface_hub packages not found", str(context.exception))

    def test_unknown_provider(self):
        """Test error for unknown provider."""
        with self.assertRaises(ValueError) as context:
            get_langchain_llm(provider='unknown_provider')
        self.assertIn("Unknown LLM provider: unknown_provider", str(context.exception))


if __name__ == '__main__':
    unittest.main()
