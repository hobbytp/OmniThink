import unittest
from unittest.mock import MagicMock, patch
import dspy

# Target modules
from src.actions.article_polish import ArticlePolishingModule, PolishPageModule, PolishPage
from src.langchain_support.dspy_equivalents import LangchainPredict, LangchainSignature
from langchain.llms.fake import FakeListLLM
from langchain_core.language_models.llms import BaseLLM # For type hinting
from typing import Optional, List, Dict # Added missing imports

# Mock for dspy.LM
class MockDSPyLM(dspy.LM):
    def __init__(self, responses=None, output_field_name='page'): # Default to 'page' for PolishPage
        super().__init__(model="mock_model")
        self.output_field_name = output_field_name
        default_response_str = f'{{"{self.output_field_name}": "dspy lm default polish response"}}'
        self.responses = responses if responses else [default_response_str]
        self.history = []

    def __call__(self, prompt: Optional[str] = None, messages: Optional[List[Dict[str, str]]] = None, only_completed=True, return_sorted=False, **kwargs):
        current_prompt_str = ""
        if prompt is not None:
            current_prompt_str = prompt
        elif messages is not None and isinstance(messages, list) and messages:
            # Extract content from the last message, assuming it's the user query
            current_prompt_str = messages[-1].get("content", "")
            # Or, more robustly, find the last 'user' role message
            for msg in reversed(messages):
                if msg.get("role") == "user" and "content" in msg:
                    current_prompt_str = msg["content"]
                    break

        response_str = self.responses.pop(0) if self.responses else f'{{"{self.output_field_name}": "dspy lm fallback polish"}}'
        # Ensure the response is a JSON string if the adapter expects it.
        # For dspy.Predict, the LM is often expected to return the raw completion string,
        # and dspy.Predict handles parsing based on the signature.
        # If an adapter like ChatAdapter or JSONAdapter is used, it might expect the LM to already return JSON.
        # The error suggests JSONAdapter is involved.
        self.history.append({"prompt": current_prompt_str, "messages": messages, "response": response_str, "kwargs": kwargs})
        return [response_str]

    def basic_request(self, prompt: str, **kwargs):
        response_str = self.responses.pop(0) if self.responses else f'{{"{self.output_field_name}": "dspy lm fallback polish basic_request"}}'
        self.history.append({"prompt": prompt, "response": response_str, "kwargs": kwargs, "method": "basic_request"})
        return response_str


# Mock for StormArticle (or similar article object used by ArticlePolishingModule)
class MockStormArticle:
    def __init__(self, text_content="Initial article text with # Section 1.\nAnd some content."):
        self.text_content = text_content
        self.processed_dict = None # To store what parse_article_into_dict would produce
        self.updated_sections = {} # To store content after insert_or_create_section

    def to_string(self) -> str:
        return self.text_content

    def insert_or_create_section(self, article_dict):
        # Simulate updating internal structure based on parsed dict
        self.updated_sections.update(article_dict)
        # For more accurate testing, this mock would need to reflect StormArticle's actual behavior

    def post_processing(self):
        # Simulate any post_processing
        pass

    # Add a method to easily check what content was inserted for testing
    def get_section_content(self, section_title):
        return self.updated_sections.get(section_title)


class TestPolishPageModule(unittest.TestCase):
    def test_init_dspy(self):
        mock_dspy_lm = MockDSPyLM()
        # PolishPage is a class in article_polish, so it should be found.
        polish_module = PolishPageModule(
            # write_lead_engine parameter removed
            polish_engine=mock_dspy_lm,
            framework='dspy'
        )
        self.assertEqual(polish_module.framework, 'dspy')
        self.assertIsInstance(polish_module.polish_page_predictor, dspy.Predict)

    def test_init_langchain(self):
        fake_lc_llm = FakeListLLM(responses=["lc polish response"])
        polish_module = PolishPageModule(
            # write_lead_engine parameter removed
            polish_engine=fake_lc_llm,
            framework='langchain'
        )
        self.assertEqual(polish_module.framework, 'langchain')
        self.assertIsInstance(polish_module.polish_page_predictor, LangchainPredict)
        self.assertIsInstance(polish_module.polish_page_predictor.signature, LangchainSignature)

    def test_forward_dspy(self):
        response_text = "Polished DSPy article content."
        # The mock LM should now return a JSON string for dspy.Predict to parse
        mock_response_json_str = f'{{"page": "{response_text}"}}'
        # Provide an extra dummy response in case dspy.Predict makes an initial call
        dummy_initial_response = f'{{"page": "dummy initial response"}}'
        mock_dspy_lm = MockDSPyLM(responses=[dummy_initial_response, mock_response_json_str], output_field_name="page")
        dspy.settings.configure(lm=mock_dspy_lm) # Configure DSPy context

        polish_module = PolishPageModule(
            # write_lead_engine parameter removed
            polish_engine=mock_dspy_lm,
            framework='dspy'
        )

        result = polish_module.forward(
            topic="Test Topic", # Not used by PolishPage signature
            draft_page="Original draft page content.",
            polish_whole_page=True # Not used by PolishPage signature
        )
        self.assertIsInstance(result, dspy.Prediction)
        self.assertEqual(result.page, response_text)

    def test_forward_langchain(self):
        response_text = "Polished LangChain article content."
        fake_lc_llm = FakeListLLM(responses=[response_text])
        polish_module = PolishPageModule(
            # write_lead_engine parameter removed
            polish_engine=fake_lc_llm,
            framework='langchain'
        )

        result = polish_module.forward(
            topic="LC Topic",
            draft_page="Original LC draft page.",
            polish_whole_page=True
        )
        self.assertIsInstance(result, dict)
        self.assertIn("page", result)
        self.assertEqual(result["page"], response_text)


class TestArticlePolishingModule(unittest.TestCase):
    def setUp(self):
        # self.mock_article_gen_lm_dummy = FakeListLLM(responses=["dummy gen response"]) # No longer needed
        pass

    def test_init_dspy(self):
        mock_dspy_polish_lm = MockDSPyLM()
        module = ArticlePolishingModule(
            # article_gen_lm parameter removed
            article_polish_lm=mock_dspy_polish_lm,
            framework='dspy'
        )
        self.assertEqual(module.framework, 'dspy')
        self.assertIsInstance(module.polish_page_module, PolishPageModule)
        self.assertEqual(module.polish_page_module.framework, 'dspy')

    def test_init_langchain(self):
        fake_lc_polish_llm = FakeListLLM(responses=["lc polish response for apm"])
        module = ArticlePolishingModule(
            # article_gen_lm parameter removed
            article_polish_lm=fake_lc_polish_llm,
            framework='langchain'
        )
        self.assertEqual(module.framework, 'langchain')
        self.assertIsInstance(module.polish_page_module, PolishPageModule)
        self.assertEqual(module.polish_page_module.framework, 'langchain')
        self.assertIsInstance(module.polish_page_module.polish_engine, BaseLLM)

    @patch('src.actions.article_polish.ArticleTextProcessing.parse_article_into_dict')
    def test_polish_article_langchain(self, mock_parse_article_into_dict):
        polished_text_response = "# Polished Title\nPolished LangChain article content."
        fake_lc_polish_llm = FakeListLLM(responses=[polished_text_response])

        # Mock the return value of parse_article_into_dict
        mock_parse_article_into_dict.return_value = {"Polished Title": "Polished LangChain article content."}

        module = ArticlePolishingModule(
            # article_gen_lm parameter removed
            article_polish_lm=fake_lc_polish_llm,
            framework='langchain'
        )

        draft_article_obj = MockStormArticle(text_content="Original text for polishing.")

        polished_article_result = module.polish_article(
            topic="AI Ethics", # Not directly used by the PolishPage signature itself
            draft_article=draft_article_obj,
            remove_duplicate=True # Corresponds to polish_whole_page
        )

        self.assertIsInstance(polished_article_result, MockStormArticle)
        # Check if parse_article_into_dict was called with the polished text
        mock_parse_article_into_dict.assert_called_once_with(polished_text_response)
        # Check if the article object was updated (simplified check)
        self.assertEqual(polished_article_result.updated_sections.get("Polished Title"), "Polished LangChain article content.")


if __name__ == '__main__':
    unittest.main()
