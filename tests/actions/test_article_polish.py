import unittest
from unittest.mock import MagicMock, patch
import dspy

# Target modules
from src.actions.article_polish import ArticlePolishingModule, PolishPageModule, PolishPage
from src.langchain_support.dspy_equivalents import LangchainPredict, LangchainSignature
from langchain.llms.fake import FakeListLLM
from langchain_core.language_models.llms import BaseLLM # For type hinting

# Mock for dspy.LM
class MockDSPyLM(dspy.dsp.LM):
    def __init__(self, responses=None):
        super().__init__("mock_model")
        self.responses = responses if responses else ["dspy lm default polish response"]
        self.history = []

    def __call__(self, prompt, only_completed=True, return_sorted=False, **kwargs):
        response = self.responses.pop(0) if self.responses else "dspy lm fallback polish"
        self.history.append({"prompt": prompt, "response": response, "kwargs": kwargs})
        return [response]
    
    def basic_request(self, prompt, **kwargs): # dspy.Predict calls this
        response = self.responses.pop(0) if self.responses else "dspy lm fallback polish basic_request"
        return response


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
            write_lead_engine=mock_dspy_lm, # Not used by forward but required by init
            polish_engine=mock_dspy_lm, 
            framework='dspy'
        )
        self.assertEqual(polish_module.framework, 'dspy')
        self.assertIsInstance(polish_module.polish_page_predictor, dspy.Predict)

    def test_init_langchain(self):
        fake_lc_llm = FakeListLLM(responses=["lc polish response"])
        polish_module = PolishPageModule(
            write_lead_engine=fake_lc_llm, # Not used by forward but required by init
            polish_engine=fake_lc_llm, 
            framework='langchain'
        )
        self.assertEqual(polish_module.framework, 'langchain')
        self.assertIsInstance(polish_module.polish_page_predictor, LangchainPredict)
        self.assertIsInstance(polish_module.polish_page_predictor.signature, LangchainSignature)

    def test_forward_dspy(self):
        response_text = "Polished DSPy article content."
        mock_dspy_lm = MockDSPyLM(responses=[response_text])
        dspy.settings.configure(lm=mock_dspy_lm) # Configure DSPy context

        polish_module = PolishPageModule(
            write_lead_engine=mock_dspy_lm,
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
            write_lead_engine=fake_lc_llm,
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
        self.mock_article_gen_lm_dummy = FakeListLLM(responses=["dummy gen response"]) # For write_lead_engine if it were used

    def test_init_dspy(self):
        mock_dspy_polish_lm = MockDSPyLM()
        module = ArticlePolishingModule(
            article_gen_lm=mock_dspy_polish_lm, # Passed to PolishPageModule's write_lead_engine
            article_polish_lm=mock_dspy_polish_lm, # Passed to PolishPageModule's polish_engine
            framework='dspy'
        )
        self.assertEqual(module.framework, 'dspy')
        self.assertIsInstance(module.polish_page_module, PolishPageModule)
        self.assertEqual(module.polish_page_module.framework, 'dspy')

    def test_init_langchain(self):
        fake_lc_polish_llm = FakeListLLM(responses=["lc polish response for apm"])
        module = ArticlePolishingModule(
            article_gen_lm=self.mock_article_gen_lm_dummy, 
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
            article_gen_lm=self.mock_article_gen_lm_dummy,
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
