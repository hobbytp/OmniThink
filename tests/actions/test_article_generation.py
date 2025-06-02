import unittest
from unittest.mock import MagicMock, patch
import dspy 

# Target modules
from src.actions.article_generation import ArticleGenerationModule, ConvToSection, WriteSection
from src.langchain_support.dspy_equivalents import LangchainPredict, LangchainSignature
from langchain.llms.fake import FakeListLLM
from langchain_core.language_models.llms import BaseLLM


# Mock for dspy.LM, if needed for DSPy path testing
class MockDSPyLM(dspy.dsp.LM):
    def __init__(self, responses=None):
        super().__init__("mock_model")
        self.responses = responses if responses else ["dspy lm default response"]
        self.history = [] # To track calls, similar to FakeListLLM

    def __call__(self, prompt, only_completed=True, return_sorted=False, **kwargs):
        # Simplified mock: just return the next response
        response = self.responses.pop(0) if self.responses else "dspy lm fallback response"
        self.history.append({
            "prompt": prompt,
            "response": response,
            "kwargs": kwargs,
        })
        return [response] # dspy.LM typically returns a list of completions

    def basic_request(self, prompt, **kwargs): # dspy.Predict calls this
        response = self.responses.pop(0) if self.responses else "dspy lm fallback basic_request"
        # Simulate dspy.Predict's expectation of how history is populated or what it gets back.
        # This might need adjustment based on how dspy.Predict interacts with it.
        # For now, let's assume it just needs the text.
        return response 


# Mock for Retriever
class MockRetriever:
    def __init__(self, search_results=None):
        self.search_results = search_results if search_results is not None else [{"text": "dummy search result"}]
    
    def search(self, query: str, k: int):
        return self.search_results
    
    # If ArticleGenerationModule directly calls other specific retriever methods, mock them here.
    # For STORM, it seems `mindmap.retrieve_information` is used, so the retriever passed to
    # ArticleGenerationModule might be part of a mindmap object or used by it.
    # Let's assume for now this simple retriever is sufficient for direct use or as part of a mock mindmap.


# Mock for MindMap used in ArticleGenerationModule.generate_section
class MockMindMap:
    def __init__(self, retrieve_info_results=None):
        self.retrieve_info_results = retrieve_info_results if retrieve_info_results is not None else [
            {"snippets": ["snippet 1 from mindmap"], "source": "mindmap_source1"}
        ]

    def retrieve_information(self, queries, search_top_k):
        # Return canned data, possibly based on queries if complex testing is needed
        return self.retrieve_info_results

    def prepare_table_for_retrieval(self): # Called in generate_article
        pass


# Mock for ArticleWithOutline (or similar object used in generate_article)
class MockArticleWithOutline:
    def __init__(self, sections=None, outline_list=None):
        self.sections = sections if sections else ["Introduction", "Body", "Conclusion"]
        self.outline_list = outline_list if outline_list else ["# Introduction", "## Point 1"]
        self.content_dict = {} # To store updated sections

    def get_first_level_section_names(self):
        return self.sections

    def get_outline_as_list(self, root_section_name, add_hashtags):
        return self.outline_list # Simplified

    def update_section(self, parent_section_name, current_section_content, current_section_info_list):
        # Store the generated content for verification
        self.content_dict[parent_section_name] = { # Assuming parent_section_name is like the overall topic for the whole article
            "content": current_section_content,     # and current_section_content is one whole section's text
            "info_list": current_section_info_list
        }
        # In reality, this method would update a more complex internal structure.
        # For testing generate_section, this might not be directly hit if we test section_gen output.
        # For testing generate_article, this is important.
        # The actual StormArticle.update_section seems to update based on section title within current_section_content
        # For simplicity here, we'll just store the whole text blob.
        if parent_section_name not in self.content_dict:
             self.content_dict[parent_section_name] = {} # Initialize if not present
        # A bit of a simplification: store based on the main topic, assuming one section content blob for now
        # Or, more accurately, the generate_article loop calls update_section for each section_title
        # So parent_section_name there is the *topic*, and current_section_content is the *content of one first-level section*.
        # The original code has `article.update_section(parent_section_name=topic, ...)` which is a bit confusing.
        # Let's assume `parent_section_name` in `update_section` refers to the section title being updated.
        # This is a detail of StormArticle, for our test, we just need to see it's called.
        # The key for `section_output_dict` is `section_name`.
        # `article.update_section(parent_section_name=section_output_dict["section_name"], ...)` might be more accurate.
        # However, the original code in ArticleGenerationModule.generate_article passes `topic` as `parent_section_name`.
        # This suggests StormArticle's `update_section` might be more nuanced.
        # For testing, the important part is that `current_section_content` is passed.
        # Let's refine this mock if direct testing of generate_article's effect on the article object is needed.
        # For now, let's assume `generate_section` is tested more directly.
        
        # A simple way to track:
        if not hasattr(self, 'updated_sections_content'):
            self.updated_sections_content = {}
        # The key in `section_output_dict_collection` is based on the section title.
        # `current_section_content` is the actual text.
        # The original `update_section` might be using the section title from *within* `current_section_content`.
        # We'll just store the raw content by the first section title for simplicity.
        first_title_in_content = current_section_content.split('\n')[0].lstrip('# ').strip()
        self.updated_sections_content[first_title_in_content] = current_section_content


    def post_processing(self):
        pass
    
    def to_string(self): # If needed by polish module
        return "\n".join(self.updated_sections_content.values())


class TestConvToSection(unittest.TestCase):
    def test_init_dspy(self):
        mock_dspy_lm = MockDSPyLM()
        # WriteSection is a class in article_generation, so it should be found.
        conv_module = ConvToSection(class_name='WriteSection', engine=mock_dspy_lm, framework='dspy')
        self.assertEqual(conv_module.framework, 'dspy')
        self.assertIsInstance(conv_module.write_section_predictor, dspy.Predict)

    def test_init_langchain(self):
        fake_lc_llm = FakeListLLM(responses=["lc response"])
        conv_module = ConvToSection(class_name='WriteSection', engine=fake_lc_llm, framework='langchain')
        self.assertEqual(conv_module.framework, 'langchain')
        self.assertIsInstance(conv_module.write_section_predictor, LangchainPredict)
        self.assertIsInstance(conv_module.write_section_predictor.signature, LangchainSignature)

    def test_forward_dspy(self):
        mock_dspy_lm = MockDSPyLM(responses=["Generated DSPy section content"])
        # Need to use dspy.settings.context with the mock LM for dspy.Predict
        dspy.settings.configure(lm=mock_dspy_lm)

        conv_module = ConvToSection(class_name='WriteSection', engine=mock_dspy_lm, framework='dspy')
        
        # The 'info' in WriteSection signature is a string, not a list of dicts.
        # ConvToSection's forward consolidates collected_info into a string.
        collected_info_list = [{"snippets": ["snippet1", "snippet2"], "source": "src1"}]
        
        result = conv_module.forward(
            topic="Test Topic", 
            outline="# Test Outline", 
            section="Test Section Title", 
            collected_info=collected_info_list, 
            language_style="formal"
        )
        self.assertIsInstance(result, dspy.Prediction)
        self.assertEqual(result.section, "Generated DSPy section content") # Assumes clean_up_section is somewhat idempotent for this test string

    def test_forward_langchain(self):
        fake_lc_llm = FakeListLLM(responses=["Generated LangChain section content"])
        conv_module = ConvToSection(class_name='WriteSection', engine=fake_lc_llm, framework='langchain')
        
        collected_info_list = [{"snippets": ["lc_snippet1", "lc_snippet2"], "source": "lc_src1"}]
        
        result = conv_module.forward(
            topic="LC Topic", 
            outline="# LC Outline", 
            section="LC Section Title", 
            collected_info=collected_info_list, 
            language_style="enthusiastic"
        )
        self.assertIsInstance(result, dict)
        self.assertIn("section", result)
        self.assertEqual(result["section"], "Generated LangChain section content")


class TestArticleGenerationModule(unittest.TestCase):
    def setUp(self):
        self.mock_retriever = MockRetriever()
        self.mock_mindmap = MockMindMap()
        self.mock_article_outline = MockArticleWithOutline()

    def test_init_dspy(self):
        mock_dspy_lm = MockDSPyLM()
        module = ArticleGenerationModule(
            retriever=self.mock_retriever,
            article_gen_lm=mock_dspy_lm,
            framework='dspy'
        )
        self.assertEqual(module.framework, 'dspy')
        self.assertIsInstance(module.section_gen, ConvToSection)
        self.assertEqual(module.section_gen.framework, 'dspy')

    def test_init_langchain(self):
        fake_lc_llm = FakeListLLM(responses=["lc response for agm"])
        module = ArticleGenerationModule(
            retriever=self.mock_retriever,
            article_gen_lm=fake_lc_llm,
            framework='langchain'
        )
        self.assertEqual(module.framework, 'langchain')
        self.assertIsInstance(module.section_gen, ConvToSection)
        self.assertEqual(module.section_gen.framework, 'langchain')
        self.assertIsInstance(module.section_gen.engine, BaseLLM) # Check if LLM is passed correctly

    def test_generate_section_langchain(self):
        section_content_response = "LangChain generated content for Introduction section."
        fake_lc_llm = FakeListLLM(responses=[section_content_response])
        
        module = ArticleGenerationModule(
            retriever=self.mock_retriever,
            article_gen_lm=fake_lc_llm,
            framework='langchain'
        )
        
        # Mock mindmap to control info passed to section_gen
        # generate_section calls mindmap.retrieve_information
        # ConvToSection (self.section_gen) receives this as collected_info
        # The LangchainPredict inside ConvToSection will then use fake_lc_llm
        
        mock_mindmap_lc = MockMindMap(retrieve_info_results=[
            {"snippets": ["info for intro"], "source": "test_source"}
        ])

        result_dict = module.generate_section(
            topic="AI in Healthcare",
            section_name="Introduction",
            mindmap=mock_mindmap_lc,
            section_query=["query for intro"], # Passed to mindmap.retrieve_information
            section_outline="# Introduction\n...", # Passed to self.section_gen
            language_style="formal"
        )

        self.assertIsInstance(result_dict, dict)
        self.assertEqual(result_dict["section_name"], "Introduction")
        self.assertEqual(result_dict["section_content"], section_content_response)
        self.assertEqual(result_dict["collected_info"][0]["snippets"][0], "info for intro")

    # generate_article is more complex due to threading and deeper object interactions.
    # A focused test on generate_section is more direct for unit testing the LangChain path.
    # To test generate_article, we would need to ensure the ThreadPoolExecutor works as expected
    # and that the MockArticleWithOutline correctly accumulates results.
    # For now, focusing on generate_section demonstrates the core LangChain integration.

if __name__ == '__main__':
    unittest.main()
