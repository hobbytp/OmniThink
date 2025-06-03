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
        # This method will be a MagicMock instance if not defined, or we can define it explicitly
        pass # If we need to assert it was called, assign MagicMock() in __init__


# Mock for ArticleWithOutline (or similar object used in generate_article)
# Updated to use MagicMock for easier testing of interactions
class MockArticleWithOutline:
    def __init__(self):
        self.get_first_level_section_names = MagicMock(return_value=[])
        self.get_outline_as_list = MagicMock(return_value="")
        self.update_section = MagicMock()
        self.post_processing = MagicMock()
        # Make it behave like a deepcopyable object if needed for the test
        # self.__deepcopy__ is a special method name, ensure it's handled correctly by mock
        # A common way is to have deepcopy return the same instance for mocks
        self.mock_deepcopy = MagicMock(return_value=self)

    def __deepcopy__(self, memo):
        # Ensure it returns the mock itself, helpful for copy.deepcopy in the code under test
        return self.mock_deepcopy(memo)
    
    def to_string(self): # If needed by any part of the code under test
        return "Mocked article content"


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
            # outline="# Test Outline", # Parameter removed
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
            # outline="# LC Outline", # Parameter removed
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
        # For generate_article tests, ensure MockMindMap has prepare_table_for_retrieval as a MagicMock
        self.mock_mindmap = MockMindMap()
        self.mock_mindmap.prepare_table_for_retrieval = MagicMock()
        
        # Use the MagicMock version of MockArticleWithOutline for generate_article tests
        self.mock_article_with_outline_magic = MockArticleWithOutline()


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

    def _run_generate_article_orchestration_test(self, framework_name, llm_instance):
        module = ArticleGenerationModule(
            retriever=self.mock_retriever,
            article_gen_lm=llm_instance,
            framework=framework_name
        )

        # Mock what generate_section would return
        module.generate_section = MagicMock(side_effect=[
            {"section_name": "Section 1", "section_content": "Content for S1", "collected_info": ["info1"]},
            {"section_name": "Section 2", "section_content": "Content for S2", "collected_info": ["info2"]},
        ])
        
        self.mock_article_with_outline_magic.get_first_level_section_names.return_value = ['Section 1', 'Section 2']
        self.mock_article_with_outline_magic.get_outline_as_list.return_value = ["# Section 1 outline", "## Subsection"]

        with patch('concurrent.futures.ThreadPoolExecutor') as mock_executor_class:
            mock_executor_instance = MagicMock()
            mock_executor_class.return_value.__enter__.return_value = mock_executor_instance

            # Make submit execute the function immediately and return a future with the result
            from concurrent.futures import Future
            def mock_submit(fn, *args, **kwargs):
                future = Future()
                try:
                    result = fn(*args, **kwargs)
                    future.set_result(result)
                except Exception as e:
                    future.set_exception(e)
                return future
            mock_executor_instance.submit.side_effect = mock_submit

            returned_article = module.generate_article(
                topic="Test Topic",
                mindmap=self.mock_mindmap,
                article_with_outline=self.mock_article_with_outline_magic,
                language_style={"style": "formal", "language_type": "English"}
            )

        self.mock_mindmap.prepare_table_for_retrieval.assert_called_once()
        self.assertEqual(module.generate_section.call_count, 2)
        
        # Check calls to generate_section
        expected_generate_section_calls = [
            unittest.mock.call(
                "Test Topic", "Section 1", self.mock_mindmap, ["# Section 1 outline", "## Subsection"], 
                "# Section 1 outline\n## Subsection", "formal English\n" 
            ),
            unittest.mock.call(
                "Test Topic", "Section 2", self.mock_mindmap, ["# Section 1 outline", "## Subsection"], 
                "# Section 1 outline\n## Subsection", "formal English\n"
            ),
        ]
        # Note: The query and outline passed to generate_section are identical for all sections in this mock setup of get_outline_as_list.
        # This might differ in real usage if get_outline_as_list is more dynamic.
        # For the test, we assert based on the mocked behavior.
        module.generate_section.assert_has_calls(expected_generate_section_calls, any_order=False)


        self.assertEqual(self.mock_article_with_outline_magic.update_section.call_count, 2)
        expected_update_calls = [
            unittest.mock.call(parent_section_name="Test Topic", current_section_content="Content for S1", current_section_info_list=["info1"]),
            unittest.mock.call(parent_section_name="Test Topic", current_section_content="Content for S2", current_section_info_list=["info2"]),
        ]
        self.mock_article_with_outline_magic.update_section.assert_has_calls(expected_update_calls, any_order=True) # Order can vary due to ThreadPool completion

        self.mock_article_with_outline_magic.post_processing.assert_called_once()
        self.assertEqual(returned_article, self.mock_article_with_outline_magic.mock_deepcopy.return_value)


    def test_generate_article_orchestration_dspy(self):
        mock_dspy_lm = MockDSPyLM()
        dspy.settings.configure(lm=mock_dspy_lm) # Configure for DSPy path if generate_section was not mocked
        self._run_generate_article_orchestration_test(framework_name='dspy', llm_instance=mock_dspy_lm)

    def test_generate_article_orchestration_langchain(self):
        fake_lc_llm = FakeListLLM(responses=["s1 content", "s2 content"]) # Responses for ConvToSection if generate_section not mocked
        self._run_generate_article_orchestration_test(framework_name='langchain', llm_instance=fake_lc_llm)


    def _run_generate_article_empty_sections_test(self, framework_name, llm_instance):
        module = ArticleGenerationModule(
            retriever=self.mock_retriever,
            article_gen_lm=llm_instance,
            framework=framework_name
        )
        module.generate_section = MagicMock() # Ensure it's not called

        self.mock_article_with_outline_magic.get_first_level_section_names.return_value = [] # No sections

        returned_article = module.generate_article(
            topic="Empty Topic",
            mindmap=self.mock_mindmap,
            article_with_outline=self.mock_article_with_outline_magic
        )

        self.mock_mindmap.prepare_table_for_retrieval.assert_called_once()
        module.generate_section.assert_not_called()
        self.mock_article_with_outline_magic.update_section.assert_not_called()
        self.mock_article_with_outline_magic.post_processing.assert_called_once()
        self.assertEqual(returned_article, self.mock_article_with_outline_magic.mock_deepcopy.return_value)

    def test_generate_article_empty_sections_dspy(self):
        mock_dspy_lm = MockDSPyLM()
        self._run_generate_article_empty_sections_test(framework_name='dspy', llm_instance=mock_dspy_lm)

    def test_generate_article_empty_sections_langchain(self):
        fake_lc_llm = FakeListLLM(responses=[])
        self._run_generate_article_empty_sections_test(framework_name='langchain', llm_instance=fake_lc_llm)


if __name__ == '__main__':
    unittest.main()
