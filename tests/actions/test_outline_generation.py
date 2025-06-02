import unittest
from unittest.mock import MagicMock, patch
import dspy

# Target modules
from src.actions.outline_generation import OutlineGenerationModule, WriteOutline, WritePageOutline, PolishPageOutline
from src.langchain_support.dspy_equivalents import LangchainPredict, LangchainSignature
from langchain.llms.fake import FakeListLLM
from langchain_core.language_models.llms import BaseLLM # For type hinting

# Mock for dspy.LM
class MockDSPyLM(dspy.dsp.LM):
    def __init__(self, responses=None):
        super().__init__("mock_model")
        self.responses = responses if responses else ["dspy lm default outline response"]
        self.history = [] # To track calls

    def __call__(self, prompt, only_completed=True, return_sorted=False, **kwargs):
        response = self.responses.pop(0) if self.responses else "dspy lm fallback outline"
        self.history.append({"prompt": prompt, "response": response, "kwargs": kwargs})
        return [response]

    def basic_request(self, prompt, **kwargs): # dspy.Predict calls this
        response = self.responses.pop(0) if self.responses else "dspy lm fallback outline basic_request"
        return response


# Mock for MindMap used in OutlineGenerationModule.generate_outline
class MockMindMap:
    def __init__(self, concepts_to_export="Concept 1\n- Sub-concept A\nConcept 2"):
        self.concepts_to_export = concepts_to_export

    def export_categories_and_concepts(self):
        return self.concepts_to_export


class TestWriteOutline(unittest.TestCase):
    def test_init_dspy(self):
        mock_dspy_lm = MockDSPyLM()
        # WritePageOutline and PolishPageOutline are classes in outline_generation
        write_outline_module = WriteOutline(engine=mock_dspy_lm, framework='dspy')
        self.assertEqual(write_outline_module.framework, 'dspy')
        self.assertIsInstance(write_outline_module.draft_page_outline_predictor, dspy.Predict)
        self.assertIsInstance(write_outline_module.polish_page_outline_predictor, dspy.Predict)

    def test_init_langchain(self):
        fake_lc_llm = FakeListLLM(responses=["lc draft outline", "lc polished outline"])
        write_outline_module = WriteOutline(engine=fake_lc_llm, framework='langchain')
        self.assertEqual(write_outline_module.framework, 'langchain')
        self.assertIsInstance(write_outline_module.draft_page_outline_predictor, LangchainPredict)
        self.assertIsInstance(write_outline_module.polish_page_outline_predictor, LangchainPredict)

    def test_forward_dspy(self):
        responses = ["Draft DSPy Outline", "Polished DSPy Outline"]
        mock_dspy_lm = MockDSPyLM(responses=list(responses)) # Pass a copy
        dspy.settings.configure(lm=mock_dspy_lm)

        write_outline_module = WriteOutline(engine=mock_dspy_lm, framework='dspy')
        
        result_outline = write_outline_module.forward(
            topic="Test DSPy Topic", 
            concepts="DSPy concepts"
        )
        # Assuming ArticleTextProcessing.clean_up_outline is somewhat transparent for these simple strings
        self.assertEqual(result_outline, "Polished DSPy Outline")
        self.assertEqual(len(mock_dspy_lm.history), 2) # Check if both predictors were called

    def test_forward_langchain(self):
        draft_response = "LangChain Draft Outline Content"
        polish_response = "LangChain Polished Outline Content"
        fake_lc_llm = FakeListLLM(responses=[draft_response, polish_response])
        
        write_outline_module = WriteOutline(engine=fake_lc_llm, framework='langchain')
        
        result_outline = write_outline_module.forward(
            topic="Test LC Topic", 
            concepts="LC concepts"
        )
        self.assertEqual(result_outline, "LangChain Polished Outline Content")
        # Check if the LLM was called twice (once for draft, once for polish)
        # FakeListLLM doesn't have a simple history, but we provided two responses and expect them to be used.


class TestOutlineGenerationModule(unittest.TestCase):
    def setUp(self):
        self.mock_mindmap = MockMindMap(concepts_to_export="Test concepts from mock mindmap")

    def test_init_dspy(self):
        mock_dspy_lm = MockDSPyLM()
        module = OutlineGenerationModule(
            outline_gen_lm=mock_dspy_lm,
            framework='dspy'
        )
        self.assertEqual(module.framework, 'dspy')
        self.assertIsInstance(module.write_outline, WriteOutline)
        self.assertEqual(module.write_outline.framework, 'dspy')

    def test_init_langchain(self):
        fake_lc_llm = FakeListLLM(responses=["lc outline response"]) # Needs two for WriteOutline
        module = OutlineGenerationModule(
            outline_gen_lm=fake_lc_llm,
            framework='langchain'
        )
        self.assertEqual(module.framework, 'langchain')
        self.assertIsInstance(module.write_outline, WriteOutline)
        self.assertEqual(module.write_outline.framework, 'langchain')
        self.assertIsInstance(module.write_outline.engine, BaseLLM)

    def test_generate_outline_langchain(self):
        draft_response = "LC Draft for Main Module"
        polish_response = "LC Polished for Main Module" # This is what should be returned
        fake_lc_llm = FakeListLLM(responses=[draft_response, polish_response])
        
        module = OutlineGenerationModule(
            outline_gen_lm=fake_lc_llm,
            framework='langchain'
        )
        
        result = module.generate_outline(
            topic="AI Ethics Outline",
            mindmap=self.mock_mindmap 
        )
        self.assertEqual(result, polish_response)


if __name__ == '__main__':
    unittest.main()
