import unittest
from unittest.mock import MagicMock

from src.langchain_support.dspy_equivalents import LangchainSignature, LangchainPredict, LangchainModule
from langchain.llms.fake import FakeListLLM
from langchain_core.language_models.llms import BaseLLM

class TestLangchainSignature(unittest.TestCase):
    def test_instantiation_and_attributes(self):
        input_fields = ["question", "context"]
        output_fields = ["answer"]
        prompt_template_str = "Q: {question} C: {context} A:"

        sig = LangchainSignature(
            input_fields=input_fields,
            output_fields=output_fields,
            prompt_template_str=prompt_template_str
        )

        self.assertEqual(sig.input_fields, input_fields)
        self.assertEqual(sig.output_fields, output_fields)
        self.assertEqual(sig.prompt_template_str, prompt_template_str)
        self.assertEqual(repr(sig), f"LangchainSignature(input_fields={input_fields}, output_fields={output_fields})")

    def test_default_prompt_template(self):
        sig = LangchainSignature(input_fields=["text"], output_fields=["summary"])
        self.assertEqual(sig.prompt_template_str, "")


class TestLangchainPredict(unittest.TestCase):
    def setUp(self):
        self.fake_llm = FakeListLLM(responses=["default response"])
        self.sig_single_output = LangchainSignature(
            input_fields=["query"],
            output_fields=["result"],
            prompt_template_str="Query: {query} Result:"
        )
        self.sig_multi_output = LangchainSignature(
            input_fields=["document"],
            output_fields=["title", "summary"],
            prompt_template_str="Doc: {document} Title: Summary:"
        )

    def test_init_success(self):
        try:
            predictor = LangchainPredict(signature=self.sig_single_output, llm=self.fake_llm)
            self.assertIsNotNone(predictor.chain)
            self.assertIsNotNone(predictor.prompt_template)
        except Exception as e:
            self.fail(f"LangchainPredict initialization failed: {e}")

    def test_init_invalid_signature(self):
        with self.assertRaises(ValueError):
            LangchainPredict(signature="not a signature", llm=self.fake_llm)

    def test_init_invalid_llm(self):
        with self.assertRaises(ValueError):
            LangchainPredict(signature=self.sig_single_output, llm="not an llm")

    def test_init_no_prompt_template(self):
        sig_no_prompt = LangchainSignature(input_fields=["in"], output_fields=["out"])
        with self.assertRaises(ValueError):
            LangchainPredict(signature=sig_no_prompt, llm=self.fake_llm)

    def test_call_single_output_string_response(self):
        responses = ["Test answer"]
        llm = FakeListLLM(responses=responses)
        predictor = LangchainPredict(signature=self.sig_single_output, llm=llm)

        result = predictor(query="What is testing?")

        self.assertIn("result", result)
        self.assertEqual(result["result"], responses[0])

    def test_call_single_output_dict_response(self):
        # LLMChain usually returns a dict like {'text': 'response'}
        # Our FakeListLLM returns strings directly, which LLMChain wraps.
        responses = ["Test answer from dict path"]
        llm = FakeListLLM(responses=responses) # FakeListLLM makes LLMChain return {'text': 'response'}

        predictor = LangchainPredict(signature=self.sig_single_output, llm=llm)
        result = predictor(query="What is testing?")

        self.assertIn("result", result)
        # The LLMChain with FakeListLLM will output a dict: {'text': "Test answer from dict path"}
        # Our LangchainPredict.__call__ should extract this.
        self.assertEqual(result["result"], responses[0])


    def test_call_multi_output_string_response(self):
        responses = ["This is a title and summary combined."]
        llm = FakeListLLM(responses=responses)
        predictor = LangchainPredict(signature=self.sig_multi_output, llm=llm)

        result = predictor(document="Some long document.")

        self.assertIn("title", result)
        self.assertIn("summary", result)
        self.assertEqual(result["title"], responses[0]) # First output field gets the string
        self.assertEqual(result["summary"], "")          # Subsequent fields are empty

    def test_call_multi_output_dict_response_matching_keys(self):
        # This scenario is harder to mock with FakeListLLM directly, as it returns strings
        # that LLMChain wraps into {'text': 'string'}.
        # To test this properly, we'd need a mock LLM that returns a dict itself.
        # For now, we acknowledge this based on the implementation: if raw_prediction is a dict
        # with matching keys, they will be used.
        # Let's simulate chain returning a dict with correct keys

        mock_chain = MagicMock()
        mock_chain.invoke = MagicMock(return_value={"title": "Test Title", "summary": "Test Summary"})

        predictor = LangchainPredict(signature=self.sig_multi_output, llm=self.fake_llm)
        predictor.chain = mock_chain # Override the chain with our mock

        result = predictor(document="Some long document.")
        self.assertEqual(result["title"], "Test Title")
        self.assertEqual(result["summary"], "Test Summary")

    def test_call_multi_output_dict_response_text_key(self):
        # Simulate LLMChain returning {'text': 'some string'} when multiple outputs are expected
        responses = ["Title and summary in one string"]
        llm = FakeListLLM(responses=responses) # This will lead to {'text': responses[0]}

        predictor = LangchainPredict(signature=self.sig_multi_output, llm=llm)
        result = predictor(document="Test doc")

        # Current logic: if raw_prediction is a dict and multiple output fields, it tries direct key mapping.
        # If 'text' is in raw_prediction but not an output_field, it won't be automatically split.
        # The first output field will try to get raw_prediction['title'] (empty), etc.
        # This part of the logic in __call__ for multiple outputs from a single 'text' field might need refinement
        # if automatic splitting was intended. Current code: `output_dict[field] = raw_prediction.get(field, "")`
        # self.assertEqual(result["title"], "") # This was the old expectation
        # self.assertEqual(result["summary"], "")
        # If the intent was for the 'text' field to be assigned to the first output_field:
        self.assertEqual(result["title"], responses[0]) # New expectation: first field gets 'text'
        self.assertEqual(result["summary"], "")     # Second field remains empty as 'summary' key is not in raw_prediction
        # This test highlights that for multiple outputs, structured dict from LLM is better.

    def test_call_missing_input_field(self):
        predictor = LangchainPredict(signature=self.sig_single_output, llm=self.fake_llm)
        with self.assertRaises(ValueError) as context:
            predictor(wrong_input_name="Test")
        self.assertIn("Missing input field 'query'", str(context.exception))


class TestLangchainModule(unittest.TestCase):
    class SimpleModule(LangchainModule):
        def __init__(self, llm: BaseLLM = None, **kwargs):
            super().__init__(llm, **kwargs)
            self.some_value = kwargs.get("some_value", 0)

        def forward(self, x: int) -> int:
            return x * 2 + self.some_value

        def another_method(self, y:int) -> int:
            if self.llm: # Just to use self.llm
                return y + 1
            return y


    def test_module_init_with_llm_and_kwargs(self):
        mock_llm = FakeListLLM(responses=[])
        module = self.SimpleModule(llm=mock_llm, some_config="test", some_value=5)
        self.assertEqual(module.llm, mock_llm)
        self.assertTrue(hasattr(module, "some_config"))
        self.assertEqual(module.some_config, "test")
        self.assertEqual(module.some_value, 5)
        self.assertEqual(module.another_method(10), 11)


    def test_module_call_invokes_forward(self):
        module = self.SimpleModule(some_value=3)
        result = module(10) # Invokes __call__ -> forward
        self.assertEqual(result, 23) # 10 * 2 + 3

    def test_module_init_without_llm(self):
        module = self.SimpleModule(some_value=1)
        self.assertIsNone(module.llm)
        self.assertEqual(module.another_method(5), 5)
        self.assertEqual(module(5),11) # 5*2+1


if __name__ == '__main__':
    unittest.main()
