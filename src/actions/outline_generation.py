import dspy
from src.tools.mindmap import MindMap # Restored import
# from unittest.mock import MagicMock # Removed temporary import
# MindMap = MagicMock() # Removed temporary mock

from src.utils.ArticleTextProcessing import ArticleTextProcessing
from typing import Union, Optional, Tuple
from src.langchain_support.dspy_equivalents import LangchainModule, LangchainSignature, LangchainPredict
from langchain_core.language_models.llms import BaseLLM # Added for type hinting

# This code is originally sourced from Repository STORM
# URL: [https://github.com/stanford-oval/storm]

class OutlineGenerationModule():

    def __init__(self,
                 outline_gen_lm: Union[dspy.LM, BaseLLM],
                 framework: str = 'dspy'):
        super().__init__()
        self.outline_gen_lm = outline_gen_lm
        self.framework = framework
        self.write_outline = WriteOutline(engine=self.outline_gen_lm, framework=self.framework)

    def generate_outline(self,
                         topic: str,
                         mindmap: MindMap,
                         ):

        concepts = mindmap.export_categories_and_concepts()
        result = self.write_outline(topic=topic, concepts=concepts)

        return result


class LangchainWritePageOutlineSignature(LangchainSignature):
    def __init__(self):
        super().__init__(
            input_fields=["topic"],
            output_fields=["outline"],
            prompt_template_str="""Write an outline for a Wikipedia page.
Here is the format of your writing:
1. Use "#" Title" to indicate section title, "##" Title" to indicate subsection title, "###" Title" to indicate subsubsection title, and so on.
2. Do not include other information.
3. Do not include topic name itself in the outline.

The topic you want to write: {topic}
Write the Wikipedia page outline:
"""
        )

class LangchainPolishPageOutlineSignature(LangchainSignature):
    def __init__(self):
        super().__init__(
            input_fields=["draft", "concepts"],
            output_fields=["outline"],
            prompt_template_str="""Improve an outline for a Wikipedia page. You already have a draft outline that covers the general information. Now you want to improve it based on the concept learned from an information-seeking to make it more informative.
Here is the format of your writing:
1. Use "#" Title" to indicate section title, "##" Title" to indicate subsection title, "###" Title" to indicate subsubsection title, and so on.
2. Do not include other information.
3. Do not include topic name itself in the outline.

Current outline:
{draft}

The information you learned from the conversation:
{concepts}

Write the page outline:
"""
        )

class WriteOutline(dspy.Module):
    """Generate the outline for the Wikipedia page."""

    def __init__(self, engine: Union[dspy.LM, BaseLLM], framework: str = 'dspy'):
        super().__init__()
        self.engine = engine
        self.framework = framework

        if self.framework == 'langchain':
            self.draft_page_outline = LangchainPredict(
                signature=LangchainWritePageOutlineSignature(),
                llm=self.engine
            )
            self.polish_page_outline = LangchainPredict(
                signature=LangchainPolishPageOutlineSignature(),
                llm=self.engine
            )
        else: # dspy
            self.draft_page_outline = dspy.Predict(WritePageOutline)
            self.polish_page_outline = dspy.Predict(PolishPageOutline)

    def forward(self, topic: str, concepts: str):
        if self.framework == 'langchain':
            draft_pred = self.draft_page_outline(topic=topic)
            cleaned_draft_outline = ArticleTextProcessing.clean_up_outline(draft_pred['outline'])

            polished_pred = self.polish_page_outline(draft=cleaned_draft_outline, concepts=concepts)
            final_outline = ArticleTextProcessing.clean_up_outline(polished_pred['outline'])

        else: # dspy
            with dspy.settings.context(lm=self.engine):
                draft_pred = self.draft_page_outline(topic=topic)
                cleaned_draft_outline = ArticleTextProcessing.clean_up_outline(draft_pred.outline)

                polished_pred = self.polish_page_outline(draft=cleaned_draft_outline, concepts=concepts)
                final_outline = ArticleTextProcessing.clean_up_outline(polished_pred.outline)

        return final_outline


class PolishPageOutline(dspy.Signature):
    """
    Improve an outline for a Wikipedia page. You already have a draft outline that covers the general information. Now you want to improve it based on the concept learned from an information-seeking to make it more informative.
    Here is the format of your writing:
    1. Use "#" Title" to indicate section title, "##" Title" to indicate subsection title, "###" Title" to indicate subsubsection title, and so on.
    2. Do not include other information.
    3. Do not include topic name itself in the outline.
    """

    draft = dspy.InputField(prefix="Current outline:\n ", format=str)
    concepts = dspy.InputField(prefix="The information you learned from the conversation:\n", format=str)
    outline = dspy.OutputField(prefix='Write the page outline:\n', format=str)


class WritePageOutline(dspy.Signature):
    """
    Write an outline for a Wikipedia page.
    Here is the format of your writing:
    1. Use "#" Title" to indicate section title, "##" Title" to indicate subsection title, "###" Title" to indicate subsubsection title, and so on.
    2. Do not include other information.
    3. Do not include topic name itself in the outline.
    """

    topic = dspy.InputField(prefix="The topic you want to write: ", format=str)
    outline = dspy.OutputField(prefix="Write the Wikipedia page outline:\n", format=str)

# Ensure this file ends with a newline character for POSIX compliance if necessary
