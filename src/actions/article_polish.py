import copy
from typing import Union
import dspy
from src.utils.ArticleTextProcessing import ArticleTextProcessing
from src.langchain_support.dspy_equivalents import LangchainModule, LangchainSignature, LangchainPredict
from langchain_core.language_models.llms import BaseLLM # Added for type hinting

# This code is originally sourced from Repository STORM
# URL: [https://github.com/stanford-oval/storm]
class ArticlePolishingModule():
    """
    The interface for article generation stage. Given topic, collected information from
    knowledge curation stage, generated outline from outline generation stage.
    """

    def __init__(self,
                 # article_gen_lm: Union[dspy.LM, BaseLLM], # Removed as it was unused
                 article_polish_lm: Union[dspy.LM, BaseLLM], # Use BaseLLM for LangChain, dspy.LM for DSPy. Removed dspy.HFModel.
                 framework: str = 'dspy'):
        # self.article_gen_lm = article_gen_lm # Removed
        self.article_polish_lm = article_polish_lm
        self.framework = framework

        self.polish_page_module = PolishPageModule(
            # write_lead_engine parameter removed from PolishPageModule
            polish_engine=self.article_polish_lm,
            framework=self.framework
        )

    def polish_article(self,
                       topic: str, # topic is passed to polish_page_module, but not used in its forward
                       draft_article,
                       remove_duplicate: bool = False): # remove_duplicate maps to polish_whole_page
        """
        Polish article.

        Args:
            topic (str): The topic of the article.
            draft_article (StormArticle): The draft article.
            remove_duplicate (bool): Whether to use one additional LM call to remove duplicates from the article.
        """

        article_text = draft_article.to_string()
        # remove_duplicate = True # This was hardcoded, let's use the parameter

        # polish_page_module returns a dict for langchain, dspy.Prediction for dspy
        polish_result_pred = self.polish_page_module(
            topic=topic, # Passed to forward, though not used by current PolishPageSignature
            draft_page=article_text,
            polish_whole_page=remove_duplicate
        )

        if self.framework == 'langchain':
            polished_article_text = polish_result_pred['page']
        else: # dspy
            polished_article_text = polish_result_pred.page

        polished_article_dict = ArticleTextProcessing.parse_article_into_dict(polished_article_text)
        polished_article_obj = copy.deepcopy(draft_article) # Keep variable name distinct
        polished_article_obj.insert_or_create_section(article_dict=polished_article_dict)
        polished_article_obj.post_processing()
        return polished_article_obj


class LangchainPolishPageSignature(LangchainSignature):
    def __init__(self):
        super().__init__(
            input_fields=["article"], # 'topic' and 'polish_whole_page' are not in dspy.PolishPage signature
            output_fields=["page"],
            prompt_template_str="""You are a faithful text editor that is good at finding repeated information in the article and deleting them to make sure there is no repetition in the article. You won't delete any non-repeated part in the article. You will keep the inline citations and article structure (indicated by "#", "##", etc.) appropriately. Do your job for the following article.

The article you need to polish:
{article}

Your revised article:
"""
        )

class PolishPage(dspy.Signature):
    """You are a faithful text editor that is good at finding repeated information in the article and deleting them to make sure there is no repetition in the article. You won't delete any non-repeated part in the article. You will keep the inline citations and article structure (indicated by "#", "##", etc.) appropriately. Do your job for the following article."""
    article = dspy.InputField(prefix="The article you need to polish:\n", format=str)
    page = dspy.OutputField( # Corresponds to 'page' in LangchainSignature output_fields
        prefix="Your revised article:\n",
        format=str)


class PolishPageModule(dspy.Module): # Can also be LangchainModule
    def __init__(self,
                 # write_lead_engine: Union[dspy.LM, BaseLLM], # Removed as it was unused
                 polish_engine: Union[dspy.LM, BaseLLM], # Use BaseLLM for LangChain, dspy.LM for DSPy. Removed dspy.HFModel.
                 framework: str = 'dspy'):
        super().__init__()
        # self.write_lead_engine = write_lead_engine # Removed
        self.polish_engine = polish_engine
        self.framework = framework

        if self.framework == 'langchain':
            self.polish_page_predictor = LangchainPredict(
                signature=LangchainPolishPageSignature(),
                llm=self.polish_engine
            )
        else: # dspy
            self.polish_page_predictor = dspy.Predict(PolishPage)

    def forward(self, topic: str, draft_page: str, polish_whole_page: bool = True):
        # topic and polish_whole_page are not used by the dspy.PolishPage signature
        # or the LangchainPolishPageSignature as defined (which matches PolishPage).
        # They are kept in the method signature for interface consistency with ArticlePolishingModule.

        if self.framework == 'langchain':
            # dspy.settings.context might not be relevant for LangChain LLMs
            # Assuming self.polish_engine is a LangChain LLM here.
            prediction = self.polish_page_predictor(article=draft_page)
            page_content = prediction['page']
            return {"page": page_content}
        else: # dspy
            with dspy.settings.context(lm=self.polish_engine):
                prediction = self.polish_page_predictor(article=draft_page)
                page_content = prediction.page
            return dspy.Prediction(page=page_content)


