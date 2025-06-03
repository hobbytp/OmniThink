import concurrent.futures
import copy
import logging
import random
from concurrent.futures import as_completed
from typing import List, Union
import random
import dspy
import sys
from src.utils.ArticleTextProcessing import ArticleTextProcessing
from src.langchain_support.dspy_equivalents import LangchainModule, LangchainSignature, LangchainPredict
from langchain_core.language_models.llms import BaseLLM # Added for type hinting



# This code is originally sourced from Repository STORM
# URL: [https://github.com/stanford-oval/storm]
class ArticleGenerationModule():
    """
    The interface for article generation stage. Given topic, collected information from
    knowledge curation stage, generated outline from outline generation stage, 
    """

    def __init__(self,
                 retriever,
                 article_gen_lm=Union[dspy.LM, BaseLLM], # Use BaseLLM for LangChain, dspy.LM for DSPy. Removed dspy.HFModel.
                 retrieve_top_k: int = 10,
                 max_thread_num: int = 10,
                 agent_name: str = 'WriteSection',
                 framework: str = 'dspy'
                 ):
        super().__init__()
        self.retrieve_top_k = retrieve_top_k
        self.article_gen_lm = article_gen_lm
        self.max_thread_num = max_thread_num
        self.retriever = retriever
        self.framework = framework
        self.section_gen = ConvToSection(engine=self.article_gen_lm, class_name=agent_name, framework=self.framework)

    def generate_section(self, topic, section_name, mindmap, section_query, section_outline, language_style):
        collected_info = mindmap.retrieve_information(queries=section_query,
                                                      search_top_k=self.retrieve_top_k)
        # ConvToSection's output will be a dspy.Prediction or dict based on framework
        output_pred = self.section_gen(
            topic=topic,
            # outline=section_outline, # outline parameter removed from ConvToSection.forward
            section=section_name,
            collected_info=collected_info,
            language_style=language_style,
        )

        if self.framework == 'langchain':
            section_content = output_pred['section']
        else: # dspy
            section_content = output_pred.section

        return {"section_name": section_name, "section_content": section_content, "collected_info": collected_info}

    def generate_article(self,
                         topic: str,
                         mindmap,
                         article_with_outline,
                         language_style=None,
                         ):
        """
        Generate article for the topic based on the information table and article outline.
        """
        mindmap.prepare_table_for_retrieval()
        language_style = "{} {}\n".format(language_style.get('style', ''),
                                          language_style.get('language_type', '')) if language_style else str()

        sections_to_write = article_with_outline.get_first_level_section_names()
        section_output_dict_collection = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_thread_num) as executor:
            future_to_sec_title = {}
            for section_title in sections_to_write:
                section_query = article_with_outline.get_outline_as_list(
                    root_section_name=section_title, add_hashtags=False
                )
                queries_with_hashtags = article_with_outline.get_outline_as_list(
                    root_section_name=section_title, add_hashtags=True
                )
                section_outline = "\n".join(queries_with_hashtags)

                future_to_sec_title[
                    executor.submit(self.generate_section,
                                    topic, section_title, mindmap, section_query, section_outline, language_style)
                ] = section_title

            for future in concurrent.futures.as_completed(future_to_sec_title):
                section_output_dict_collection.append(future.result())

        article = copy.deepcopy(article_with_outline)
        for section_output_dict in section_output_dict_collection:
            article.update_section(parent_section_name=topic,
                                   current_section_content=section_output_dict["section_content"],
                                   current_section_info_list=section_output_dict["collected_info"],
                                   )

        article.post_processing()

        return article


class LangchainWriteSectionSignature(LangchainSignature):
    def __init__(self):
        super().__init__(
            input_fields=['topic', 'info', 'section', 'language_style'],
            output_fields=['output'],
            prompt_template_str="""Write a Wikipedia section based on the collected information.

Here is the format of your writing:
    1. Use "#" Title" to indicate section title, "##" Title" to indicate subsection title, "###" Title" to indicate subsubsection title, and so on.
    2. Use [1], [2], ..., [n] in line (for example, "The capital of the United States is Washington, D.C.[1][3]."). You DO NOT need to include a References or Sources section to list the sources at the end.
    3. The language style should resemble that of Wikipedia: concise yet informative, formal yet accessible.

The Collected information:
{info}

The topic of the page: {topic}
The section you need to write: {section}
The language style you needs to imitate: {language_style}

Write the section with proper inline citations (Start your writing with # section title. Don't include the page title or try to write other sections):
"""
        )

class ConvToSection(dspy.Module): # Can also be LangchainModule if we refactor further, but dspy.Module is fine for now
    """Use the information collected from the information-seeking conversation to write a section."""

    def __init__(self, class_name: str, engine: Union[dspy.LM, BaseLLM], framework: str = 'dspy'):
        super().__init__()
        self.framework = framework
        self.engine = engine # This is the LLM
        # self.class_name = class_name # Not strictly needed if Langchain path doesn't use it.

        if self.framework == 'langchain':
            # LangchainWriteSectionSignature is defined in this file.
            self.write_section = LangchainPredict(
                signature=LangchainWriteSectionSignature(),
                llm=self.engine
            )
        else: # dspy
            current_module = globals()
            if class_name in current_module and issubclass(current_module.get(class_name), dspy.Signature):
                cls = current_module.get(class_name)
                self.write_section = dspy.Predict(cls) # Use 'write_section' to match example
            else:
                raise ValueError(f"Class '{class_name}' not found or not a dspy.Signature!")

    def forward(self, topic: str, section: str, collected_info: List, language_style: str):
        # `section` here is the section title/name
        # The instruction asks to revert to looping through collected_info and calling predictor for each.
        # This implies iterative refinement or processing of the section content.
        # `processed_section_content` will hold the content from the last iteration.

        processed_section_content = "" # Initialize or carry over if section is refined.
                                     # For now, assume each call generates content for the given 'section' (title)
                                     # based on one 'info_item'. The last one wins.

        for idx, info_item in enumerate(collected_info):
            # Reconstruct info_str for each item
            # The original WriteSection signature has `info` as InputField, not `info_item` or `consolidated_info_str`
            # This was a point of confusion in my previous reasoning.
            # The `info` field in the signature is `dspy.InputField(prefix="The Collected information:\n", format=str)`
            # This suggests it expects *all* info.
            # However, the prompt example in instructions for `forward` implies looping.
            # Let's stick to the loop as per current instruction for `forward`.

            current_info_str = f'[{idx + 1}]\n' + '\n'.join(info_item['snippets']) + '\n\n'
            current_info_str = ArticleTextProcessing.limit_word_count_preserve_newline(current_info_str, 1500)

            if self.framework == 'langchain':
                prediction_output = self.write_section( # use self.write_section
                    topic=topic,
                    info=current_info_str, # Pass the current info item's string
                    section=section, # section title/name
                    language_style=language_style
                )
                current_segment = prediction_output['output']
            else: # dspy
                with dspy.settings.context(lm=self.engine):
                    prediction_output = self.write_section( # use self.write_section
                        topic=topic,
                        info=current_info_str, # Pass the current info item's string
                        section=section, # section title/name
                        language_style=language_style
                    )
                    current_segment = prediction_output.output

            # The original code re-assigned 'section' (the content) in the loop.
            # We use 'processed_section_content' to store the latest version.
            processed_section_content = ArticleTextProcessing.clean_up_section(current_segment)

        # Final cleanup after the loop (using the content from the last iteration)
        final_section_content = processed_section_content.replace('\[', '[').replace('\]', ']')
        # Return type should be consistent: a dspy.Prediction for dspy, and a dict for LangChain.
        if self.framework == 'dspy':
            return dspy.Prediction(section=final_section_content)
        else:
            return {"section": final_section_content}


class WriteSection(dspy.Signature):
    """Write a Wikipedia section based on the collected information.

    Here is the format of your writing:
        1. Use "#" Title" to indicate section title, "##" Title" to indicate subsection title, "###" Title" to indicate subsubsection title, and so on.
        2. Use [1], [2], ..., [n] in line (for example, "The capital of the United States is Washington, D.C.[1][3]."). You DO NOT need to include a References or Sources section to list the sources at the end.
        3. The language style should resemble that of Wikipedia: concise yet informative, formal yet accessible.
    """
    info = dspy.InputField(prefix="The Collected information:\n", format=str)
    topic = dspy.InputField(prefix="The topic of the page: ", format=str)
    section = dspy.InputField(prefix="The section you need to write: ", format=str)
    language_style = dspy.InputField(prefix='the language style you needs to imitate: ', format=str)
    output = dspy.OutputField(
        prefix="Write the section with proper inline citations (Start your writing with # section title. Don't include the page title or try to write other sections):\n",
        format=str)


class WriteSectionAgentEnglish(dspy.Signature):
    """Generate an English Wikipedia section with formal yet accessible tone, adhering to standard Wikipedia guidelines.

    Writing specifications:
        1. Use "#" Title" to indicate section title, "##" Title" to indicate subsection title, "###" Title" to indicate subsubsection title, and so on.
        2. Use [1], [2], ..., [n] in line (for example, "The capital of the United States is Washington, D.C.[1][3].").
        3. Maintain formal yet accessible English language style
        4. Follow standard English Wikipedia formatting and style guidelines
        5. Ensure clarity and readability for international English readers
    """
    info = dspy.InputField(prefix="Collected source materials:\n", format=str)
    topic = dspy.InputField(prefix="Article topic: ", format=str)
    section = dspy.InputField(prefix="Target section to write: ", format=str)
    language_style = dspy.InputField(prefix='Target writing style (formal English): ', format=str)
    output = dspy.OutputField(
        prefix="Generate the English section with proper inline citations (start with # section title, exclude page header):\n",
        format=str)


class WriteSectionAgentChinese(dspy.Signature):
    """Generate a Chinese Wikipedia section adhering to standard formatting and style guidelines.

    Writing specifications:
        1. Use "#" Title" to indicate section title, "##" Title" to indicate subsection title, "###" Title" to indicate subsubsection title, and so on.
        2. Use [1], [2], ..., [n] in line (for example, "The capital of the United States is Washington, D.C.[1][3].").
        3. Maintain formal yet accessible Chinese language style
        4. Follow standard Chinese Wikipedia formatting and style guidelines
        5. Use Simplified Chinese characters and proper punctuation
    """
    info = dspy.InputField(prefix="Collected source materials:\n", format=str)
    topic = dspy.InputField(prefix="Article topic: ", format=str)
    section = dspy.InputField(prefix="Target section to write: ", format=str)
    language_style = dspy.InputField(prefix='Target writing style (formal Chinese): ', format=str)
    output = dspy.OutputField(
        prefix="Generate the Chinese section with proper inline citations (start with # section title, exclude page header):\n",
        format=str)


class WriteSectionAgentFormalChinese(dspy.Signature):
    """Generate a formal Chinese Wikipedia section based on the collected information.

    Writing specifications:
    1. Use "#" Title" to indicate section title, "##" Title" to indicate subsection title, "###" Title" to indicate subsubsection title, and so on.
    2. Use [1], [2], ..., [n] in line (for example, "The capital of the United States is Washington, D.C.[1][3].").
    3. Maintain formal and professional Chinese language style
    4. Follow Wikipedia's neutral tone and encyclopedic writing standards
    5. Use standard Simplified Chinese characters and proper punctuation
    """
    info = dspy.InputField(prefix="Collected source materials:\n", format=str)
    topic = dspy.InputField(prefix="Article topic: ", format=str)
    section = dspy.InputField(prefix="Target section to write: ", format=str)
    language_style = dspy.InputField(prefix='Target writing style (formal Chinese): ', format=str)
    output = dspy.OutputField(
        prefix="Generate the formal Chinese section with proper inline citations (start with # section title, exclude table of contents):\n",
        format=str)


class WriteSectionAgentEnthusiasticChinese(dspy.Signature):
    """Generate an engaging Chinese Wikipedia section with enthusiastic tone while maintaining factual accuracy.

    Writing specifications:
        1. Use "#" Title" to indicate section title, "##" Title" to indicate subsection title, "###" Title" to indicate subsubsection title, and so on.
        2. Use [1], [2], ..., [n] in line (for example, "The capital of the United States is Washington, D.C.[1][3].").
        3. Employ lively yet professional Chinese language style
        4. Maintain Wikipedia's neutral point of view while using engaging expressions
        5. Use appropriate rhetorical devices to enhance readability
    """
    info = dspy.InputField(prefix="Collected source materials:\n", format=str)
    topic = dspy.InputField(prefix="Article topic: ", format=str)
    section = dspy.InputField(prefix="Target section to write: ", format=str)
    language_style = dspy.InputField(prefix='Target writing style (enthusiastic Chinese): ', format=str)
    output = dspy.OutputField(
        prefix="Generate the enthusiastic Chinese section with proper inline citations (start with # section title, maintain engaging tone):\n",
        format=str)


class WriteSectionAgentEnthusiasticEnglish(dspy.Signature):
    """Generate an engaging English Wikipedia section with enthusiastic tone while maintaining factual accuracy.

    Writing specifications:
        1. Use "#" Title" to indicate section title, "##" Title" to indicate subsection title, "###" Title" to indicate subsubsection title, and so on.
        2. Use [1], [2], ..., [n] in line (for example, "The capital of the United States is Washington, D.C.[1][3].").
        3. Employ lively yet professional English language style
        4. Maintain Wikipedia's neutral point of view while using engaging expressions
        5. Use appropriate rhetorical devices to enhance readability
        6. Follow standard English Wikipedia formatting guidelines
    """
    info = dspy.InputField(prefix="Collected source materials:\n", format=str)
    topic = dspy.InputField(prefix="Article topic: ", format=str)
    section = dspy.InputField(prefix="Target section to write: ", format=str)
    language_style = dspy.InputField(prefix='Target writing style (enthusiastic English): ', format=str)
    output = dspy.OutputField(
        prefix="Generate the enthusiastic English section with proper inline citations (start with # section title, maintain engaging tone):\n",
        format=str)


class WriteSectionAgentFormalEnglish(dspy.Signature):
    """Generate a formal English Wikipedia section adhering to standard formatting and style guidelines.

    Writing specifications:
        1. Use "#" Title" to indicate section title, "##" Title" to indicate subsection title, "###" Title" to indicate subsubsection title, and so on.
        2. Use [1], [2], ..., [n] in line (for example, "The capital of the United States is Washington, D.C.[1][3].").
        3. Maintain formal and professional English language style
        4. Follow standard English Wikipedia formatting and style guidelines
        5. Ensure clarity and readability for international English readers
    """
    info = dspy.InputField(prefix="Collected source materials:\n", format=str)
    topic = dspy.InputField(prefix="Article topic: ", format=str)
    section = dspy.InputField(prefix="Target section to write: ", format=str)
    language_style = dspy.InputField(prefix='Target writing style (formal English): ', format=str)
    output = dspy.OutputField(
        prefix="Generate the formal English section with proper inline citations (start with # section title, exclude page header):\n",
        format=str)


if __name__ == '__main__':
    print("--- Demonstrating ArticleGenerationModule with LangChain (Conceptual) ---")

    # Import the LLM configuration utility
    from src.langchain_support.llm_config import get_langchain_llm
    from src.tools.retriever import execute_search_queries # Dummy retriever for instantiation

    # 1. Instantiate a LangChain LLM (using FakeListLLM for this example)
    print("\n1. Instantiating LangChain LLM (FakeListLLM)...")
    try:
        # These responses should align with what LangchainWriteSectionSignature expects if we were to call it.
        # For example, the 'output' field of the signature.
        fake_llm_responses = [
            "Generated section content for topic 'AI Ethics', section 'Introduction'.",
            "Another generated section for a different call.",
        ]
        langchain_llm = get_langchain_llm(provider='fake', responses=fake_llm_responses)
        print(f"LangChain LLM instantiated: {langchain_llm}")
    except Exception as e:
        print(f"Error instantiating FakeListLLM: {e}")
        langchain_llm = None

    # (Optional) Example for OpenAI - requires OPENAI_API_KEY environment variable
    # print("\n(Optional) Trying to instantiate ChatOpenAI...")
    # try:
    #     openai_llm = get_langchain_llm(provider='openai', model_name='gpt-3.5-turbo')
    #     print(f"ChatOpenAI LLM instantiated: {openai_llm}")
    # except Exception as e:
    #     print(f"Could not instantiate ChatOpenAI (this is expected if API key is not set): {e}")
    #     openai_llm = None

    if langchain_llm:
        print("\n2. Instantiating ArticleGenerationModule with LangChain LLM...")

        # For ArticleGenerationModule, a retriever is needed.
        # We'll use a placeholder or a very simple dummy retriever for this conceptual demonstration.
        # In a real scenario, this would be a properly configured retriever instance.
        class DummyRetriever:
            def __init__(self):
                print("DummyRetriever initialized.")
            def search(self, query: str, k: int):
                print(f"DummyRetriever: Searching for '{query}' with k={k}")
                return [{"text": f"Dummy search result for {query}"}]
            # Add other methods if ArticleGenerationModule's direct usage requires them,
            # e.g. if it calls specific retriever methods during __init__ or simple calls.
            # For now, assuming it's primarily passed along.

        dummy_retriever_instance = DummyRetriever()

        try:
            # Using the FakeListLLM for this example
            article_gen_module_lc = ArticleGenerationModule(
                retriever=dummy_retriever_instance, # Pass the dummy/placeholder retriever
                article_gen_lm=langchain_llm,       # Pass the instantiated LangChain LLM
                framework='langchain',              # Specify the framework
                agent_name='WriteSection'           # Default agent_name
            )
            print(f"ArticleGenerationModule (LangChain) instantiated: {article_gen_module_lc}")
            print(f"  - Framework: {article_gen_module_lc.framework}")
            print(f"  - LLM: {article_gen_module_lc.article_gen_lm}")
            print(f"  - Section Generator (ConvToSection): {article_gen_module_lc.section_gen}")
            print(f"    - ConvToSection Framework: {article_gen_module_lc.section_gen.framework}")
            print(f"    - ConvToSection Predictor: {article_gen_module_lc.section_gen.write_section_predictor}")

            print("\n3. Conceptual method call (not executing full logic due to complex inputs):")
            print("   - To fully run `generate_section` or `generate_article`, MindMap and Article/Outline objects are needed.")
            print("   - Example: article_gen_module_lc.generate_section(topic='AI Ethics', section_name='Introduction', ...)")
            print("   - The key demonstration here is the setup with a LangChain LLM.")

        except Exception as e:
            print(f"Error instantiating or inspecting ArticleGenerationModule with LangChain: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("\nSkipping ArticleGenerationModule instantiation as LangChain LLM was not created.")

    print("\n--- End of Demonstration ---")