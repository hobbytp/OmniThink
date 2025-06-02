import os
import sys
from argparse import ArgumentParser
import yaml # Added
from src.tools.lm import DeepSeekChatModel, DeepSeekReasonerModel # Updated
from src.tools.rm import GoogleSearchAli
from src.tools.mindmap import MindMap
from src.actions.outline_generation import OutlineGenerationModule
from src.dataclass.Article import Article
from src.actions.article_generation import ArticleGenerationModule
from src.actions.article_polish import ArticlePolishingModule

def load_config(config_path: str) -> dict:
    try:
        with open(config_path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
        return config if config else {}
    except FileNotFoundError:
        print(f"Info: Config file not found at {config_path}. Using defaults and command-line args.")
        return {}
    except yaml.YAMLError as e:
        print(f"Warning: Error parsing config file {config_path}: {e}. Using defaults and command-line args.")
        return {}

def main(args, parser_obj): # Pass parser_obj to access defaults
    config_data = load_config(args.cfg)
    ds_config = config_data.get('deepseekr1_settings', {})

    # Resolve framework choice first
    framework_choice = args.framework # From CLI
    if framework_choice is None: # If not set by CLI
        framework_choice = config_data.get('framework', 'dspy') # Try config (from root of config_data), else default 'dspy'

    print(f"[*] Using framework: {framework_choice}")
    # setattr(args, 'selected_framework', framework_choice) # Optional: store it back

    # Resolve settings with precedence: CLI > Config > Argparse default
    outputdir_resolved = args.outputdir
    if args.outputdir == parser_obj.get_default('outputdir'):
        outputdir_resolved = ds_config.get('outputdir', args.outputdir)

    threadnum_resolved = args.threadnum
    if args.threadnum == parser_obj.get_default('threadnum'):
        threadnum_resolved = ds_config.get('threadnum', args.threadnum)

    retriever_resolved = args.retriever
    if args.retriever is None: # No default in argparse, so check if None
        retriever_resolved = ds_config.get('retriever', args.retriever)

    retrievernum_resolved = args.retrievernum
    if args.retrievernum == parser_obj.get_default('retrievernum'):
        retrievernum_resolved = ds_config.get('retrievernum', args.retrievernum)

    llm_resolved = args.llm
    if args.llm is None: # No default in argparse
        llm_resolved = ds_config.get('llm', args.llm)

    depth_resolved = args.depth
    if args.depth == parser_obj.get_default('depth'):
        depth_resolved = ds_config.get('depth', args.depth)

    # Use resolved values
    kwargs = {
        'temperature': 1.0, # These are not in config currently
        'top_p': 0.9,
    }

    rm = None # Initialize rm
    if retriever_resolved == 'google': # Use resolved value
        rm = GoogleSearchAli(k=retrievernum_resolved) # Use resolved value

    # Use DeepSeekChatModel by default or based on llm_resolved
    # Fallback to "deepseek-chat" if llm_resolved is still None after config
    lm_model_name = llm_resolved if llm_resolved else "deepseek-chat"

    # Here you could add logic to choose between DeepSeekChatModel and DeepSeekReasonerModel
    # based on lm_model_name if needed. For this example, defaulting to DeepSeekChatModel.
    if "reasoner" in lm_model_name.lower():
        lm = DeepSeekReasonerModel(model=lm_model_name, max_tokens=2000, **kwargs)
    else:
        lm = DeepSeekChatModel(model=lm_model_name, max_tokens=2000, **kwargs)


    topic = input('Topic: ')
    file_name = topic.replace(' ', '_')

    mind_map = MindMap(
        retriever=rm,
        gen_concept_lm=lm,
        depth=depth_resolved # Use resolved value
    )

    generator = mind_map.build_map(topic)   
    for layer in generator:
        print(layer)
    mind_map.prepare_table_for_retrieval()

    ogm = OutlineGenerationModule(lm)
    outline = ogm.generate_outline(topic= topic, mindmap = mind_map)

    article_with_outline = Article.from_outline_str(topic=topic, outline_str=outline)
    # Note: threadnum_resolved is available. For ArticleGenerationModule,
    # it was originally hardcoded to 10.
    # Using threadnum_resolved if it's positive, else default to 10.
    max_threads_for_ag = threadnum_resolved if threadnum_resolved > 0 else 10
    ag = ArticleGenerationModule(retriever=rm, article_gen_lm=lm, retrieve_top_k=3, max_thread_num=max_threads_for_ag)
    article = ag.generate_article(topic=topic, mindmap=mind_map, article_with_outline=article_with_outline)
    ap = ArticlePolishingModule(article_gen_lm=lm, article_polish_lm=lm)
    article = ap.polish_article(topic=topic, draft_article=article)

    if not os.path.exists(outputdir_resolved): # Use resolved value
        os.makedirs(outputdir_resolved)
    map_dir = os.path.join(outputdir_resolved, 'map')
    outline_dir = os.path.join(outputdir_resolved, 'outline')
    article_dir = os.path.join(outputdir_resolved, 'article')
    if not os.path.exists(map_dir): os.makedirs(map_dir)
    if not os.path.exists(outline_dir): os.makedirs(outline_dir)
    if not os.path.exists(article_dir): os.makedirs(article_dir)
        
    path = os.path.join(map_dir, file_name)
    with open(path, 'w', encoding='utf-8') as file:
        mind_map.save_map(mind_map.root, path)

    path = os.path.join(outline_dir, file_name)
    with open(path, 'w', encoding='utf-8') as file:
        file.write(outline)

    path = os.path.join(article_dir, file_name)
    with open(path, 'w', encoding='utf-8') as file:
        file.write(article.to_string())

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--cfg', type=str, default='config.yaml',
                        help='Path to the configuration YAML file.')
    parser.add_argument('--framework', type=str, default=None,
                        choices=['dspy', 'langchain'],
                        help='The framework to use ("dspy" or "langchain"). Overrides config.yaml.')
    parser.add_argument('--outputdir', type=str, default='./results',
                        help='Directory to store the outputs.')
    parser.add_argument('--threadnum', type=int, default=3,
                        help='Maximum number of threads to use.')
    parser.add_argument('--retriever', type=str, default=None,
                        help='The search engine API to use for retrieving information (e.g., google).')
    parser.add_argument('--retrievernum', type=int, default=5,
                        help='Number of search results to retrieve.')
    parser.add_argument('--llm', type=str, default=None,
                        help='The language model name (e.g., deepseek-chat, deepseek-reasoner).')
    parser.add_argument('--depth', type=int, default=2,
                        help='The depth of knowledge seeking.')

    parsed_args = parser.parse_args()
    main(parsed_args, parser) # Pass parser object to main