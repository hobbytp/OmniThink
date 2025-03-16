import os
import sys
import yaml
import re
from argparse import ArgumentParser
from src.actions.article_generation import ArticleGenerationModule
from src.actions.article_polish import ArticlePolishingModule
from src.actions.outline_generation import OutlineGenerationModule
from src.dataclass.Article import Article
from src.tools.lm import OpenAIModel_dashscope, ZhiPuModel
from src.tools.mindmap import MindMap
from src.tools.rm import GoogleSearchAli, BingSearchAli_new


def load_config(config_path: str) -> dict:
    with open(config_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    return config


def extract_agent_arguments(text: str, pattern: str = r'[A-Za-z]+'):
    """
    Extract words that conform to the regular expression pattern from the input text.
    """
    matches = re.findall(pattern, text)
    function_name = ''.join(matches)
    language_style = {"language": matches[-1], "style": matches[-2]} if len(matches) > 2 else None
    return function_name, language_style


def main(args):
    kwargs = {
        'temperature': 1.0,
        'top_p': 0.9,
        # 'api_key': os.getenv("OPENAI_API_KEY"),
        # 'api_key': 'sk-LdjOX2HyTjqqr6qtr7el3VyVSWGYdciQTgvEX8J2e8zbIxco',
        # 'api_key': 'sk-EGH4X4I171QTGki7pRESrYP9uUGyvWmqoFSAXBCDh4Fxj2DE',
        'api_key': '1524ca4f1d944a3ab8940933c16d7597.D093usP88Rj2JGxj'
    }

    if args.retriever == 'google':
        rm = GoogleSearchAli(k=args.retrievernum)

    try:
        config = load_config(args.cfg)
    except FileNotFoundError as e:
        print(e)

    agent = config['agent']
    class_name, language_style = extract_agent_arguments(agent.get('name'))

    lm = ZhiPuModel(model=args.llm, max_tokens=1000, **kwargs)

    # topic = input('Topic: ')
    topic = 'Write an introduction to the job of a translator'
    topic = 'Taylor Swift EN'
    file_name = topic.replace(' ', '_')

    mind_map = MindMap(
        retriever=rm,
        gen_concept_lm=lm,
        depth=args.depth
    )

    # generator = mind_map.build_map(topic)
    # for layer in generator:
    #     print(layer)

    mindmap = mind_map.load_map(f'{args.outputdir}/map/Taylor_Swift.json')
    print(mindmap)

    ogm = OutlineGenerationModule(lm)
    outline = ogm.generate_outline(topic=topic, mindmap=mind_map)

    article_with_outline = Article.from_outline_str(topic=topic, outline_str=outline)
    ag = ArticleGenerationModule(retriever=rm, article_gen_lm=lm, retrieve_top_k=3, max_thread_num=10,
                                 agent_name=class_name)
    article = ag.generate_article(topic=topic, mindmap=mind_map, article_with_outline=article_with_outline,
                                  language_style=language_style)
    ap = ArticlePolishingModule(article_gen_lm=lm, article_polish_lm=lm)
    article = ap.polish_article(topic=topic, draft_article=article)

    if not os.path.exists(args.outputdir):
        os.makedirs(args.outputdir)
    if not os.path.exists(f'{args.outputdir}/map'):
        os.makedirs(f'{args.outputdir}/map')
    if not os.path.exists(f'{args.outputdir}/outline'):
        os.makedirs(f'{args.outputdir}/outline')
    if not os.path.exists(f'{args.outputdir}/article'):
        os.makedirs(f'{args.outputdir}/article')

    path = f'{args.outputdir}/map/{file_name}'
    with open(path, 'w', encoding='utf-8') as file:
        mind_map.save_map(mind_map.root, path)

    path = f'{args.outputdir}/outline/{file_name}'
    with open(path, 'w', encoding='utf-8') as file:
        file.write(outline)

    path = f'{args.outputdir}/article/{file_name}'
    with open(path, 'w', encoding='utf-8') as file:
        file.write(article.to_string())


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--outputdir', type=str, default='./results',
                        help='Directory to store the outputs.')
    parser.add_argument('--threadnum', type=int, default=3,
                        help='Maximum number of threads to use. The information seeking part and the article generation'
                             'part can speed up by using multiple threads. Consider reducing it if keep getting '
                             '"Exceed rate limit" error when calling LM API.')
    parser.add_argument('--retriever', type=str,
                        help='The search engine API to use for retrieving information.')
    parser.add_argument('--retrievernum', type=int, default=5,
                        help='The search engine API to use for retrieving information.')

    parser.add_argument('--llm', type=str,
                        help='The language model API to use for generating content.')
    parser.add_argument('--depth', type=int, default=2,
                        help='The depth of knowledge seeking.')
    parser.add_argument('--cfg', type=str, default='config.yaml',
                        help='the config of knowledge seeking and ')

    main(parser.parse_args())
