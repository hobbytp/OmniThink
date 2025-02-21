import os
import json
import re
import numpy as np
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor, as_completed
from prometheus_eval import PrometheusEval
from prometheus_eval.prompts import ABSOLUTE_PROMPT, SCORE_RUBRIC_TEMPLATE
from FActScore.factscore.atomic_facts import AtomicFactGenerator, normalize_answer
from prometheus_eval.litellm import LiteLLM
from prometheus_eval.vllm import VLLM

from trim import process_document

class ComprehensiveEvaluator:
    def __init__(self, model_path):
        model = VLLM(model = model_path)
        self.judge = PrometheusEval(model=model, absolute_grade_template=ABSOLUTE_PROMPT)

    def grade(self, rubric_data, topic, response):
        instruction = f"You are an experienced writer, and you need to write an article for the topic: {topic}."
        params = {
            "max_tokens": 2048,
            "repetition_penalty": 1.03,
            "best_of": 1,
            "temperature": 1.0,
            "top_p": 0.9,
        }
        rubric = SCORE_RUBRIC_TEMPLATE.format(**rubric_data)
        feedback, score = self.judge.single_absolute_grade(
            instruction=instruction, response=response, rubric=rubric, params=params
        )
        return feedback, score

    def Coverage(self, topic, response):
        rubric_data = {
        "criteria":"Broad Coverage: Does the article provide an in-depth exploration of the topic and have good coverage?",
        "score1_description":"Severely lacking; offers little to no coverage of the topic’s primary aspects, resulting in a very narrow perspective.",
        "score2_description":"Partial coverage; includes some of the topic’s main aspects but misses others, resulting in an incomplete portrayal.",
        "score3_description":"Acceptable breadth; covers most main aspects, though it may stray into minor unnecessary details or overlook some relevant points.",
        "score4_description":"Good coverage; achieves broad coverage of the topic, hitting on all major points with minimal extraneous information.",
        "score5_description":"Exemplary in breadth; delivers outstanding coverage, thoroughly detailing all crucial aspects of the topic without including irrelevant information."
        }
        return self.grade(rubric_data, topic, response)
    
    def Novelty(self, topic, response):
        rubric_data = {
        "criteria":"Novelty: Does the report cover novel aspects that relate to the user’s initial intent but are not directly derived from it?",
        "score1_description":"Lacks novelty; the report strictly follows the user’s initial intent with no additional insights.",
        "score2_description":"Minimal novelty; includes few new aspects but they are not significantly related to the initial intent",
        "score3_description":"Moderate novelty; introduces some new aspects that are somewhat related to the initial intent.",
        "score4_description":"Good novelty; covers several new aspects that enhance the understanding of the initial intent.",
        "score5_description":"Excellent novelty; introduces numerous new aspects that are highly relevant and significantly enrich the initial intent."
        }
        return self.grade(rubric_data, topic, response)

    def Relevance(self, topic, response):
        rubric_data = {
        "criteria":"Relevance and Focus: How effectively does the report maintain relevance and focus, given the dynamic nature of the discourse?",
        "score1_description":"Very poor focus; discourse diverges significantly from the initial topic and intent with many irrelevant detours.",
        "score2_description":"Poor focus; some relevant information, but many sections diverge from the initial topic.",
        "score3_description":" Moderate focus; mostly stays on topic with occasional digressions that still provide useful information.",
        "score4_description":"Good focus; maintains relevance and focus throughout the discourse with minor divergences that add value.",
        "score5_description":" Excellent focus; consistently relevant and focused discourse, even when exploring divergent but highly pertinent aspects."
        }
        return self.grade(rubric_data, topic, response)
    
    def Depth(self, topic, response):
        rubric_data = {
        "criteria":"Depth of Exploration: How thoroughly does the report explore the initial topic and its related areas, reflecting the dynamic discourse?",
        "score1_description":"Very superficial; provides only a basic overview with significant gaps in exploration.",
        "score2_description":"Superficial; offers some detail but leaves many important aspects unexplored.",
        "score3_description":"Moderate depth; covers key aspects but may lack detailed exploration in some areas.",
        "score4_description":"Good depth; explores most aspects in detail with minor gaps.",
        "score5_description":"Excellent depth; thoroughly explores all relevant aspects with comprehensive detail, reflecting a deep and dynamic discourse."
        }
        return self.grade(rubric_data, topic, response)


    def process_file(self, file_path, grader):
        topic = os.path.basename(file_path).replace('_', ' ')
        try:
            response = process_document(file_path, 2000)

            feedback, score = grader(topic, response)
            return {
                'topic': topic,
                'score': score,
                'feedback': feedback
            }
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return None


    def evaluate_files(self, file_paths, grader, max_workers=1):
        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self.process_file, file_path, grader): file_path for file_path in file_paths}
            for future in as_completed(futures):
                result = future.result()
                if result:
                    print(result)
                    results.append(result)
        return results



    
def main(args):
    model_path = args.modelpath
    stor_path = args.articlepath

    evaluator = ComprehensiveEvaluator(model_path=model_path)

    # Replace with your file paths
    file_paths = []
    for dirs in os.listdir(stor_path):
        topic = dirs.replace('_', ' ')
        txt_path = os.path.join(stor_path, dirs)
        file_paths.append(txt_path)


    Relevance_results = evaluator.evaluate_files(file_paths, evaluator.Relevance)
    avg_Relevance_score = sum(r['score'] for r in Relevance_results) / len(Relevance_results)

    Depth_results = evaluator.evaluate_files(file_paths, evaluator.Depth)
    avg_Depth_score = sum(r['score'] for r in Depth_results) / len(Depth_results)


    Novelty_results = evaluator.evaluate_files(file_paths, evaluator.Novelty)
    avg_Novelty_score = sum(r['score'] for r in Novelty_results) / len(Novelty_results)

    Coverage_results = evaluator.evaluate_files(file_paths, evaluator.Coverage)
    avg_Coverage_score = sum(r['score'] for r in Coverage_results) / len(Coverage_results)

    print("Average Relevance Score:", avg_Relevance_score)
    print("Average Depth Score:", avg_Depth_score)
    print("Average Novelty Score:", avg_Novelty_score)
    print("Average Coverage Score:", avg_Coverage_score)


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--articlepath', type=str, default='../results/article',
                        help='Directory to store the articles.')
    parser.add_argument('--modelpath', type=str, default='./models/prometheus-7b-v2.0',
                        help='Directory to store the model.')
    main(parser.parse_args())