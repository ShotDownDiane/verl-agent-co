import numpy as np
import random
import time
import re
import textwrap
import os
from dataclasses import dataclass
from typing import Optional
import sys
from dotenv import load_dotenv

# Load environment variables from .env file if present
load_dotenv()

# Add rl4co-urban to python path if not present
sys.path.append('/root/autodl-tmp/rl4co-urban')

from rl4co.envs.routing.tdvrp.baselines.evaluator import TDVRPEvaluator

try:
    from openai import OpenAI, RateLimitError, APIError
except ImportError:
    OpenAI = None
    RateLimitError = None
    APIError = None

def extract_code_blocks(response):
    pattern_backticks = r"```python\s*(.*?)\s*```"
    pattern_dashes = r"^-{3,}\s*\n(.*?)\n-{3,}"
    blocks = re.findall(pattern_backticks, response, re.DOTALL)
    blocks.extend(re.findall(pattern_dashes, response, re.DOTALL | re.MULTILINE))
    return blocks

@dataclass
class Solution:
    code: str
    score: Optional[float] = None
    feedback: Optional[str] = None
    response: Optional[str] = None

class GreedyRefine:
    def __str__(self):
        return f"Greedy Refinement"

    def __init__(self, problem_description, timeout=10, model='gpt-4o-mini', max_iter=64,
                 reasoning_effort='medium', additional_prompt=''):
        self.problem_description = problem_description
        self.timeout = timeout
        self.model = model
        self.solution = []
        self.max_iter = max_iter
        self.reasoning_effort = reasoning_effort
        self.additional_prompt = additional_prompt
        
        self.client = None
        if OpenAI:
            api_key = os.environ.get("OPENAI_API_KEY")
            base_url = os.environ.get("OPENAI_BASE_URL")
            
            # Try loading from openai_key.txt if not in env
            if not api_key:
                try:
                    if os.path.exists("openai_key.txt"):
                        with open("openai_key.txt", "r") as f:
                            api_key = f.read().strip()
                except Exception:
                    pass

            if api_key:
                self.client = OpenAI(api_key=api_key, base_url=base_url)

    def step(self):
        if not self.client:
             print("Warning: OpenAI client not initialized. Set OPENAI_API_KEY.")
             return ""

        if len(self.solution) == 0:
            prompt = (
                f"You are an expert in Operation Research problem. "
                f"Solve the following problem:\n\n{self.problem_description}\n\n"
                f"Ensure your algorithm is as effective as possible. You may use any Python package. "
                f"Enclose all your code within a code block: ```python ... ``` and name the main function `def solve(**kwargs)`. "
                f"Your function has a {self.timeout}-second timeout; aim to return the best possible results within this limit."
            )
        else:
            previous_best = sorted(self.solution, key=lambda x: x.score)[-1]
            prompt = (
                f"You are an expert in Operations Research."
                f" You are tasked with solving the following problem:\n\n{self.problem_description}\n\n"
                f"Below is a previously developed solution. Your goal is to enhance this solution to further improve its test-time performance:\n\n"
                f"{previous_best.code}\n\n"
                f"Here are the evaluation scores of the existing solution for each test case and example:\n\n"
                f"{previous_best.feedback}\n\n"
                f"These scores are normalized relative to a reference solution, with higher values indicating better performance. "
                f"Analyze these evaluation results carefully to identify areas for improvement.\n\n"
                f"First, outline a concise, clear plan in natural language describing how you intend to improve the solution."
                f" Then, implement your proposed improvements in Python based on the previous solution provided. "
                f"You are encouraged to propose significant, innovative improvements—your solution should be distinctly different and clearly superior. "
                f"If you have a completely new and more effective approach, feel free to abandon the previous method and adopt your new approach. "
                f"Enclose all your code within a Python code block using: ```python ... ``` and ensure the main function is named `def solve(**kwargs)`. "
                f"Do not use separator lines (e.g., '-----'). "
                f"Ensure your code is as effective as possible. You may use any Python package. "
                f"Your function has a {self.timeout}-second timeout; aim to return the best possible results within this limit."
            )
        prompt = prompt + '\n' + self.additional_prompt
        
        try:
            kwargs = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
            }
            # if "o3-mini" in self.model:
            #     kwargs["reasoning_effort"] = self.reasoning_effort
            
            max_retries = 5
            base_delay = 2
            
            response_text = ""
            for attempt in range(max_retries):
                try:
                    response_obj = self.client.chat.completions.create(**kwargs)
                    
                    if not hasattr(response_obj, 'choices'):
                         raise ValueError("Response is not an object with choices")

                    response_text = response_obj.choices[0].message.content
                    break
                except (RateLimitError, APIError) as e:
                    if attempt == max_retries - 1:
                        raise e
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                    print(f"LLM API Error: {e}. Retrying in {delay:.2f}s...")
                    time.sleep(delay)
            
            code_blocks = extract_code_blocks(response_text)
            if code_blocks:
                code = textwrap.dedent(code_blocks[0])
            else:
                code = ""
        except Exception as e:
            print(f"Error calling LLM: {e}")
            response_text = str(e)
            code = ""
            
        self.solution.append(Solution(code=code, response=response_text))
        return code

    def feedback(self, score, feedback):
        if not self.solution:
            return
        self.solution[-1].score = score
        self.solution[-1].feedback = feedback
        return

    def finalize(self):
        valid_solutions = [s for s in self.solution if s.score is not None]
        if not valid_solutions:
            return None
        previous_best = sorted(valid_solutions, key=lambda x: x.score)[-1]
        return previous_best.code


def solve_greedy_refine(matrix, duration, time_windows, service_times, config):
    """
    Solve TDVRP using Greedy Refine.
    This function is called by run_baseline.py.
    """
    
    # Define the problem description
    problem_description = """
    You are solving the Time-Dependent VRP (TDVRP).
    The goal is to minimize the total travel duration.
    
    You need to implement a function `solve(matrix, duration, time_windows, service_times)`:
    
    Args:
        matrix: (N, N, T) numpy array, travel times between nodes at different time steps.
        duration: float, duration of one time step.
        time_windows: (N, 2) numpy array, [start, end] for each node.
        service_times: (N) numpy array, service duration for each node.
        
    Returns:
        routes: list of lists, where each list is a route (sequence of node indices, starting and ending at 0).
        cost: float, total travel duration.
    """
    
    iterations = config.get("iterations", 5)
    timeout = config.get("timeout", 10)
    
    gr = GreedyRefine(problem_description, timeout=timeout, max_iter=iterations)
    
    best_routes = []
    best_cost = float('inf')
    
    if not gr.client:
        print("GreedyRefine skipped: No OpenAI client. Please set OPENAI_API_KEY env var or create openai_key.txt")
        return [], float('inf')

    print(f"Starting GreedyRefine for {iterations} iterations...")

    for i in range(iterations):
        try:
            code = gr.step()
            if not code:
                gr.feedback(float('-inf'), "No code generated")
                continue
                
            local_scope = {'np': np, 'random': random}
            
            try:
                exec(code, local_scope)
            except Exception as e:
                gr.feedback(float('-inf'), f"Syntax/Import Error: {e}")
                continue

            if 'solve' not in local_scope:
                gr.feedback(float('-inf'), "Function 'solve' not found in generated code")
                continue
                
            solve_func = local_scope['solve']
            
            start_t = time.time()
            try:
                routes, cost = solve_func(matrix, duration, time_windows, service_times)
            except Exception as e:
                print(f"DEBUG: Runtime Error in generated code: {e}")
                gr.feedback(float('-inf'), f"Runtime Error: {e}")
                continue
                
            end_t = time.time()
            exec_time = end_t - start_t
            
            if exec_time > timeout:
                 gr.feedback(float('-inf'), "Execution timed out")
                 continue
                 
            # Verify result using the standard evaluator
            try:
                evaluator = TDVRPEvaluator(matrix, duration, time_windows, service_times)
                real_cost, details = evaluator.calculate_cost(routes, return_details=True)
                
                # Check for validity
                if details['total_violation'] > 0:
                    # print(f"DEBUG: Constraint violation in generated solution: {details['total_violation']}")
                    gr.feedback(float('-inf'), f"Constraint violations: {details}")
                    continue

                # Check for coverage
                visited = set()
                for r in routes:
                    visited.update(r)
                
                num_nodes = matrix.shape[0]
                # Assume 0 is depot, customers are 1 to N-1
                expected = set(range(1, num_nodes)) 
                missing = expected - visited
                
                if missing:
                    # print(f"DEBUG: Missing {len(missing)} nodes")
                    gr.feedback(float('-inf'), f"Invalid solution: Missing {len(missing)} nodes. You must visit all customers (indices 1 to {num_nodes-1}).")
                    continue
                    
                # We want to maximize the negative cost (since we want to minimize cost)
                score = -real_cost
                feedback_str = f"Valid solution found. Cost: {real_cost:.2f}"
                gr.feedback(score, feedback_str)
                
                if real_cost < best_cost:
                    best_cost = real_cost
                    best_routes = routes
                    print(f"New best solution found! Cost: {best_cost:.2f}")
                    
            except Exception as e:
                 gr.feedback(float('-inf'), f"Evaluation Error: {e}")
                 continue
                 
        except Exception as e:
            print(f"Unexpected error in iteration {i}: {e}")
            
    return best_routes, best_cost
