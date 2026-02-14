import numpy as np
import random
import time
from collections import defaultdict
import re
import textwrap
import importlib.util
import os
from dataclasses import dataclass
from typing import Optional, List, Tuple
import sys
from dotenv import load_dotenv

# Load environment variables from .env file if present
load_dotenv()

# Add rl4co-urban to python path if not present
sys.path.append('/root/autodl-tmp/rl4co-urban')

from rl4co.envs.routing.tdtsp.baselines.evaluator import TDTSPEvaluator

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


class CodeSolution:
    """Represents a solution to the optimization problem."""

    def __init__(self, code, prompt_id=None, version=None):
        self.code = code
        self.prompt_id = prompt_id
        self.version = version
        self.score = None
        self.score_detail = None

    def __len__(self):
        return len(self.code)


def softmax(x, temperature=1.0):
    """Compute softmax values for each set of scores in x."""
    e_x = np.exp((x - np.max(x)) / temperature)
    return e_x / e_x.sum()


class Cluster:
    """A cluster of solutions with the same score."""

    def __init__(self, score, solution):
        self.score = score
        self.solutions = [solution]

    def add_solution(self, solution):
        self.solutions.append(solution)

    def sample_solution(self):
        """Sample a solution, preferring shorter ones."""
        if len(self.solutions) == 1:
            return self.solutions[0]

        lengths = [len(s) for s in self.solutions]
        min_len = min(lengths)
        max_len = max(lengths)
        if min_len == max_len:
            return random.choice(self.solutions)

        normalized_lengths = [(l - min_len) / (max_len - min_len + 1e-6) for l in lengths]
        probs = softmax(-np.array(normalized_lengths))
        return self.solutions[np.random.choice(len(self.solutions), p=probs)]


class Island:
    """A subpopulation of solutions."""

    def __init__(self, solution_template, functions_per_prompt=2):
        self.solution_template = solution_template
        self.functions_per_prompt = functions_per_prompt
        self.clusters = {}  # score -> Cluster
        self.best_score = float('-inf')
        self.best_solution = None
        self.next_version = 0

    def register_solution(self, solution, score):
        """Register a solution with its score."""
        solution.score = score

        # Update best solution
        if score > self.best_score:
            self.best_score = score
            self.best_solution = solution

        # Add to appropriate cluster
        if score not in self.clusters:
            self.clusters[score] = Cluster(score, solution)
        else:
            self.clusters[score].add_solution(solution)

    def get_prompt(self):
        """Generate a prompt using the top solutions."""
        if not self.clusters:
            # Return the template if no solutions yet
            return self.solution_template, self.next_version

        # Choose clusters based on score
        scores = list(self.clusters.keys())
        scores.sort(reverse=True)  # Sort in descending order

        # Take the top N scores
        num_solutions = min(self.functions_per_prompt, len(scores))
        chosen_scores = scores[:num_solutions]

        # Sample solutions from each chosen cluster
        chosen_solutions = [self.clusters[score].sample_solution() for score in chosen_scores]

        # Generate prompt with previous solutions
        prompt = self._generate_prompt(chosen_solutions)
        version = self.next_version
        self.next_version += 1
        return prompt, version

    def _generate_prompt(self, solutions):
        """Create a prompt that incorporates the previous solutions."""
        prompt = self.solution_template + "\n\n"
        prompt += ("# Here are some previous solutions for reference. "
                   "The score is normalized where higher is better "
                   "(score > 0 means better than baseline, < 0 means worse):\n\n")

        for i, solution in enumerate(solutions):
            prompt += f"# Solution {i + 1} (score: {solution.score:.4f}):\n"
            prompt += solution.code + "\n\n"
            prompt += "Detailed performance: " + str(solution.score_detail) + "\n\n"

        prompt += ("# Please provide an improved solution that addresses the limitations of previous attempts. "
                   "Analyze the evaluation results and think about how to improve:\n"
                   "1. First outline a brief plan in natural language\n"
                   "2. Then implement this improvement in Python\n"
                   "3. Focus on improving tour quality while respecting time windows\n"
                   "4. Consider time-dependency of travel times\n"
                   "Your solution should be significantly different and better than previous solutions. "
                   "Enclose all your code within: ```python ... ``` and name the main function `def solve(**kwargs)`. "
                   "Do not use -----. Your function has a timeout; aim to return the best results within this limit.")
        return prompt


class FunsearchAgent:
    """Implementation of the Funsearch methodology for TDTSP-TW."""

    def __init__(self, problem_description, num_islands=10, functions_per_prompt=2,
                 reset_period=4 * 60 * 60, timeout=10):
        """Initialize the Funsearch system."""
        self.problem_description = problem_description

        # Add problem description to the template
        full_template = (f"You are an expert in Operation Research and combinatorial optimization. "
                         f"Solve the following problem:\n\n"
                         f"# Problem Description:\n{problem_description}\n\n"
                         f"Ensure your algorithm is as effective as possible. You may use numpy and standard libraries. "
                         f"Enclose all your code within: ```python ... ``` and name the main function `def solve(**kwargs)`. "
                         f"Your function has a {timeout}-second timeout; aim to return the best possible results within this limit.")

        # Initialize islands
        self.num_islands = num_islands
        self.islands = [Island(full_template, functions_per_prompt) for _ in range(num_islands)]

        # Track prompt information
        self.prompts = {}  # prompt_id -> (island_id, version, solution)
        self.next_prompt_id = 0

        # Reset parameters
        self.reset_period = reset_period
        self.last_reset_time = time.time()

        # Best solution tracking
        self.best_scores = [float('-inf')] * num_islands
        self.best_solutions = [None] * num_islands

    def get_prompt(self):
        """Get the next prompt to send to the LLM."""
        # Choose an island randomly
        island_id = random.randint(0, self.num_islands - 1)
        island = self.islands[island_id]

        # Get prompt from the island
        prompt, version = island.get_prompt()

        # Store the prompt details
        prompt_id = self.next_prompt_id
        self.prompts[prompt_id] = (island_id, version, None)
        self.next_prompt_id += 1

        return prompt, prompt_id

    def pull_score(self, prompt_id, score, score_detail, solution_code):
        """Process a score for a generated solution."""
        if prompt_id not in self.prompts:
            raise ValueError(f"Unknown prompt ID: {prompt_id}")

        island_id, version, _ = self.prompts[prompt_id]

        # Create a solution object
        solution = CodeSolution(solution_code, prompt_id, version)
        solution.score = score
        solution.score_detail = score_detail

        # Update the prompts dictionary with the solution
        self.prompts[prompt_id] = (island_id, version, solution)

        # Register the solution with the appropriate island
        self.islands[island_id].register_solution(solution, score)

        # Check if this solution improves the best solution for this island
        if score > self.best_scores[island_id]:
            self.best_scores[island_id] = score
            self.best_solutions[island_id] = solution

        # Check if it's time to reset islands
        current_time = time.time()
        if current_time - self.last_reset_time > self.reset_period:
            self.reset_islands()
            self.last_reset_time = current_time

    def reset_islands(self):
        """Reset the weaker islands to maintain diversity."""
        # Sort islands by their best score
        indices = np.argsort(self.best_scores)

        # Reset the bottom half of islands
        num_to_reset = self.num_islands // 2
        for i in range(num_to_reset):
            island_id = indices[i]

            # Choose a donor island from the top half
            donor_id = indices[-(i % (self.num_islands - num_to_reset) + 1)]

            # Create a new island with the same template
            self.islands[island_id] = Island(self.islands[island_id].solution_template,
                                             self.islands[island_id].functions_per_prompt)

            # Seed it with the best solution from the donor island
            if self.best_solutions[donor_id] is not None:
                donor_solution = self.best_solutions[donor_id]
                self.islands[island_id].register_solution(donor_solution, donor_solution.score)

            # Reset the best score for this island
            self.best_scores[island_id] = self.islands[island_id].best_score
            self.best_solutions[island_id] = self.islands[island_id].best_solution


@dataclass
class ExecutionLog:
    prompt_id: int
    code: str
    score: Optional[float] = None
    feedback: Optional[str] = None
    response: Optional[str] = None


class FunSearch:
    def __init__(self,
                 problem_description,
                 timeout=10,
                 model='gpt-4o',
                 max_iter=64,
                 reasoning_effort='medium',
                 num_islands=10,
                 functions_per_prompt=2,
                 reset_period=2 * 60 * 60
                 ):
        self.problem_description = problem_description
        self.timeout = timeout
        self.model = model
        self.solution = []
        self.max_iter = max_iter
        self.reasoning_effort = reasoning_effort

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

        self.agent = FunsearchAgent(problem_description,
                                    num_islands=num_islands,
                                    functions_per_prompt=functions_per_prompt,
                                    reset_period=reset_period,
                                    timeout=timeout)

    def step(self):
        prompt, prompt_id = self.agent.get_prompt()

        if not self.client:
             print("Warning: OpenAI client not initialized. Set OPENAI_API_KEY.")
             return ""

        try:
            kwargs = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
            }

            max_retries = 5
            base_delay = 2

            for attempt in range(max_retries):
                try:
                    response_obj = self.client.chat.completions.create(**kwargs)

                    if not hasattr(response_obj, 'choices'):
                         print(f"DEBUG: response_obj type: {type(response_obj)}")
                         print(f"DEBUG: response_obj content: {response_obj}")
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

        self.solution.append(ExecutionLog(prompt_id=prompt_id, code=code, response=response_text))
        return code

    def feedback(self, score, feedback):
        if not self.solution:
            return
        self.solution[-1].score = score
        self.solution[-1].feedback = feedback
        if self.solution[-1].code:
            self.agent.pull_score(self.solution[-1].prompt_id, score, feedback, self.solution[-1].code)
        return

    def finalize(self):
        valid_solutions = [s for s in self.solution if s.score is not None]
        if not valid_solutions:
            return None
        previous_best = sorted(valid_solutions, key=lambda x: x.score)[-1]
        return previous_best.code


def solve_funsearch(travel_time_matrix, duration, time_windows, service_times=None, config=None):
    """
    Solve TDTSP-TW using FunSearch.
    This function is called by run_baseline.py or similar.

    Args:
        travel_time_matrix: (N, N, T) numpy array, time-dependent travel times
        duration: float, duration of one time step (in seconds)
        time_windows: (N, 2) numpy array, [earliest, latest] arrival times
        service_times: (N,) numpy array, service time at each node (optional)
        config: dict with 'iterations', 'timeout', etc.

    Returns:
        tour: list of node indices representing the solution
        cost: float, total travel time
    """

    if config is None:
        config = {}

    # Define the problem description
    problem_description = f"""
You are solving the Time-Dependent TSP with Time Windows (TDTSP-TW).
The goal is to find a tour visiting all nodes exactly once, minimizing total travel time while respecting time windows.

You need to implement a function `solve(travel_time_matrix, duration, time_windows, service_times)`:

Args:
    travel_time_matrix: (N, N, T) numpy array
        - travel_time_matrix[i, j, t] is the travel time from node i to j when departing at time step t
        - N is the number of nodes (including depot at index 0)
        - T is the number of time steps
    duration: float
        - Duration of one time step in seconds (e.g., 3600 for 1 hour)
    time_windows: (N, 2) numpy array
        - time_windows[i] = [earliest, latest] for node i (in seconds)
        - Must arrive at node i within [earliest, latest]
        - Depot (node 0) typically has wide time window
    service_times: (N,) numpy array or None
        - Service time at each node in seconds
        - Can be None if no service time

Returns:
    tour: list of integers, sequence of nodes (must be a permutation of 0 to N-1)
          Example: [0, 3, 1, 2, 4, 0] for a 5-node problem
          IMPORTANT: Return a permutation of ALL nodes, not just a path
    cost: float, total travel time for the tour in seconds

Key considerations:
1. Time-dependency: Travel time depends on departure time
   - Use: travel_time = travel_time_matrix[from_node, to_node, time_step]
   - time_step = int(current_time // duration)
2. Time windows: Must arrive within [earliest, latest] for each node
3. Waiting is allowed: If arrive before earliest, wait until earliest
4. Penalties: Large cost if time windows are violated (avoid violations!)
5. Service time: After arriving at a node, must spend service_time before departing

Suggested approaches:
- Nearest neighbor with time window awareness
- Insertion heuristics considering time-dependent costs
- 2-opt or 3-opt local search with time window checking
- Greedy randomized adaptive search
- Consider time-dependent costs when making decisions
"""

    iterations = config.get("iterations", 10)
    timeout = config.get("timeout", 10)
    model = config.get("model", "gpt-4o")

    fs = FunSearch(problem_description, timeout=timeout, max_iter=iterations, model=model)

    best_tour = []
    best_cost = float('inf')

    if not fs.client:
        print("FunSearch skipped: No OpenAI client. Please set OPENAI_API_KEY env var or create openai_key.txt")
        # Return a simple nearest neighbor as fallback
        return simple_nearest_neighbor(travel_time_matrix, duration, time_windows, service_times)

    print(f"Starting FunSearch for TDTSP-TW with {iterations} iterations...")

    for i in range(iterations):
        try:
            code = fs.step()
            print(f"\n{'='*60}")
            print(f"Iteration {i+1}/{iterations}")
            print(f"{'='*60}")

            if not code:
                fs.feedback(float('-inf'), "No code generated")
                continue

            # Print generated code snippet
            code_preview = code[:200] + "..." if len(code) > 200 else code
            print(f"Generated code preview:\n{code_preview}\n")

            local_scope = {'np': np, 'random': random, 'time': time}

            try:
                exec(code, local_scope)
            except Exception as e:
                print(f"❌ Syntax/Import Error: {e}")
                fs.feedback(float('-inf'), f"Syntax/Import Error: {e}")
                continue

            if 'solve' not in local_scope:
                print(f"❌ Function 'solve' not found in generated code")
                fs.feedback(float('-inf'), "Function 'solve' not found in generated code")
                continue

            solve_func = local_scope['solve']

            start_t = time.time()
            try:
                tour, cost = solve_func(travel_time_matrix, duration, time_windows, service_times)
                exec_time = time.time() - start_t
                print(f"✓ Execution completed in {exec_time:.2f}s")
                print(f"  Tour: {tour[:10]}{'...' if len(tour) > 10 else ''}")
                print(f"  Reported cost: {cost:.2f}")
            except Exception as e:
                print(f"❌ Runtime Error: {e}")
                fs.feedback(float('-inf'), f"Runtime Error: {e}")
                continue

            if exec_time > timeout:
                 print(f"❌ Execution timed out ({exec_time:.2f}s > {timeout}s)")
                 fs.feedback(float('-inf'), "Execution timed out")
                 continue

            # Verify result using the standard evaluator
            try:
                evaluator = TDTSPEvaluator(
                    matrix=travel_time_matrix,
                    duration=duration,
                    time_windows=time_windows,
                    service_times=service_times,
                    penalty_value=3.0
                )
                real_cost = evaluator.calculate_cost(tour)

                # Calculate normalized score (higher is better)
                # Use negative cost so higher score = better solution
                score = -real_cost

                feedback_detail = {
                    'cost': real_cost,
                    'tour_length': len(tour),
                    'exec_time': exec_time,
                    'was_late': evaluator.was_late
                }

                fs.feedback(score, feedback_detail)

                print(f"✓ Verified cost: {real_cost:.2f}")
                print(f"  Score: {score:.2f} (higher is better)")
                print(f"  Late violations: {evaluator.was_late}")

                if real_cost < best_cost:
                    best_cost = real_cost
                    best_tour = tour
                    print(f"🎉 NEW BEST SOLUTION! Cost: {best_cost:.2f}")

            except Exception as e:
                print(f"❌ Validation Error: {e}")
                fs.feedback(float('-inf'), f"Validation Error: {e}")

        except Exception as e:
            print(f"❌ System Error: {e}")
            fs.feedback(float('-inf'), f"System Error: {e}")

    print(f"\n{'='*60}")
    print(f"FunSearch finished. Best cost: {best_cost:.2f}")
    print(f"{'='*60}")

    # If no valid solution found, use simple heuristic
    if not best_tour:
        print("No valid solution found, using nearest neighbor fallback")
        return simple_nearest_neighbor(travel_time_matrix, duration, time_windows, service_times)

    return best_tour, best_cost


def simple_nearest_neighbor(travel_time_matrix, duration, time_windows, service_times=None):
    """Simple nearest neighbor heuristic as fallback."""
    N = travel_time_matrix.shape[0]
    T = travel_time_matrix.shape[2]

    tour = [0]
    visited = {0}
    current_node = 0
    current_time = 0.0
    total_cost = 0.0

    # Apply time window at depot
    if time_windows is not None:
        earliest, latest = time_windows[0]
        if current_time < earliest:
            current_time = earliest

    while len(visited) < N:
        best_next = None
        best_cost = float('inf')

        for next_node in range(N):
            if next_node in visited:
                continue

            # Calculate travel time from current node to next_node
            t_idx = min(int(current_time // duration), T - 1)
            travel_time = travel_time_matrix[current_node, next_node, t_idx]
            arrival_time = current_time + travel_time

            # Check time window
            if time_windows is not None:
                earliest, latest = time_windows[next_node]
                if arrival_time > latest + 1000:  # Skip if too late
                    continue

            if travel_time < best_cost:
                best_cost = travel_time
                best_next = next_node

        if best_next is None:
            # No feasible next node, just pick any unvisited
            for next_node in range(N):
                if next_node not in visited:
                    best_next = next_node
                    break

        if best_next is None:
            break

        # Add service time at current node
        if service_times is not None:
            current_time += service_times[current_node]

        # Move to next node
        t_idx = min(int(current_time // duration), T - 1)
        travel_time = travel_time_matrix[current_node, best_next, t_idx]
        current_time += travel_time
        total_cost += travel_time

        # Apply time window waiting
        if time_windows is not None:
            earliest, latest = time_windows[best_next]
            if current_time < earliest:
                current_time = earliest

        tour.append(best_next)
        visited.add(best_next)
        current_node = best_next

    # Evaluate final tour
    evaluator = TDTSPEvaluator(
        matrix=travel_time_matrix,
        duration=duration,
        time_windows=time_windows,
        service_times=service_times,
        penalty_value=3.0
    )
    total_cost = evaluator.calculate_cost(tour)

    return tour, total_cost
