import argparse
import json
import os
import re
import unittest
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

# Try importing vllm, but allow running without it for testing/parsing only
try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False

@dataclass
class SFTEntry:
    prompt: str
    completion: str
    metadata: Dict[str, Any]

class CoTGenerator:
    """
    A class to generate Chain-of-Thought (CoT) datasets for Supervised Fine-Tuning (SFT)
    using existing agent trajectories and a powerful teacher model.
    """
    
    def __init__(self, model_name: str, template_type: str = "flp", 
                 tensor_parallel_size: int = 1, gpu_memory_utilization: float = 0.9,
                 max_model_len: int = 4096):
        self.model_name = model_name
        self.template_type = template_type
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.llm = None
        
    def initialize_model(self):
        """Initialize the vLLM model if available."""
        if not VLLM_AVAILABLE:
            raise ImportError("vLLM is not installed. Please install it to run inference.")
            
        print(f"Initializing vLLM with model: {self.model_name}")
        self.llm = LLM(
            model=self.model_name,
            tensor_parallel_size=self.tensor_parallel_size,
            gpu_memory_utilization=self.gpu_memory_utilization,
            max_model_len=self.max_model_len,
            trust_remote_code=True
        )

    def parse_action(self, action_str: str) -> str:
        """Extract the action label from the boxed string (e.g., '\\boxed{A}' -> 'A')."""
        match = re.search(r"\\boxed\{(.*?)\}", action_str)
        if match:
            return match.group(1)
        return action_str.strip()

    def construct_teacher_prompt(self, obs: str, action: str) -> str:
        """
        Construct a prompt for the teacher model to explain the action.
        """
        clean_action = self.parse_action(action)
        
        system_prompt = (
            "You are an expert in Combinatorial Optimization and reasoning. "
            "Your task is to analyze the provided Observation and the Chosen Action, "
            "and generate a step-by-step Chain-of-Thought (CoT) reasoning that justifies why this action is the optimal choice.\n\n"
            "Guidelines:\n"
            "1. Analyze the status and the list of candidates provided in the observation.\n"
            "2. Compare the Chosen Action's metrics (e.g., distance reduction, cost) against other options.\n"
            "3. Explicitly state why the Chosen Action is superior or valid.\n"
            "4. Conclude with the action selection.\n"
            "5. The output should be the 'Thought' process followed by the 'Action'."
        )
        
        user_prompt = (
            f"### Observation:\n{obs}\n\n"
            f"### Chosen Action:\n{action}\n\n"
            f"### Instruction:\n"
            f"Provide a detailed reasoning (CoT) justifying the choice of Option {clean_action}. "
            f"Explain what this option represents (e.g., which node, what gain/cost) and why it stands out."
        )
        
        # Format for chat models (simplified)
        return f"{system_prompt}\n\n{user_prompt}"

    def process_file(self, input_file: str) -> List[Dict[str, Any]]:
        """Parse the input JSON file containing agent trajectories."""
        with open(input_file, 'r') as f:
            data = json.load(f)
            
        processing_tasks = []
        
        for episode_idx, episode in enumerate(data):
            obs_list = episode.get('obs_list', [])
            trajectory = episode.get('trajectory', [])
            
            # Ensure alignment
            min_len = min(len(obs_list), len(trajectory))
            
            for step_idx in range(min_len):
                obs = obs_list[step_idx]
                action = trajectory[step_idx]
                
                # Filter out steps where observation might be empty or action invalid
                if not obs or not action:
                    continue
                    
                prompt = self.construct_teacher_prompt(obs, action)
                
                processing_tasks.append({
                    "episode_idx": episode_idx,
                    "step_idx": step_idx,
                    "obs": obs,
                    "target_action": action,
                    "teacher_prompt": prompt
                })
                
        return processing_tasks

    def generate_cot(self, tasks: List[Dict[str, Any]], temperature: float = 0.7, top_p: float = 0.95) -> List[SFTEntry]:
        """Generate CoT for all tasks using vLLM."""
        if self.llm is None:
            self.initialize_model()
            
        prompts = [task['teacher_prompt'] for task in tasks]
        sampling_params = SamplingParams(temperature=temperature, top_p=top_p, max_tokens=1024)
        
        print(f"Generating CoT for {len(prompts)} samples...")
        outputs = self.llm.generate(prompts, sampling_params)
        
        results = []
        for i, output in enumerate(outputs):
            generated_text = output.outputs[0].text
            
            # Construct the final SFT entry
            # The SFT model should take 'obs' as input and output 'reasoning + action'
            # Here we assume the generated text is the reasoning
            
            task = tasks[i]
            
            # Format: <Thought> ... </Thought> Action: \boxed{X}
            # Or just raw text depending on preference. Let's use a structured format.
            final_completion = f"{generated_text.strip()}\n\nAction: {task['target_action']}"
            
            results.append(SFTEntry(
                prompt=task['obs'],
                completion=final_completion,
                metadata={
                    "episode_idx": task['episode_idx'],
                    "step_idx": task['step_idx']
                }
            ))
            
        return results

    def save_results(self, results: List[SFTEntry], output_file: str):
        """Save the generated dataset in JSONL format."""
        with open(output_file, 'w') as f:
            for entry in results:
                json_obj = {
                    "messages": [
                        {"role": "user", "content": entry.prompt},
                        {"role": "assistant", "content": entry.completion}
                    ],
                    "metadata": entry.metadata
                }
                f.write(json.dumps(json_obj) + "\n")
        print(f"Saved {len(results)} entries to {output_file}")


class TestCoTGenerator(unittest.TestCase):
    """Unit tests for the CoTGenerator."""
    
    def setUp(self):
        self.generator = CoTGenerator(model_name="test-model")
        
    def test_parse_action(self):
        self.assertEqual(self.generator.parse_action("\\boxed{A}"), "A")
        self.assertEqual(self.generator.parse_action("  \\boxed{B} "), "B")
        self.assertEqual(self.generator.parse_action("C"), "C")
        
    def test_construct_prompt(self):
        obs = "Option A: dist 10\nOption B: dist 20"
        action = "\\boxed{A}"
        prompt = self.generator.construct_teacher_prompt(obs, action)
        self.assertIn("Option A", prompt)
        self.assertIn("expert", prompt)
        self.assertIn("justifies why this action", prompt)

def main():
    parser = argparse.ArgumentParser(description="Generate CoT SFT Dataset using vLLM")
    parser.add_argument("--input_file", type=str, required=True, help="Path to input agent output JSON")
    parser.add_argument("--output_file", type=str, required=True, help="Path to output SFT JSONL")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-72B-Instruct", help="Teacher model name/path")
    parser.add_argument("--tp_size", type=int, default=1, help="Tensor Parallel size")
    parser.add_argument("--test_mode", action="store_true", help="Run unit tests only")
    parser.add_argument("--dry_run", action="store_true", help="Process files but do not run inference (print prompts)")
    
    args = parser.parse_args()
    
    if args.test_mode:
        sys.argv = [sys.argv[0]]
        unittest.main()
        return

    generator = CoTGenerator(
        model_name=args.model,
        tensor_parallel_size=args.tp_size
    )
    
    print(f"Processing input file: {args.input_file}")
    tasks = generator.process_file(args.input_file)
    print(f"Found {len(tasks)} samples.")
    
    if args.dry_run:
        print("Dry run enabled. First 3 prompts:")
        for i in range(min(3, len(tasks))):
            print(f"\n--- Prompt {i+1} ---\n{tasks[i]['teacher_prompt']}\n------------------")
        return

    sft_entries = generator.generate_cot(tasks)
    generator.save_results(sft_entries, args.output_file)

if __name__ == "__main__":
    import sys
    main()
