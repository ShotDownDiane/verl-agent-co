import json
import re
import os
import sys
import time
import random
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import asdict

# Add path to import agents
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

try:
    from examples.prompt_agent.vlm_agent import VLMAgent
    from examples.prompt_agent.llm_agent import LLMAgent
except ImportError:
    print("Warning: Agents not found. Using Mock.")
    class VLMAgent:
        def __init__(self, **kwargs): pass
        def generate(self, **kwargs): return "1. Yes\n2. Yes\n<Observation> The nodes are clustered. </Observation>"
    class LLMAgent:
        def __init__(self, **kwargs): pass
        def generate(self, **kwargs): return "<Thought> Option A is best. </Thought>"

from .geometry_engine import GeometryEngine, VerificationQuestion
from .prompts import PromptManager

class CoTPipeline:
    def __init__(self, vlm_agent, llm_agent):
        self.vlm = vlm_agent
        self.llm = llm_agent
        self.prompt_mgr = PromptManager()

    def _parse_verification_answers(self, response: str, num_questions: int) -> List[str]:
        """Parses '1. Yes', '2. No' from response."""
        answers = []
        lines = response.split('\n')
        for i in range(num_questions):
            # Look for "i+1. Yes/No"
            pattern = re.compile(rf"{i+1}\.\s*(Yes|No)", re.IGNORECASE)
            match = None
            for line in lines:
                m = pattern.search(line)
                if m:
                    match = m
                    break
            
            if match:
                answers.append(match.group(1).title()) # Normalize to Yes/No
            else:
                answers.append("Unknown")
        return answers

    def _extract_tag_content(self, text: str, tag: str) -> str:
        pattern = re.compile(rf"<{tag}>(.*?)</{tag}>", re.DOTALL)
        match = pattern.search(text)
        return match.group(1).strip() if match else ""

    def process_step(self, 
                     step_data: Dict[str, Any], 
                     geo_engine: GeometryEngine, 
                     vehicle_capacity: float) -> Optional[str]:
        """
        Runs the 3-stage distillation for a single step.
        Returns the final CoT string if successful, else None.
        """
        # Unpack data
        obs_text = step_data.get('obs', '')
        image_b64 = step_data.get('image', None)
        current_idx = step_data.get('current_node_idx', 0) # Need to ensure this exists in data
        candidates = step_data.get('candidates', [])
        current_load = step_data.get('current_load', 0)
        
        # Expert Action Parsing (e.g. "\\boxed{A}" -> Node ID)
        action_raw = str(step_data['trajectory'])
        option_map = {chr(65+i): cid for i, cid in enumerate(candidates)}
        # Inverse map
        # Assume action_raw is like "\boxed{A}" or just "A"
        cleaned_action = action_raw.replace("\\boxed{", "").replace("}", "").strip()
        expert_action_idx = option_map.get(cleaned_action, candidates[0] if candidates else 0)
        
        expert_action_desc = f"Option {cleaned_action} [Node {expert_action_idx}]"

        # --- Stage 1: Obs Generation (Program-Verified) ---
        questions = geo_engine.generate_verification_questions(
            current_idx, expert_action_idx, candidates, current_load, vehicle_capacity
        )
        
        # Generate multiple samples (N=3)
        verified_obs = None
        for _ in range(3):
            prompt = self.prompt_mgr.get_obs_prompt([asdict(q) for q in questions])
            response = self.vlm.generate(
                system_prompt="You are a geometric reasoning engine.",
                text=prompt,
                image=image_b64,
                temperature=0.7 # High temp for diversity
            )
            
            # Verify Answers
            model_answers = self._parse_verification_answers(response, len(questions))
            is_correct = True
            for ans, gt_q in zip(model_answers, questions):
                if ans != gt_q.expected_answer:
                    is_correct = False
                    break
            
            if is_correct:
                obs_content = self._extract_tag_content(response, "Observation")
                if obs_content:
                    verified_obs = obs_content
                    break
        
        if not verified_obs:
            print(f"Stage 1 Failed: Could not generate verified observation for Step {step_data['step_idx']}")
            return None

        # --- Stage 2: Thk Generation (Action-Consistent) ---
        candidates_desc = "\n".join([f"Option {chr(65+i)}: Node {c}" for i, c in enumerate(candidates)])
        thk_prompt = self.prompt_mgr.get_thk_prompt(verified_obs, obs_text, candidates_desc, expert_action_desc)
        
        final_thk = None
        for _ in range(3):
            response = self.llm.generate(
                system_prompt="You are a logistics strategist.",
                text=thk_prompt,
                temperature=0.7
            )
            
            # Verify if it chose the expert action
            # Simple check: does it mention the expert option as the choice?
            # Ideally, we'd parse a \boxed{} from it, but Thk usually ends with selection.
            if cleaned_action in response or str(expert_action_idx) in response:
                final_thk = self._extract_tag_content(response, "Thought") or response
                break
        
        if not final_thk:
            print(f"Stage 2 Failed: Logic did not align with expert action.")
            return None

        # --- Stage 3: Refinement ---
        refine_prompt = self.prompt_mgr.get_refinement_prompt(verified_obs, final_thk, expert_action_desc)
        final_cot_response = self.llm.generate(
            system_prompt="You are a data refiner.",
            text=refine_prompt,
            temperature=0.2 # Low temp for deterministic format
        )
        
        return final_cot_response

