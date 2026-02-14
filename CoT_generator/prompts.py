from typing import List, Dict

class PromptManager:
    @staticmethod
    def get_obs_prompt(questions: List[Dict[str, str]]) -> str:
        q_text = "\n".join([f"{i+1}. {q['question_text']}" for i, q in enumerate(questions)])
        return (
            f"Please analyze the current routing situation.\n\n"
            f"Part 1: Answer the following Verification Questions with 'Yes' or 'No'.\n"
            f"{q_text}\n\n"
            f"Part 2: Write a detailed <Observation> describing the geometric relations, "
            f"spatial distribution, and key constraints visible in the map and data. "
            f"Focus on the expert action's relation to the depot and other candidates.\n"
            f"Format:\n"
            f"1. Yes/No\n"
            f"2. Yes/No\n"
            f"...\n"
            f"<Observation> ... </Observation>"
        )

    @staticmethod
    def get_thk_prompt(verified_obs: str, state_summary: str, candidates_desc: str, expert_action_desc: str) -> str:
        return (
            f"You are a logistics expert.\n\n"
            f"**Verified Observation**:\n{verified_obs}\n\n"
            f"**State Summary**:\n{state_summary}\n\n"
            f"**Candidates**:\n{candidates_desc}\n\n"
            f"**Task**: Explain why the expert action **{expert_action_desc}** is the optimal choice compared to other candidates. "
            f"Use the geometric facts from the observation and the constraints from the state summary.\n"
            f"Write your reasoning as a <Thought> trace."
        )

    @staticmethod
    def get_refinement_prompt(verified_obs: str, generated_thk: str, expert_action_desc: str) -> str:
        return (
            f"Refine the following reasoning trace into a standardized OTD (Observation-Thought-Decision) format.\n\n"
            f"Input Observation: {verified_obs}\n"
            f"Input Thought: {generated_thk}\n"
            f"Expert Action: {expert_action_desc}\n\n"
            f"Requirements:\n"
            f"1. <Observation>: Summarize key geometric facts (Cluster, Direction, Capacity status). Concise.\n"
            f"2. <Thought>: Explain the strategy (e.g., Angular Sweep, Depot Return). Link facts to the decision.\n"
            f"3. <Decision>: Output the action in \\boxed{{}} format.\n\n"
            f"Output Format:\n"
            f"<Observation> ... </Observation>\n"
            f"<Thought> ... </Thought>\n"
            f"<Decision> \\boxed{{...}} </Decision>"
        )
