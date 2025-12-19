import json
import copy
from .. import utils

def load_json(file):
    try:
        with open(file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        raise e
    return data


root_dir = utils.get_root_dir()

OVERALL_TEMPLATE = load_json(f"{root_dir}/agent_system/environments/prompts/prompts/agent_prompt_template.json")["template"]
decision_making_template = load_json(f"{root_dir}/agent_system/environments/prompts/prompts/decision_making_template.json")["template"]


def decision_making(sample_info, answer_option_form,task):
    TASK_INFO_PATH = f"{root_dir}/agent_system/environments/env_package/road_planning/Data/task_info.json"
    task_info = load_json(TASK_INFO_PATH)
    overall_template = OVERALL_TEMPLATE.replace("<task_description>", task_info["task_description"])
    overall_template = overall_template.replace("<data_schema>", task_info["data_schema"])
    overall_template = overall_template.replace("<domain_knowledge>", task_info["domain_knowledge"])
    query = copy.copy(overall_template)
    
    data_text = sample_info
    query = query.replace("<data_analysis>", "N/A")

    query = query.replace("<data_text>", data_text)
    query = query.replace("<step_instruction>", decision_making_template)
    query = query.replace("<answer_option_form>", answer_option_form[0])

    query = query.replace("<experience>", "N/A")

    return query