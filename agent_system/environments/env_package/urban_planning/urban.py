import json
import copy

def load_json(file):
    try:
        with open(file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        raise e
    return data

OVERALL_TEMPLATE = load_json("/home/zhangwentao/myspace/verl-agent-urban/agent_system/environments/prompts/prompts/agent_prompt_template.json")["template"]
decision_making_template = load_json("/home/zhangwentao/myspace/verl-agent-urban/agent_system/environments/prompts/prompts/decision_making_template.json")["template"]


def decision_making(sample_info, answer_option_form,task):
    TASK_INFO_PATH = f"/home/zhangwentao/myspace/verl-agent-urban/agent_system/environments/env_package/urban_planning/cfg/task_info.json"
    task_info = load_json(TASK_INFO_PATH)
    overall_template = OVERALL_TEMPLATE.replace("<task_description>", task_info["task_description"])
    overall_template = overall_template.replace("<data_schema>", task_info["data_schema"])
    overall_template = overall_template.replace("<step_instruction>", decision_making_template)
    overall_template = overall_template.replace("<task_target>", task_info["task_target"])
    overall_template = overall_template.replace("<domain_knowledge>", task_info["domain_knowledge"])

    if isinstance(sample_info, list):
        querys = []
        for sample in sample_info:
            query = copy.copy(overall_template)
            data_text = sample
            query = query.replace("<data_analysis>", "N/A")
            query = query.replace("<data_text>", data_text)
            query = query.replace("<answer_option_form>", answer_option_form)

            if "experience" in sample:
                experience = sample["experience"]
            else:
                experience = "N/A"
            query = query.replace("<experience>", experience)

            querys.append(query)
        return querys
    else:
        query = copy.copy(overall_template)        
        data_text = sample_info
        query = query.replace("<data_analysis>", "N/A")
        query = query.replace("<data_text>", data_text)
        answer_option_form = answer_option_form.strip().split("/")
        answer_option = "".join([f"{option}\n" for option in answer_option_form])
        query = query.replace("<answer_option_form>", answer_option)
        query = query.replace("<experience>", "N/A")
        return query