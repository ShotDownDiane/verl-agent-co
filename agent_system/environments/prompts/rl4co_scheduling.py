RL4CO_JSSP_TEMPLATE_NO_HIS = """
Solve the Job Shop Scheduling Problem (JSSP) with {num_jobs} jobs and {num_machines} machines. Each job consists of {num_ops} operations which need to be sequentially processed on specific machines. Each machine can process only one job at a time and each job can be processed by only one machine at a time. Identify the schedule that minimizes the maximum completion time (makespan). The input includes the information of operations for each job, including their specific machine and processing time, as well as the operators with lowest processing time and their respective machines and processing times. Provide the solution in the following format:

1. Schedule: List the order that jobs are processed on each machine.
2. Makespan: The makespan of the schedule.

Input:
{job_descriptions}

Reply with a JSON object:
{{
  "schedule": [[<machine 0 job order>], [<machine 1 job order>], ...],
  "makespan": <makespan value>
}}
"""


RL4CO_JSSP_TEMPLATE = """
Solve the Job Shop Scheduling Problem (JSSP) with {num_jobs} jobs and {num_machines} machines. Each job consists of {num_ops} operations which need to be sequentially processed on specific machines. Each machine can process only one job at a time and each job can be processed by only one machine at a time. Identify the schedule that minimizes the maximum completion time (makespan). The input includes the information of operations for each job, including their specific machine and processing time, as well as the operators with lowest processing time and their respective machines and processing times. Provide the solution in the following format:

1. Schedule: List the order that jobs are processed on each machine.
2. Makespan: The makespan of the schedule.

Input:
{job_descriptions}

Reply with a JSON object:
{{
  "schedule": [[<machine 0 job order>], [<machine 1 job order>], ...],
  "makespan": <makespan value>
}}
"""


RL4CO_FFSP_TEMPLATE_NO_HIS = """
Solve a Flexible Flow Shop Problem (FFSP).

There are multiple stages; each stage can have one or more parallel machines.
Each job must pass through all stages in order. At each stage, you must assign the job to one of the machines in that stage.
Objective:
- Minimize the makespan (time when the last job completes all stages).

Instance summary:
- Number of stages: {num_stages}
- Machines per stage: {machines_per_stage}
- Jobs with per-stage processing times: {job_stage_times}

You will build the schedule stage by stage by assigning the next operation (job, stage) to a machine.
Final solution conceptually should include:
1) Schedule: for each stage/machine, sequence of assigned operations with start/end times.
2) Objective: the makespan value.

At this step, choose the NEXT admissible operation (job at a certain stage) to assign.
Reply with a JSON object:
{{
  "thoughts": "consider machine availability and downstream stages; explain why this assignment is good",
  "action": "<integer operation index>"
}}
"""


RL4CO_FFSP_TEMPLATE = """
Solve a Flexible Flow Shop Problem (FFSP).

There are multiple stages; each stage can have one or more parallel machines.
Each job must pass through all stages in order. At each stage, you must assign the job to one of the machines in that stage.
Objective:
- Minimize the makespan (time when the last job completes all stages).

Instance summary:
- Number of stages: {num_stages}
- Machines per stage: {machines_per_stage}
- Jobs with per-stage processing times: {job_stage_times}

Current scheduling context:
- Current time (approx / aggregated): {current_time}
- Admissible operations to assign next:
{admissible_actions}

History (last {history_length} decisions):
{action_history}

Ultimately, the solution should include:
1) Schedule: for each stage/machine, sequence of assigned operations with start/end times.
2) Objective: the makespan value.

At this step, choose the NEXT admissible operation (job at a certain stage) to assign.
Reply with a JSON object:
{{
  "thoughts": "consider machine availability, stage queues, and downstream impact; justify this assignment",
  "action": "<integer operation index>"
}}
"""



