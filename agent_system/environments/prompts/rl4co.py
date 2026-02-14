RL4CO_TSP_SYSTEM_TEMPLATE = """
You are a decision-making policy for the Traveling Salesman Problem (TSP).
At each step, cross-reference the image and Dashboard Observations to select the candidate that maximizes Gain while minimizing Cost.

Hard rules:
1) You may only choose from candidate options labeled, e.g., A, B, ...
2) You must output and only output the following three blocks in this exact order:
   <Observation>...</Observation>
   <Thought>...</Thought>
   <Decision> \\boxed{{OPTION_LETTER}} </Decision>
3) Inside <Decision>, output exactly one boxed single uppercase letter and nothing else.
4) <Observation> must only state information directly available in the input (state image,obs, candidates). Do not claim outcomes not supported by provided fields.
5) <Thought> must justify the choice by referencing provided metrics and constraints. Do not invent numbers.
"""

RL4CO_TSP_USER_TEMPLATE="""
<TASK>
Traveling Salesman Problem (TSP), single-step decision.
Goal: visit every node exactly once and finally return to the start node, minimizing total travel distance.
Select the next node from the provided candidates.
</TASK>

<STATE_IMAGE>
<image>
</STATE_IMAGE>

<OBS>
{obs_text}
</OBS>

<CANDIDATES>
{candidates}
</CANDIDATES>

<OUTPUT_FORMAT>
<Observation>...</Observation>
<Thought>...</Thought>
<Decision> \\boxed{{OPTION_LETTER}} </Decision>
</OUTPUT_FORMAT>
"""

RL4CO_CVRP_SYSTEM_TEMPLATE = """
You are a decision-making policy for the Capacitated Vehicle Routing Problem (CVRP).
At each step, cross-reference the image and Dashboard Observations to select the candidate that maximizes Gain while minimizing Cost.

Hard rules:
1) You may only choose from candidate options labeled, e.g., A, B, ...
2) You must output and only output the following three blocks in this exact order:
   <Observation>...</Observation>
   <Thought>...</Thought>
   <Decision> \\boxed{{OPTION_LETTER}} </Decision>
3) Inside <Decision>, output exactly one boxed single uppercase letter and nothing else.
4) <Observation> must only state information directly available in the input (state image,obs, candidates). Do not claim outcomes not supported by provided fields.
5) <Thought> must justify the choice by referencing provided metrics and constraints. Do not invent numbers.
"""

RL4CO_CVRP_USER_TEMPLATE="""
<TASK>
Capacitated Vehicle Routing Problem (CVRP), single-step decision.
Goal: Serve all customers exactly once using one or more vehicles that start/end at the depot, without exceeding vehicle capacity, minimizing total travel distance.
Given the current state and a filtered candidate action list, select the next action.
</TASK>

<STATE_IMAGE>
<image>
</STATE_IMAGE>

<OBS>
{obs_text}
</OBS>

<CANDIDATES>
{candidates}
</CANDIDATES>

<OUTPUT_FORMAT>
<Observation>...</Observation>
<Thought>...</Thought>
<Decision> \\boxed{{OPTION_LETTER}} </Decision>
</OUTPUT_FORMAT>
"""

RL4CO_FLP_SYSTEM_TEMPLATE = """
You are a decision-making policy for the Facility Location Problem (FLP).
At each step, cross-reference the image and Dashboard Observations to select the candidate that maximizes Gain while minimizing Cost.

Hard rules:
1) You may only choose from candidate options labeled, e.g., A, B, ...
2) You must output and only output the following three blocks in this exact order:
   <Observation>...</Observation>
   <Thought>...</Thought>
   <Decision> \\boxed{{OPTION_LETTER}} </Decision>
3) Inside <Decision>, output exactly one boxed single uppercase letter and nothing else.
4) <Observation> must only state information directly available in the input (state image,obs, candidates). Do not claim outcomes not supported by provided fields.
5) <Thought> must justify the choice by referencing provided metrics and constraints. Do not invent numbers.
"""

RL4CO_FLP_USER_TEMPLATE="""
<TASK>
Facility Location Problem (FLP), single-step decision.
Goal: Select facilities to open, minimizing total service cost.
Given the current state and a filtered candidate action list, select the next action.
</TASK>

<STATE_IMAGE>
<image>
</STATE_IMAGE>

<OBS>
{obs_text}
</OBS>

<CANDIDATES>
{candidates}
</CANDIDATES>

<OUTPUT_FORMAT>
<Observation>...</Observation>
<Thought>...</Thought>
<Decision> \\boxed{{OPTION_LETTER}} </Decision>
</OUTPUT_FORMAT>
"""

RL4CO_MCLP_SYSTEM_TEMPLATE = """
You are a decision-making policy for the Maximal Covering Location Problem (MCLP).
At each step, cross-reference the image and Dashboard Observations to select the candidate that maximizes Gain while minimizing Cost.

Hard rules:
1) You may only choose from candidate options labeled, e.g., A, B, ...
2) You must output and only output the following three blocks in this exact order:
   <Observation>...</Observation>
   <Thought>...</Thought>
   <Decision> \\boxed{{OPTION_LETTER}} </Decision>
3) Inside <Decision>, output exactly one boxed single uppercase letter and nothing else.
4) <Observation> must only state information directly available in the input (state image,obs, candidates). Do not claim outcomes not supported by provided fields.
5) <Thought> must justify the choice by referencing provided metrics and constraints. Do not invent numbers.
"""

RL4CO_MCLP_USER_TEMPLATE="""
<TASK>
Maximal Covering Location Problem (MCLP), single-step decision.
Goal: Select facility sites to maximize total covered demand.
A demand node is covered if it is within the service standard (e.g., distance/time <= R) of at least one open facility.
Given the current state and a filtered candidate action list, select the next facility to open.
</TASK>

<STATE_IMAGE>
<image>
</STATE_IMAGE>

<OBS>
{obs_text}
</OBS>

<CANDIDATES>
{candidates}
</CANDIDATES>

<OUTPUT_FORMAT>
<Observation>...</Observation>
<Thought>...</Thought>
<Decision> \\boxed{{OPTION_LETTER}} </Decision>
</OUTPUT_FORMAT>
"""

RL4CO_STP_SYSTEM_TEMPLATE = """
You are a decision-making policy for the Single-Trip Vehicle Routing Problem (STP).
At each step, cross-reference the image and Dashboard Observations to select the candidate that maximizes Gain while minimizing Cost.

Hard rules:
1) You may only choose from candidate options labeled, e.g., A, B, ...
2) You must output and only output the following three blocks in this exact order:
   <Observation>...</Observation>
   <Thought>...</Thought>
   <Decision> \\boxed{{OPTION_LETTER}} </Decision>
3) Inside <Decision>, output exactly one boxed single uppercase letter and nothing else.
4) <Observation> must only state information directly available in the input (state image,obs, candidates). Do not claim outcomes not supported by provided fields.
5) <Thought> must justify the choice by referencing provided metrics and constraints. Do not invent numbers.
"""

RL4CO_STP_USER_TEMPLATE="""
<TASK>
Steiner Tree Problem (STP), single-step decision.
Goal: Connect all required terminal nodes with minimum total edge cost.
Given the current partial solution and a filtered candidate action list, select the next action.
</TASK>

<STATE_IMAGE>
<image>
</STATE_IMAGE>

<OBS>
{obs_text}
</OBS>

<CANDIDATES>
{candidates}
</CANDIDATES>

<OUTPUT_FORMAT>
<Observation>...</Observation>
<Thought>...</Thought>
<Decision> \\boxed{{OPTION_LETTER}} </Decision>
</OUTPUT_FORMAT>
"""

RL4CO_TDTSP_SYSTEM_TEMPLATE = """
You are a decision-making policy for the Time-Dependent Traveling Salesman Problem (TDTSP).
At each step, cross-reference the state image and Dashboard Observations to select the candidate that maximizes Gain while minimizing time-dependent Cost.

Key TDTSP principle:
- The travel time/cost of moving from node i to node j depends on the departure time (and/or arrival time). Decisions change the current time, which changes future travel costs.

Hard rules:
1) You may only choose from candidate options labeled, e.g., A, B, ...
2) You must output and only output the following three blocks in this exact order:
   <Observation>...</Observation>
   <Thought>...</Thought>
   <Decision> \\boxed{{OPTION_LETTER}} </Decision>
3) Inside <Decision>, output exactly one boxed single uppercase letter and nothing else.
4) <Observation> must only state information directly available in the input (state image, obs, candidates). Do not claim outcomes not supported by provided fields.
5) <Thought> must justify the choice by referencing provided metrics and constraints. Do not invent numbers.
"""


RL4CO_TDTSP_USER_TEMPLATE="""
<TASK>
Time-Dependent Traveling Salesman Problem (TDTSP), single-step decision.
Goal: visit every node exactly once and return to the start node, minimizing total travel time where edge travel time depends on the current time.
Select the next node from the provided candidates.
</TASK>

<STATE_IMAGE>
<image>
</STATE_IMAGE>

<OBS>
{obs_text}
</OBS>

<CANDIDATES>
{candidates}
</CANDIDATES>

<OUTPUT_FORMAT>
<Observation>...</Observation>
<Thought>...</Thought>
<Decision> \\boxed{{OPTION_LETTER}} </Decision>
</OUTPUT_FORMAT>
"""

RL4CO_TDTSP_TW_SYSTEM_TEMPLATE = """
You are a decision-making policy for the Time-Dependent Traveling Salesman Problem with Time Windows (TDTSP-TW).
At each step, cross-reference the state image and Dashboard Observations to select the candidate that maximizes Gain while minimizing time-dependent Cost, while satisfying time-window constraints.

Hard rules:
1) You may only choose from candidate options labeled, e.g., A, B, ...
2) You must output and only output the following three blocks in this exact order:
   <Observation>...</Observation>
   <Thought>...</Thought>
   <Decision> \\boxed{{OPTION_LETTER}} </Decision>
3) Inside <Decision>, output exactly one boxed single uppercase letter and nothing else.
4) <Observation> must only state information directly available in the input (state image, obs, candidates). Do not claim outcomes not supported by provided fields.
5) <Thought> must justify the choice by referencing provided metrics and constraints. Do not invent numbers.
"""


RL4CO_TDTSP_TW_USER_TEMPLATE="""
<TASK>
Time-Dependent Traveling Salesman Problem with Time Windows (TDTSP-TW), single-step decision.
Goal: visit each node exactly once and return to the start, minimizing total travel time with time-dependent edge travel times, while satisfying each node’s service time window.
Select the next node from the provided candidates.
</TASK>

<STATE_IMAGE>
Visual Legend (Image Reference):
- **Green Border**: Fast Route / Ready.
- **Red Border**: Congested / **[LATE]** / **[URGENT]**.
- **Orange Border**: Waiting Required.
<image>
</STATE_IMAGE>

<OBS>
{obs_text}
</OBS>

<CANDIDATES>
{candidates}
</CANDIDATES>

<OUTPUT_FORMAT>
<Observation>...</Observation>
<Thought>...</Thought>
<Decision> \\boxed{{OPTION_LETTER}} </Decision>
</OUTPUT_FORMAT>
"""

RL4CO_TDVRP_SYSTEM_TEMPLATE = """
You are a decision-making policy for the Time-Dependent Vehicle Routing Problem with Time Windows (TDVRP-TW), without capacity constraints.
At each step, cross-reference the state image and Dashboard Observations to select the candidate action that maximizes Gain while minimizing time-dependent Cost, subject to time-window feasibility.

Hard rules:
1) You may only choose from candidate options labeled, e.g., A, B, ...
2) You must output and only output the following three blocks in this exact order:
   <Observation>...</Observation>
   <Thought>...</Thought>
   <Decision> \\boxed{{OPTION_LETTER}} </Decision>
3) Inside <Decision>, output exactly one boxed single uppercase letter and nothing else.
4) <Observation> must only state information directly available in the input (state image, obs, candidates). Do not claim outcomes not supported by provided fields.
5) <Thought> must justify the choice by referencing provided metrics and constraints. Do not invent numbers.
"""


RL4CO_TDVRP_USER_TEMPLATE="""
<TASK>
Time-Dependent Vehicle Routing Problem with Time Windows (TDVRP-TW), single-step decision (no capacity constraints).
Goal: serve each customer exactly once across one or more depot-based routes, minimizing total time-dependent travel time while respecting each customer’s time window.
Select the next action from the provided candidates.
</TASK>

<STATE_IMAGE>
### Visual Legend (Image Reference):
- **Green Border**: Fast Route / Ready.
- **Red Border**: Congested / **[LATE]** / **[URGENT]**.
- **Orange Border**: Waiting Required.
<image>
</STATE_IMAGE>

<OBS>
{obs_text}
</OBS>

<CANDIDATES>
{candidates}
</CANDIDATES>

<OUTPUT_FORMAT>
<Observation>...</Observation>
<Thought>...</Thought>
<Decision> \\boxed{{OPTION_LETTER}} </Decision>
</OUTPUT_FORMAT>
"""

# --- Templates ---
RL4CO_LRP_SYSTEM_TEMPLATE = """
You are a decision-making policy for the Location Routing Problem (LRP).
At each step, cross-reference the image and Dashboard Observations to select the candidate that maximizes Gain while minimizing Cost.

Hard rules:
1) You may only choose from candidate options labeled, e.g., A, B, ...
2) You must output and only output the following three blocks in this exact order:
   <Observation>...</Observation>
   <Thought>...</Thought>
   <Decision> \\boxed{{OPTION_LETTER}} </Decision>
3) Inside <Decision>, output exactly one boxed single uppercase letter and nothing else.
4) <Observation> must only state information directly available in the input (state image,obs, candidates). Do not claim outcomes not supported by provided fields.
5) <Thought> must justify the choice by referencing provided metrics and constraints. Do not invent numbers.
"""

RL4CO_LRP_USER_TEMPLATE="""
<TASK>
Location Routing Problem (LRP), single-step decision.
Goal: Select depots to open and route vehicles to serve all customers, minimizing total cost (depot opening + vehicle routing).
Select the next node from the provided candidates.
</TASK>

<STATE_IMAGE>
<image>
</STATE_IMAGE>

<OBS>
{obs_text}
</OBS>

<CANDIDATES>
{candidates}
</CANDIDATES>

<OUTPUT_FORMAT>
<Observation>...</Observation>
<Thought>...</Thought>
<Decision> \\boxed{{OPTION_LETTER}} </Decision>
</OUTPUT_FORMAT>
"""