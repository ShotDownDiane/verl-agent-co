RL4CO_TSP_TEMPLATE_COT = """
The task is to identify the shortest route that visits every city exactly once and returns to the starting city. 

Each city is represented as a node with $(x, y)$ coordinates. The input provides the city coordinates, the $k$ nearest neighbors for each city, and their respective distances.Based on the provided data, determine the most logical next city to visit to minimize the total travel distance while respecting the TSP constraints.

Input:
{text_obs}
Solve the problem efficiently and clearly. Think step-by-step before answering. The last line of your response should be: 'Therefore, the final answer is: \\boxed{{ANSWER}}'. where ANSWER is the ID/number of the next city.
"""


RL4CO_TSP_TEMPLATE = """
The task is to identify the shortest route that visits every city exactly once and returns to the starting city. 

Each city is represented as a node with $(x, y)$ coordinates. The input provides the city coordinates, the current trajectory, and a list of **Top-K candidate neighbors** (e.g., Option A, B, C...) based on distance.

Based on the provided data and candidates, determine the most logical next city to visit to minimize the total travel distance while respecting the TSP constraints.

Input:
{text_obs}
Response strictly in the format: "Therefore, the final answer is: \\boxed{{OPTION_LETTER}}".
"""


__ = """
Reply with a JSON object:
{{
  "route": [<list of node indices in visiting order>],
  "objective": <total travel distance>
}}
"""


RL4CO_CVRP_TEMPLATE_NO_HIS = """
Solve the Capacitated Vehicle Routing Problem (CVRP). Each customer node has a demand. All vehicles have the same capacity. You must assign each customer to exactly one route and ensure that the sum of demands on each route does not exceed the vehicle capacity. Minimize the total distance traveled.

Input:
{text_obs}

Reply with a JSON object:
{{
  "routes": [[<route 1 as list of node indices>], [<route 2 as list of node indices>], ...],
  "objective": <total distance of all routes>
}}
"""


RL4CO_CVRP_TEMPLATE = """
Solve the Capacitated Vehicle Routing Problem (CVRP). Each customer node has a demand. All vehicles have the same capacity. You must assign each customer to exactly one route and ensure that the sum of demands on each route does not exceed the vehicle capacity. Minimize the total distance traveled.

Input:
{text_obs}

Which option is the best next step? 
Response strictly in the format: "Therefore, the final answer is: \\boxed{{OPTION_LETTER}}".
"""


RL4CO_OP_TEMPLATE_NO_HIS = """
Solve the Orienteering Problem (OP). Each node has coordinates and a prize. You must plan a route that starts at the depot, collecting the maximum total prize possible, subject to a maximum route length T.

Input:
{text_obs}

Reply with a JSON object:
{{
  "route": [<list of node indices in visiting order>],
  "objective": <total collected prize>
}}
"""


RL4CO_OP_TEMPLATE = """
Solve the Orienteering Problem (OP). Each node has coordinates and a prize. You must plan a route that starts at the depot, collecting the maximum total prize possible, subject to a maximum route length T.

Input:
{text_obs}

Which option is the best next step? 
Response strictly in the format: "Therefore, the final answer is: \\boxed{{OPTION_LETTER}}".
"""

RL4CO_FLP_TEMPLATE_NO_HIS = """
Solve the Facility Location Problem (FLP). You need to select a subset of facilities to open to minimize the total cost (opening cost + transportation cost).

Input:
{text_obs}

Reply with a JSON object:
{{
  "facilities": [<list of opened facility indices>],
  "objective": <total cost>
}}
"""

RL4CO_FLP_TEMPLATE = """
Solve the Facility Location Problem (FLP). You need to select a subset of facilities to open to minimize the total cost.

Input:
{text_obs}
Response strictly in the format: "Therefore, the final answer is: \\boxed{{OPTION_LETTER}}".
"""

RL4CO_FLP_TEMPLATE_COT = """
Solve the Facility Location Problem (FLP). You need to select a subset of facilities to open to minimize the total cost.

Input:
{text_obs}
Solve the problem efficiently and clearly. Think step-by-step before answering. The last line of your response should be: "Therefore, the final answer is: \\boxed{{OPTION_LETTER}}".
"""

RL4CO_MCLP_TEMPLATE_NO_HIS = """
Solve the Maximal Covering Location Problem (MCLP). You need to select p facilities to maximize the total covered demand.

Input:
{text_obs}

Reply with a JSON object:
{{
  "facilities": [<list of selected facility indices>],
  "objective": <total covered demand>
}}
"""

RL4CO_MCLP_TEMPLATE = """
Solve the Maximal Covering Location Problem (MCLP). You need to select p facilities to maximize the total covered demand.

Input:
{text_obs}

Which option is the best next step? 
Response strictly in the format: "Therefore, the final answer is: \\boxed{{OPTION_LETTER}}".
"""

RL4CO_STP_TEMPLATE_NO_HIS = """
Solve the Steiner Tree Problem (STP). You need to connect all terminal nodes using a minimum cost tree.

Input:
{text_obs}

Reply with a JSON object:
{{
  "edges": [[u, v], ...],
  "objective": <total cost>
}}
"""

RL4CO_STP_TEMPLATE = """
Solve the Steiner Tree Problem (STP). You need to connect all terminal nodes using a minimum cost tree.

Input:
{text_obs}

Which option is the best next step? 
Response strictly in the format: "Therefore, the final answer is: \\boxed{{OPTION_LETTER}}".
"""
