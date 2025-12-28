RL4CO_TSP_TEMPLATE_NO_HIS = """
The task is to identify the shortest route that visits every city exactly once and returns to the starting city. 

Each city is represented as a node with $(x, y)$ coordinates. The input provides the city coordinates, the $k$ nearest neighbors for each city, and their respective distances.Based on the provided data, determine the most logical next city to visit to minimize the total travel distance while respecting the TSP constraints.

Input:
{text_obs}

Solve the problem efficiently and clearly. Think step-by-step before answering. The last line of your response should be: 'Therefore, the final answer is: $\\boxed{{ANSWER}}$. I hope it is correct' where ANSWER is the ID/number of the next city.
"""


RL4CO_TSP_TEMPLATE = """
The task is to identify the shortest route that visits every city exactly once and returns to the starting city. 

Each city is represented as a node with $(x, y)$ coordinates. The input provides the city coordinates, the current trajectory, and a list of **Top-K candidate neighbors** (e.g., Option A, B, C...) based on distance.

Based on the provided data and candidates, determine the most logical next city to visit to minimize the total travel distance while respecting the TSP constraints.

Input:
{text_obs}

Which option is the best next step? 
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
Solve the Capacitated Vehicle Routing Problem (CVRP) with {num_customers} customers and 1 depot (node 0). Each customer node has a demand. All vehicles have the same capacity {vehicle_capacity}. You must assign each customer to exactly one route and ensure that the sum of demands on each route does not exceed the vehicle capacity. Minimize the total distance traveled.

The input includes city coordinates, the {k_nn} nearest neighbors for each city, and their respective distances. Provide the solution in the following format:
1. A list of routes, each route as an ordered list of visited nodes (start/end at the depot).
2. Objective: The total distance of all routes.

Input:
{node_coords_with_neighbors}

Reply with a JSON object:
{{
  "routes": [[<route 1 as list of node indices>], [<route 2 as list of node indices>], ...],
  "objective": <total distance of all routes>
}}
"""


RL4CO_CVRP_TEMPLATE = """
Solve the Capacitated Vehicle Routing Problem (CVRP) with {num_customers} customers and 1 depot (node 0). Each customer node has a demand. All vehicles have the same capacity {vehicle_capacity}. You must assign each customer to exactly one route and ensure that the sum of demands on each route does not exceed the vehicle capacity. Minimize the total distance traveled.

The input includes city coordinates, the {k_nn} nearest neighbors for each city, and their respective distances. Provide the solution in the following format:
1. A list of routes, each route as an ordered list of visited nodes (start/end at the depot).
2. Objective: The total distance of all routes.

Input:
{node_coords_with_neighbors}

Reply with a JSON object:
{{
  "routes": [[<route 1 as list of node indices>], [<route 2 as list of node indices>], ...],
  "objective": <total distance of all routes>
}}
"""


RL4CO_OP_TEMPLATE_NO_HIS = """
Solve the Orienteering Problem with {num_nodes} nodes. Each node has (x, y) coordinates and a prize for visiting it. You must plan a route that starts at depot {start_node}, collecting the maximum total prize possible, subject to a maximum route length T = {max_route_length:.1f}. You may visit a subset of nodes, but the total distance traveled must not exceed T.

The input includes city coordinates, the {k_nn} nearest neighbors for each city, and their respective distances. Provide the solution in the following format:
1. Route: The ordered list of visited nodes.
2. Objective: The objective value (summation of the collecting prizes).

Input:
{node_coords_with_neighbors}

Reply with a JSON object:
{{
  "route": [<list of node indices in visiting order>],
  "objective": <total collected prize>
}}
"""


RL4CO_OP_TEMPLATE = """
Solve the Orienteering Problem with {num_nodes} nodes. Each node has (x, y) coordinates and a prize for visiting it. You must plan a route that starts at depot {start_node}, collecting the maximum total prize possible, subject to a maximum route length T = {max_route_length:.1f}. You may visit a subset of nodes, but the total distance traveled must not exceed T.

The input includes city coordinates, the {k_nn} nearest neighbors for each city, and their respective distances. Provide the solution in the following format:
1. Route: The ordered list of visited nodes.
2. Objective: The objective value (summation of the collecting prizes).

Input:
{node_coords_with_neighbors}

Reply with a JSON object:
{{
  "route": [<list of node indices in visiting order>],
  "objective": <total collected prize>
}}
"""

