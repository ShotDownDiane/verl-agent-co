RL4CO_TSP_TEMPLATE_NO_HIS = """
Solve the Traveling Salesman Problem (TSP) for the given list of {num_nodes} cities. Each city is represented as a node with coordinates (x, y). Identify the shortest route that visits every city exactly once and returns to the starting city. The input includes city coordinates, the {k_nn} nearest neighbors for each city, and their respective distances. Provide the solution in the following format:

1. Route: List the nodes in the order they are visited.
2. Objective: The objective value (total travel distance).

Input:
{city_coords_with_neighbors}

Reply with a JSON object:
{{
  "route": [<list of node indices in visiting order>],
  "objective": <total travel distance>
}}
"""


RL4CO_TSP_TEMPLATE = """
Solve the Traveling Salesman Problem (TSP) for the given list of {num_nodes} cities. Each city is represented as a node with coordinates (x, y). Identify the shortest route that visits every city exactly once and returns to the starting city. The input includes city coordinates, the {k_nn} nearest neighbors for each city, and their respective distances. Provide the solution in the following format:

1. Route: List the nodes in the order they are visited.
2. Objective: The objective value (total travel distance).

Input:
{city_coords_with_neighbors}
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

