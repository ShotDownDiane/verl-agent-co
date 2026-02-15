export interface Node {
    id: number;
    type: "depot" | "customer";
    lat: number;
    lon: number;
    x: number;
    y: number;
    demand: number;
}

export interface State {
    nodes: Node[];
    current_path: number[];
    current_cost: number;
    capacity: number;
    remaining_capacity: number;
    mode: "real" | "virtual";
    logs: string[];
}
