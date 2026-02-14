

<h3 align="center">
<b>See to Solve: A Geometry-Aware Vision-Language Agent for Real-World Routing Problem</b>
</h3>

## 📝 Abstract

Despite being foundational in logistics, routing problems remain computationally challenging under complex, real-world constraints. Classical heuristics typically lack explicit modeling of data distributions, necessitating expert-driven operator design and extensive manual tuning to ensure algorithmic reliability. Conversely, neural-network-based methods frequently struggle with distribution shifts across diverse urban environments, leading to poor generalization in real-world deployments. To address these dual limitations, we introduce **See-to-Solve (STS)**, a novel framework that employs a Vision-Language Model (VLM) as an autonomous agent to construct solutions for complex routing problems in an autoregressive manner. At each step, STS integrates multimodal observations to capture geometric structures and constraint-specific signals. To ensure logical grounding and mitigate hallucinations, we enforce an **Observation--Thought--Decision** reasoning protocol. Our approach further utilizes a two-stage sim-to-real pipeline: supervised fine-tuning on synthetic datasets, followed by GRPO-based reinforcement learning for rapid adaptation to target domains.

Evaluation on two real-world routing problems across three cities demonstrates that STS consistently outperforms strong heuristic and neural baselines. Notably, STS exhibits robust cross-city generalization and superior scalability to large-scale instances, bridging the gap between theoretical optimization and real-world deployment.

**Code Repository:** [https://github.com/ShotDownDiane/verl-agent-co](https://github.com/ShotDownDiane/verl-agent-co)

---

## 🏗️ Foundations

This repository is built upon [verl-agent](https://github.com/volcengine/verl) and [rl4co](https://github.com/ai4co/rl4co). We extend these powerful frameworks to support multimodal reasoning and verifiable geometry for combinatorial optimization.

## 🚀 Key Features

| Feature | Description |
| :--- | :--- |
| **Multimodal Observation** | 👁️ **STS** generates **dual-modality inputs** (Text + Image) for routing tasks. <br>✅ **Text**: Structured prompt with precise numerical constraints (Time Windows, Costs). <br>✅ **Visual**: 224x224 semantic maps highlighting Depot, Candidates, Trajectory, and Geometric Relations (e.g., Convex Hull, Clusters). |
| **Verifiable Reasoning** | 🛡️ **Program-Verified Distillation**: A pipeline to generate high-quality Chain-of-Thought (CoT) data. <br>✅ **Geometry Engine**: Programmatically calculates ground-truth relations (e.g., "Is the expert action on the convex hull?", "Does it follow an angular sweep?"). <br>✅ **Verification Loop**: Filters VLM/LLM reasoning traces against hard geometric facts. |
| **Specialized Solvers** | 🚚 Supports a wide range of routing problems: <br>✅ **TSP / TSPTW**: Traveling Salesperson Problem (with Time Windows). <br>✅ **CVRP / CVRPTW**: Capacitated Vehicle Routing Problem. <br>✅ **Time-Dependent Variants**: Simulates traffic fluctuations. |
| **OTD Framework** | 🧠 Enforces a strict **Observation -> Thought -> Decision** reasoning structure. <br>✅ **Obs**: Visual/Geometric fact extraction. <br>✅ **Thk**: Strategic reasoning (e.g., "Rejecting closer node due to capacity"). <br>✅ **Dec**: Final action selection. |

## 📂 Project Structure

```
verl-agent-co/
├── agent_system/
│   ├── environments/env_package/rl4co/
│   │   ├── route_obs.py       # Core observation builders & visualizers
│   │   │   ├── build_obs_tdtsp_tw()  # Multimodal prompt construction for TSP
│   │   │   ├── build_obs_tdvrp()     # Multimodal prompt construction for VRP
│   │   │   ├── render_tdtsptw_smart_dual_view() # Semantic map rendering
│   │   │   └── render_tdvrp_smart_dual_view()
├── CoT_generator_verified/    # Program-Verified Reasoning Pipeline
│   ├── geometry_engine.py     # Geometric rule verification (Global/Trajectory/Location)
│   ├── generation_pipeline.py # Obs -> Thk -> Refine distillation loop
│   ├── prompts.py             # Verified reasoning prompts
│   └── main.py                # Pipeline entry point
├── examples/                  # Training & Inference scripts
│   └── grpo_trainer/prompt_agent/
│       ├── generate_cot_dataset_cvrp.py # Legacy generator
│       └── ...
```

## 🛠️ Usage

### 1. Generating Expert Trajectories
First, generate raw expert trajectories using heuristic solvers (LKH-3, HGS, etc.). This produces a JSON file containing states, actions, and candidates.

### 2. Running the Verification Pipeline
Use the `CoT_generator` module to distill high-quality CoT data from the expert trajectories.

```bash
python3 -m CoT_generator.main \
    --input_file "path/to/expert_trajectories.json" \
    --output_file "path/to/verified_cot_dataset.json"
```

This pipeline will:
1.  **Analyze Geometry**: Compute ground-truth relations (e.g., "Action A is the Nearest Neighbor").
2.  **Generate Observation**: Ask the VLM verifiable Yes/No questions to ensure it "sees" the geometry correctly.
3.  **Generate Thought**: Ask the LLM to reason about the expert action using the verified observation.
4.  **Refine**: Synthesize the output into a clean OTD format.

### 3. Visualizing Observations
The observation builders in `route_obs.py` automatically handle multimodal data:
*   **Text**: Includes "Task Instruction", "System Status" (Time, Trend, Workload), and "Candidate List" (Coordinates, Cost, Slack, Traffic).
*   **Image**: A 224x224 RGB image visualizing the route history, current location, and color-coded candidates (Red=Urgent, Green=Ready).

## 🧩 Supported Patterns (Geometry Engine)

The verification engine checks three levels of patterns:

*   **Global**: Clustered/Uniform distribution, Outliers.
*   **Trajectory**: Directional Trend (Outward/Returning), Convex Hull alignment, Angular Sweep (Counter-Clockwise), Isolated Pickups.
*   **Location**: High-density zones, Boundary nodes, Path intersections.

<!-- ## 📄 Citation

If you use this framework, please cite our work:

```bibtex
@article{see2solve2025,
  title={See to Solve: A Geometry-Aware Vision-Language Agent for Real-World Routing Problem},
  author={ShotDownDiane et al.},
  journal={arXiv preprint},
  year={2025},
  url={https://github.com/ShotDownDiane/verl-agent-co}
}
``` -->
