
import unittest
from unittest.mock import MagicMock
import torch
import sys
import os
sys.dont_write_bytecode = True

# Add current directory to path
sys.path.append("/root/autodl-tmp/verl-agent-co/examples/prompt_agent")

class TestWorkerJSON(unittest.TestCase):
    def setUp(self):
        self.agent = MagicMock()
        self.agent.generate_thought_and_action.return_value = ("Thought", "0")
        
        self.envs = MagicMock()
        # obs should be {0: {'text': ..., 'image': ...}}
        mock_obs = {0: {'text': "observation text", 'image': None}}
        self.envs.reset.return_value = (mock_obs, {})
        # step returns obs, reward, done, info (4 values)
        self.envs.step.return_value = (mock_obs, torch.tensor([1.0]), torch.tensor([False]), {})
        
        # Handle hasattr(worker, 'env') check in FLP
        # If the code checks worker.env, we want it to use self.envs as well
        self.envs.env = self.envs

        # Mock tensordict
        self.envs._td = {
            'locs': torch.randn(1, 10, 2), # Batch size 1, 10 nodes, 2 coords
            'facility_locs': torch.randn(1, 10, 2),
            'topk_acts': [torch.tensor([0, 1, 2])], # List of tensors or tensor
            'current_node': torch.tensor([0]) # Current node index
        }
        
        # Mock create_envs to return self.envs (so reset/step are called on it)
        self.envs.create_envs.return_value = self.envs

    def test_flp_worker(self):
        from Qwen3_single_worker_flp import run_agent_loop
        print("\nTesting FLP Worker...")
        
        # FLP might use facility_locs
        self.envs._td = {
            'facility_locs': torch.randn(1, 10, 2),
            'topk_acts': [torch.tensor([0, 1, 2])]
        }
        
        mock_obs = {0: {'text': "observation text", 'image': None}}
        self.envs.step.side_effect = [
             (mock_obs, torch.tensor([0.0]), torch.tensor([False]), {}),
             (mock_obs, torch.tensor([1.0]), torch.tensor([True]), {})
        ]

        obs_list, image_list, trajectory, candidates_list, node_coords, pure_obs_path, final_sol_path = run_agent_loop(self.envs, self.agent, solution_tour=[0,1,2], instance_idx=0)
        self.assertIsInstance(node_coords, dict)
        self.assertTrue(len(candidates_list) > 0, "Candidates list should not be empty")
        # Check new images paths
        # They might be None if render failed (e.g. cv2 issue in mock environment) or if coords are missing
        # But we expect the return values to be unpacked correctly.

    def test_tsp_worker(self):
        from Qwen3_single_worker_tsp import run_agent_loop
        print("\nTesting TSP Worker...")
        
        # TSP uses 'locs'
        self.envs._td = {
            'locs': torch.randn(1, 10, 2),
            'topk_acts': [torch.tensor([0, 1, 2])],
            'current_node': torch.tensor([0])
        }
        
        mock_obs = {0: {'text': "observation text", 'image': None}}
        # Mock step to return done=True eventually
        self.envs.step.side_effect = [
            (mock_obs, torch.tensor([0.0]), torch.tensor([False]), {}),
            (mock_obs, torch.tensor([1.0]), torch.tensor([True]), {})
        ]

        obs_list, image_list, trajectory, candidates_list, node_coords, pure_obs_path, final_sol_path = run_agent_loop(self.envs, self.agent, solution_tour=[0,1,2], instance_idx=0)
        
        self.assertIsInstance(node_coords, dict)
        self.assertTrue(len(node_coords) > 0)
        self.assertIsInstance(trajectory, list)
        self.assertIsInstance(candidates_list, list)
        self.assertTrue(len(candidates_list) > 0, "Candidates list should not be empty")
        self.assertIn("node_coords", locals()) # Just checking unpacking worked

    def test_cvrp_worker(self):
        from Qwen3_single_worker_cvrp import run_agent_loop
        print("\nTesting CVRP Worker...")
        
        self.envs._td = {
            'locs': torch.randn(1, 10, 2),
            'demand': torch.rand(1, 10),
            'vehicle_capacity': torch.tensor([1.0]),
            'used_capacity': torch.tensor([0.0]),
            'action_mask': torch.zeros(1, 10),
            'topk_acts': [torch.tensor([0, 1, 2])],
            'current_node': torch.tensor([0])
        }
        
        mock_obs = {0: {'text': "observation text", 'image': None}}
        self.envs.step.side_effect = [
             (mock_obs, torch.tensor([0.0]), torch.tensor([False]), {}),
             (mock_obs, torch.tensor([1.0]), torch.tensor([True]), {})
        ]
        
        obs_list, image_list, trajectory, candidates_list, node_coords, pure_obs, final_sol = run_agent_loop(self.envs, self.agent, solution_tour=[0,1,2], instance_idx=0)
        self.assertIsInstance(node_coords, dict)
        self.assertTrue(len(candidates_list) > 0, "Candidates list should not be empty")

    def test_mclp_worker(self):
        from Qwen3_single_worker_mclp import run_agent_loop
        print("\nTesting MCLP Worker...")
        
        # MCLP might use facility_locs
        self.envs._td = {
            'facility_locs': torch.randn(1, 10, 2),
            'demand_locs': torch.randn(1, 10, 2),
            'topk_acts': [torch.tensor([0, 1, 2])],
            'coverage_radius': torch.tensor([0.1])
        }
        
        mock_obs = {0: {'text': "observation text", 'image': None}}
        self.envs.step.side_effect = [
             (mock_obs, torch.tensor([0.0]), torch.tensor([False]), {}),
             (mock_obs, torch.tensor([1.0]), torch.tensor([True]), {})
        ]

        obs_list, image_list, trajectory, candidates_list, node_coords, pure_obs_path, final_sol_path = run_agent_loop(self.envs, self.agent, solution_tour=[0,1,2], instance_idx=0)
        self.assertIsInstance(node_coords, dict)
        self.assertTrue(len(candidates_list) > 0, "Candidates list should not be empty")

    def test_stp_worker(self):
        from Qwen3_single_worker_stp import run_agent_loop
        print("\nTesting STP Worker...")
        
        self.envs._td = {
            'locs': torch.randn(1, 10, 2),
            'topk_acts': [torch.tensor([0, 1, 2])],
            'terminals': torch.tensor([0, 1, 0, 0, 1, 0, 0, 0, 0, 0]), # Mask style
            'edge_index': torch.tensor([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]]).T # (2, E)
        }
        
        mock_obs = {0: {'text': "observation text", 'image': None}}
        self.envs.step.side_effect = [
             (mock_obs, torch.tensor([0.0]), torch.tensor([False]), {}),
             (mock_obs, torch.tensor([1.0]), torch.tensor([True]), {})
        ]

        obs_list, image_list, trajectory, candidates_list, node_coords, pure_obs_path, final_sol_path = run_agent_loop(self.envs, self.agent, solution_tour=[0,1,2], instance_idx=0)
        self.assertIsInstance(node_coords, dict)
        self.assertTrue(len(candidates_list) > 0, "Candidates list should not be empty")

if __name__ == '__main__':
    unittest.main()
