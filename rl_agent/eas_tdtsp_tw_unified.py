import argparse
import torch
import swanlab
import math
import os
import time
import logging
import copy
from rl4co.models.rl import REINFORCE
from rl4co.models.zoo.pomo import POMO
from TMAT.matnet_time import MatNetTimePolicy

# Import from unified training script
# Ensure train_tdtsp_tw_unified.py is in the same directory
try:
    from train_tdtsp_tw_unified import TDTSPTWMatNetWrapper, TDTSPPolicy, TDTSPTWEnv, TDTSPTWGenerator, get_paths, TDTSPTWMatrixGenerator
except ImportError:
    # Fallback if import fails (e.g. strict import issues), though typically works in same dir
    from env.tdtsp.env_tw import TDTSPTWEnv, TDTSPTWGenerator
    from env.tdtsp.embeddings import TDTSPInitEmbedding, TDTSPContext
    from rl4co.models.zoo.am import AttentionModelPolicy
    from tensordict import TensorDict
    
    class TDTSPPolicy(AttentionModelPolicy):
        def __init__(self, embed_dim=128, num_locs=20, num_time_steps=37, **kwargs):
            super().__init__(env_name="tsp", embed_dim=embed_dim, **kwargs)
            self.encoder.init_embedding = TDTSPInitEmbedding(embed_dim, num_locs=num_locs, num_time_steps=num_time_steps)
            self.decoder.context_embedding = TDTSPContext(embed_dim)

    class TDTSPTWMatNetWrapper(TDTSPTWEnv):
        name = "tdtsp_tw_matnet"
        def _reset(self, td: TensorDict, **kwargs) -> TensorDict:
            out_td = super()._reset(td, **kwargs)
            return out_td
            
    def get_paths(city, size):
        # Determine the project root dynamically
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir) # Go up one level to verl-agent-co
        data_root = os.path.join(project_root, "data")
        
        train_data_path = os.path.join(data_root, "tdtsptw", f"{city}_{size}_train.npz")
        valid_data_path = os.path.join(data_root, "tdtsptw", f"{city}_{size}_test.npz")
        base_data_path = os.path.join(data_root, "vrptdt-benchmark", "instances")
        matrix_path = os.path.join(data_root, "vrptdt-benchmark", "instances")
        return train_data_path, valid_data_path, base_data_path, matrix_path

# Setup logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

def run_eas():
    parser = argparse.ArgumentParser(description="Unified EAS Script for TDTSP-TW")
    parser.add_argument("--model", type=str, required=True, choices=["am", "matnet", "pomo"], help="Model type")
    parser.add_argument("--city", type=str, default="berlin", help="City name")
    parser.add_argument("--size", type=int, default=20, help="Number of locations")
    parser.add_argument("--ckpt_path", type=str, default=None, help="Path to checkpoint")
    parser.add_argument("--eas_lr", type=float, default=1e-4, help="EAS learning rate")
    parser.add_argument("--eas_steps", type=int, default=100, help="EAS steps per instance/batch") 
    parser.add_argument("--eas_batch_size", type=int, default=64, help="EAS batch size")
    
    args = parser.parse_args()

    # Paths
    if args.ckpt_path is None:
        # Default checkpoint naming convention
        # Check if model specific file exists first
        candidates = [
            f"tdtsptw_{args.model}_{args.city}_{args.size}.ckpt",
            f"last.ckpt", # Generic
            f"{args.model}-tdtsp-tw-{args.city}-{args.size}.ckpt"
        ]
        args.ckpt_path = candidates[0] # Default to first
        
    train_path, valid_path, base_path, matrix_path = get_paths(args.city, args.size)
    num_locs = args.size + 1
    num_time_steps = 37

    # Initialize SwanLab
    experiment_name=f"eas-{args.model}-{args.city}-{args.size}"
    if args.ckpt_path == "random":
        experiment_name += "-random"
        
    swanlab.init(
        project="TDTSP-TW-0207",
        experiment_name=experiment_name,
        config=vars(args)
    )

    # Environment Setup
    if args.model == "matnet":
        EnvClass = TDTSPTWMatNetWrapper
    else:
        EnvClass = TDTSPTWEnv
        
    # Validation Env (Test Set)
    valid_env = EnvClass(
        data_file_path=valid_path,
        base_data_path=base_path,
        matrix_path=matrix_path,
        penalty_value=3.0,
    )
    
    # Train Env (for EAS)
    train_env = EnvClass(
        data_file_path=train_path,
        base_data_path=base_path,
        matrix_path=matrix_path,
        penalty_value=3.0,
    )

    # Dummy Env for Initialization
    dummy_env = EnvClass(
        generator=TDTSPTWMatrixGenerator(num_loc=num_locs, num_time_steps=num_time_steps),
        penalty_value=3.0
    )

    # Policy Setup
    if args.model == "am":
        policy = TDTSPPolicy(
            embed_dim=128,
            num_locs=num_locs,
            num_time_steps=num_time_steps,
            num_encoder_layers=3,
            num_heads=8,
            normalization="instance"
        )
    elif args.model == "pomo":
        policy = TDTSPPolicy(
            embed_dim=128,
            num_locs=num_locs,
            num_time_steps=num_time_steps,
            num_encoder_layers=6,
            num_heads=8,
            normalization="instance",
            use_graph_context=False
        )
    elif args.model == "matnet":
        policy = MatNetTimePolicy(
            env_name="atsp",
            embed_dim=128,
            num_encoder_layers=3,
            num_heads=8,
            normalization="instance",
            init_embedding_kwargs={"mode": "Random"},
            num_matrix_steps=num_time_steps
        )

    # Load Checkpoint
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if os.path.exists(args.ckpt_path):
        log.info(f"Loading checkpoint from {args.ckpt_path}")
        try:
            checkpoint = torch.load(args.ckpt_path, map_location=lambda storage, loc: storage, weights_only=False)
            if "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
                # Handle prefixes
                new_state_dict = {}
                for k, v in state_dict.items():
                    key = k
                    if key.startswith("model.policy."):
                        key = key[13:]
                    elif key.startswith("policy."):
                        key = key[7:]
                    elif key.startswith("model."):
                        key = key[6:]
                        
                    new_state_dict[key] = v
                
                # Filter out unexpected keys
                policy_keys = set(policy.state_dict().keys())
                filtered_state_dict = {k: v for k, v in new_state_dict.items() if k in policy_keys}
                
                policy.load_state_dict(filtered_state_dict, strict=False)
            else:
                policy.load_state_dict(checkpoint, strict=False)
        except Exception as e:
            log.error(f"Failed to load checkpoint: {e}")
            return
    else:
        log.warning(f"Checkpoint {args.ckpt_path} not found. Using random initialization.")

    policy = policy.to(device)
    policy.eval()

    # Initial Validation
    log.info("Running initial validation on Test Set...")
    valid_env.generator.phase = "all"
    valid_env.generator._load_data()
    valid_env.generator.random_sample = False
    valid_env.generator.current_instance_id = 0
    
    num_samples = valid_env.generator.num_samples
    val_batch_size = 50 # Safe batch size
    num_steps = math.ceil(num_samples / val_batch_size)
    
    val_rewards = []
    decode_type_val = "multistart_greedy" if args.model == "pomo" else "greedy"
    num_starts = num_locs if args.model == "pomo" else 0
    
    with torch.no_grad():
        for i in range(num_steps):
            current_bs = min(val_batch_size, num_samples - i * val_batch_size)
            td = valid_env.reset(batch_size=[current_bs]).to(device)
            out = policy(td, valid_env, decode_type=decode_type_val, num_starts=num_starts)
            
            rewards = out["reward"]
            if args.model == "pomo":
                # [B * N] -> [B, N] -> max -> [B]
                rewards = rewards.view(current_bs, num_starts).max(dim=1)[0]
                
            val_rewards.append(rewards)
            
    avg_val_reward = torch.cat(val_rewards).mean().item()
    log.info(f"Initial Validation Reward: {avg_val_reward:.4f}")
    swanlab.log({"initial_val_reward": avg_val_reward})

    # EAS on Train Data
    log.info("Starting EAS on Train Data...")
    train_env.generator.phase = "all"
    train_env.generator._load_data()
    train_env.generator.random_sample = False
    train_env.generator.current_instance_id = 0
    
    num_train_samples = train_env.generator.num_samples
    max_eas_samples = 2000
    log.info(f"Limiting EAS to first {max_eas_samples} samples.")
    
    train_steps = math.ceil(min(num_train_samples, max_eas_samples) / args.eas_batch_size)
    
    policy.train()
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.eas_lr)
    
    total_initial_reward = 0
    total_improved_reward = 0
    
    for i in range(train_steps):
        current_bs = min(args.eas_batch_size, max_eas_samples - i * args.eas_batch_size)
        td_init = train_env.reset(batch_size=[current_bs]).to(device)
        
        # Initial Batch Performance
        with torch.no_grad():
            out_init = policy(td_init.clone(), train_env, decode_type=decode_type_val, num_starts=num_starts)
            r_init = out_init["reward"]
            if args.model == "pomo":
                r_init = r_init.view(current_bs, num_starts).max(dim=1)[0]
            total_initial_reward += r_init.sum().item()

        # EAS Inner Loop
        for step in range(args.eas_steps):
            optimizer.zero_grad()
            
            if args.model == "pomo":
                # POMO Logic
                out = policy(td_init.clone(), train_env, decode_type="sampling", num_starts=num_starts)
                rewards = out["reward"] # [B*N]
                log_probs = out["log_likelihood"] # [B*N]
                
                rewards_reshaped = rewards.view(current_bs, num_starts)
                baseline = rewards_reshaped.mean(dim=1, keepdim=True)
                advantage = rewards_reshaped - baseline
                
                loss = -(advantage * log_probs.view(current_bs, num_starts)).mean()
                
            else:
                # AM/MatNet Logic (Sampling + REINFORCE)
                out = policy(td_init.clone(), train_env, decode_type="sampling")
                rewards = out["reward"]
                log_probs = out["log_likelihood"]
                
                baseline = rewards.mean()
                advantage = rewards - baseline
                loss = -(advantage * log_probs).mean()
            
            loss.backward()
            optimizer.step()
            
        # Final Batch Performance
        with torch.no_grad():
            out_final = policy(td_init.clone(), train_env, decode_type=decode_type_val, num_starts=num_starts)
            r_final = out_final["reward"]
            if args.model == "pomo":
                r_final = r_final.view(current_bs, num_starts).max(dim=1)[0]
            total_improved_reward += r_final.sum().item()
            
        log.info(f"Batch {i+1}: Avg Reward {r_init.mean().item():.4f} -> {r_final.mean().item():.4f}")
        
    avg_init = total_initial_reward / min(num_train_samples, max_eas_samples)
    avg_imp = total_improved_reward / min(num_train_samples, max_eas_samples)
    
    log.info(f"EAS Completed. Initial: {avg_init:.4f}, Improved: {avg_imp:.4f}")
    swanlab.log({
        "eas_initial_reward": avg_init,
        "eas_improved_reward": avg_imp,
        "eas_improvement": avg_imp - avg_init
    })

    # Final Validation with greedy decoding (start from 0)
    log.info("Running Final Validation on Test Set (Greedy)...")
    valid_env.generator.phase = "all"
    valid_env.generator._load_data() # Ensure data is ready
    valid_env.generator.current_instance_id = 0 # Reset pointer
    
    val_rewards_final = []
    # Force greedy decoding to ensure single trajectory starting from depot (node 0)
    decode_type_final = "greedy"
    num_starts_final = 0
    
    total_inference_time = 0.0
    
    with torch.no_grad():
        for i in range(num_steps):
            current_bs = min(val_batch_size, num_samples - i * val_batch_size)
            td = valid_env.reset(batch_size=[current_bs]).to(device)
            
            start_t = time.time()
            out = policy(td, valid_env, decode_type=decode_type_final, num_starts=num_starts_final)
            end_t = time.time()
            total_inference_time += (end_t - start_t)
            
            rewards = out["reward"]
            val_rewards_final.append(rewards)
            
    avg_val_reward_final = torch.cat(val_rewards_final).mean().item()
    avg_time_per_instance = total_inference_time / num_samples
    
    log.info(f"Final Validation Reward (Greedy): {avg_val_reward_final:.4f}")
    log.info(f"Average Inference Time per Instance: {avg_time_per_instance:.4f}s")
    
    swanlab.log({
        "final_val_reward_greedy": avg_val_reward_final,
        "avg_inference_time": avg_time_per_instance
    })

if __name__ == "__main__":
    run_eas()
