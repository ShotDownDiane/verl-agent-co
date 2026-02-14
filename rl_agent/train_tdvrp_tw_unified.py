
import argparse
import logging
import os
import torch
import swanlab
from lightning.pytorch.loggers import WandbLogger
from tensordict import TensorDict

from env.tdvrp.env import TDVRPEnv
from env.tdtsp.embeddings import TDTSPInitEmbedding, TDTSPContext
from rl4co.models.zoo.am import AttentionModelPolicy
from TMAT.matnet_time import MatNetTimePolicy
from rl4co.models.zoo.pomo import POMO
from rl4co.models.rl import REINFORCE
from rl4co.utils.trainer import RL4COTrainer

# Setup logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# --- Custom Classes ---

class TDVRPPolicy(AttentionModelPolicy):
    """
    Attention Model Policy adapted for TDVRP-TW.
    Uses TDTSPInitEmbedding and TDTSPContext (shared embeddings).
    """
    def __init__(self, embed_dim=128, num_locs=20, num_time_steps=37, **kwargs):
        # Pass "tsp" to avoid ValueError in init_embedding lookup
        super().__init__(
            env_name="tsp", 
            embed_dim=embed_dim,
            **kwargs
        )
        
        # Override embeddings with our custom ones for TDTSP/TDVRP
        self.encoder.init_embedding = TDTSPInitEmbedding(
            embed_dim, 
            num_locs=num_locs, 
            num_time_steps=num_time_steps
        )
        self.decoder.context_embedding = TDTSPContext(embed_dim)


class TDVRPMpVRPMatNetWrapper(TDVRPEnv):
    """
    Wrapper for TDVRP to work with MatNet.
    MatNet typically expects a full matrix input.
    """
    name = "tdvrp_tw_matnet"
    
    def _reset(self, td: TensorDict, **kwargs) -> TensorDict:
        out_td = super()._reset(td, **kwargs)
        return out_td


# --- Helper Functions ---

def get_paths(city, size):
    # Determine the project root dynamically
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir) # Go up one level to verl-agent-co
    data_root = os.path.join(project_root, "data")
    
    # TDVRP uses "tdtsp_dataset_random" path convention in existing scripts
    train_data_path = os.path.join(data_root, "tdvrptw", f"{city}_{size}_random_train.npz")
    valid_data_path = os.path.join(data_root, "tdvrptw", f"{city}_{size}_random_test.npz")
    base_data_path = os.path.join(data_root, "vrptdt-benchmark", "instances")
    matrix_path = os.path.join(data_root, "vrptdt-benchmark", "instances")
    
    # Instance path for dummy generator init (berlin_2000 is standard benchmark base)
    instance_path = os.path.join(data_root, "vrptdt-benchmark", "instances", "berlin_2000.json")
    matrix_file_path = os.path.join(data_root, "vrptdt-benchmark", "instances", "berlin_2000_tt.json.bz2")
    
    return train_data_path, valid_data_path, base_data_path, matrix_path, instance_path, matrix_file_path

def main():
    parser = argparse.ArgumentParser(description="Unified Training Script for TDVRP-TW")
    parser.add_argument("--model", type=str, required=True, choices=["am", "matnet", "pomo"], help="Model to train")
    parser.add_argument("--city", type=str, default="berlin", help="City name")
    parser.add_argument("--size", type=int, default=20, help="Number of locations")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size (default depends on model)")
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    
    args = parser.parse_args()
    
    # Defaults
    if args.batch_size is None:
        args.batch_size = 64 if args.model == "pomo" else 512
        
    num_locs = args.size + 1 # +1 for depot
    num_time_steps = 37
    train_path, valid_path, base_path, matrix_path, instance_path, matrix_file_path = get_paths(args.city, args.size)
    
    log.info(f"Training {args.model.upper()} on {args.city}-{args.size}")
    
    # 1. Environment Setup
    if args.model == "matnet":
        EnvClass = TDVRPMpVRPMatNetWrapper
    else:
        EnvClass = TDVRPEnv
        
    train_generator_params = {
        "data_path": train_path,
        "base_data_path": base_path,
        "matrix_path": matrix_path,
        "num_matrix_steps": num_time_steps,
    }

    valid_generator_params = {
        "data_path": valid_path,
        "base_data_path": base_path,
        "matrix_path": matrix_path,
        "num_matrix_steps": num_time_steps,
    }

    env_generator_params = {
        "instance_path": instance_path,
        "matrix_path": matrix_file_path,
        "num_nodes": num_locs,
        "num_matrix_steps": num_time_steps,
    }
        
    train_env = EnvClass(
        generator_params=train_generator_params,
        penalty_value=0.0 # Consistent with POMO implementation
    )
    
    # Dummy env for model initialization
    dummy_env = EnvClass(
        generator_params=env_generator_params,
        penalty_value=0.0
    )
    
    # 2. Policy Setup
    if args.model == "am":
        policy = TDVRPPolicy(
            embed_dim=128,
            num_locs=num_locs,
            num_time_steps=num_time_steps,
            num_encoder_layers=3,
            num_heads=8,
            normalization="instance"
        )
    elif args.model == "pomo":
        policy = TDVRPPolicy(
            embed_dim=128,
            num_locs=num_locs,
            num_time_steps=num_time_steps,
            num_encoder_layers=6, # POMO typically uses deeper encoder
            num_heads=8,
            normalization="instance",
            use_graph_context=False # POMO disables graph context
        )
    elif args.model == "matnet":
        policy = MatNetTimePolicy(
            env_name="atsp", # MatNet usually treats problem as ATSP-like matrix
            embed_dim=128,
            num_encoder_layers=3,
            num_heads=8,
            normalization="instance",
            init_embedding_kwargs={"mode": "Random"},
            num_matrix_steps=num_time_steps
        )
        
    # 3. Model Setup
    if args.model == "pomo":
        model = POMO(
            env=dummy_env,
            policy=policy,
            baseline="shared",
            num_starts=args.size, # num_starts = N (number of customers)
            batch_size=args.batch_size,
            val_batch_size=100,
            test_batch_size=1000,
            train_data_size=20000, 
            val_data_size=1000,
            optimizer_kwargs={"lr": args.lr, "weight_decay": 1e-6},
            dataloader_num_workers=2,
            train_file=train_path,
            val_file=valid_path,
        )
    else: # AM or MatNet
        model = REINFORCE(
            env=dummy_env,
            policy=policy,
            baseline="rollout", # Standard AM/MatNet baseline
            batch_size=args.batch_size,
            val_batch_size=100,
            test_batch_size=1000,
            train_data_size=20000, 
            val_data_size=1000,
            optimizer_kwargs={"lr": args.lr, "weight_decay": 1e-6},
            dataloader_num_workers=2,
            train_file=train_path,
            val_file=valid_path,
        )

    # 4. Logger
    wandb_logger = WandbLogger(
        project="TDVRP-TW",
        name=f"{args.model}-tdvrp-tw-{args.city}-{args.size}",
    )
    
    # 5. Trainer
    trainer = RL4COTrainer(
        max_epochs=args.epochs,
        accelerator="gpu",
        devices=args.gpus,  
        logger=wandb_logger,
        log_every_n_steps=100,
        enable_checkpointing=False,
        gradient_clip_val=1.0,
        check_val_every_n_epoch=5,
    )
    
    # 6. Fit
    trainer.fit(model)
    trainer.save_checkpoint(f"ckpt/tdvrptw/{args.model}/{args.city}_{args.size}.ckpt")
    
if __name__ == "__main__":
    main()
