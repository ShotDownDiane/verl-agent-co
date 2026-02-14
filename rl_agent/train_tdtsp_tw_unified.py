
import argparse
import logging
import os
import torch
import swanlab
from lightning.pytorch.loggers import WandbLogger
from tensordict import TensorDict

from env.tdtsp.env_tw import TDTSPTWEnv, TDTSPTWGenerator
from env.tdtsp.generator_matrix import TDTSPTWGenerator as TDTSPTWMatrixGenerator
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

class TDTSPPolicy(AttentionModelPolicy):
    """
    Attention Model Policy adapted for TDTSP-TW.
    Uses TDTSPInitEmbedding and TDTSPContext.
    """
    def __init__(self, embed_dim=128, num_locs=20, num_time_steps=37, **kwargs):
        # Pass "tsp" to avoid ValueError in init_embedding lookup
        super().__init__(
            env_name="tsp", 
            embed_dim=embed_dim,
            **kwargs
        )
        
        # Override embeddings with our custom ones for TDTSP
        self.encoder.init_embedding = TDTSPInitEmbedding(
            embed_dim, 
            num_locs=num_locs, 
            num_time_steps=num_time_steps
        )
        self.decoder.context_embedding = TDTSPContext(embed_dim)


class TDTSPTWMatNetWrapper(TDTSPTWEnv):
    """
    Wrapper for TDTSPTW to work with MatNet.
    - Adds 'cost_matrix' to observations for MatNet.
    - Uses the mean of the time-dependent travel time matrix as the static cost matrix.
    """
    name = "tdtsp_tw_matnet"
    
    def _reset(self, td: TensorDict, **kwargs) -> TensorDict:
        # Call original reset to get the standard TDTSPTW observations
        # This includes "travel_time_matrix" [B, N, N, T]
        out_td = super()._reset(td, **kwargs)
        # MatNetTime handles the matrix directly from 'travel_time_matrix' usually,
        # but if specific keys are needed by the base MatNet, they should be added here.
        return out_td


# --- Helper Functions ---

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

def main():
    parser = argparse.ArgumentParser(description="Unified Training Script for TDTSP-TW")
    parser.add_argument("--model", type=str, required=True, choices=["am", "matnet", "pomo"], help="Model to train")
    parser.add_argument("--city", type=str, default="berlin", help="City name")
    parser.add_argument("--size", type=int, default=20, help="Number of locations")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size (default depends on model)")
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--baseline", type=str, default="exponential",
                        choices=["rollout", "exponential", "shared", "mean"],
                        help="Baseline type (rollout is best but slowest, exponential/shared are faster)")
    parser.add_argument("--train_data_size", type=int, default=20000,
                        help="Training dataset size per epoch (reduce for faster epoch transitions)")
    parser.add_argument("--val_data_size", type=int, default=1000,
                        help="Validation dataset size for baseline evaluation")

    args = parser.parse_args()
    
    # Defaults
    if args.batch_size is None:
        args.batch_size = 64 if args.model == "pomo" else 512
        
    num_locs = args.size + 1 # +1 for depot
    num_time_steps = 37
    train_path, valid_path, base_path, matrix_path = get_paths(args.city, args.size)
    
    log.info(f"Training {args.model.upper()} on {args.city}-{args.size}")
    
    # 1. Environment Setup
    if args.model == "matnet":
        EnvClass = TDTSPTWMatNetWrapper
    else:
        EnvClass = TDTSPTWEnv
        
    train_env = EnvClass(
        data_file_path=train_path,
        base_data_path=base_path,
        matrix_path=matrix_path,
        penalty_value=3.0,
        phase="all" # Use all data from the file without splitting
    )
    
    valid_env = EnvClass(
        data_file_path=valid_path,
        base_data_path=base_path,
        matrix_path=matrix_path,
        penalty_value=3.0,
        phase="all" # Use all data from the file
    )
    
    # Monkey-patch train_env to handle validation dataset generation
    # This allows using separate files for train and validation
    original_train_generator = train_env.generator
    def hybrid_dataset(batch_size=[], phase="train", filename=None):
        if phase == "train":
            return original_train_generator(batch_size)
        else:
            return valid_env.generator(batch_size)
    
    # Bind the method to train_env instance
    train_env.dataset = hybrid_dataset
    
    # 2. Policy Setup
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
            num_encoder_layers=3, # POMO typically uses deeper encoder
            num_heads=8,
            normalization="instance",
            use_graph_context=False # POMO disables graph context
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
        
    # 3. Model Setup
    if args.model == "pomo":
        model = POMO(
            env=train_env,
            policy=policy,
            baseline="shared",
            num_starts=args.size, # num_starts = N (number of customers)
            batch_size=args.batch_size,
            val_batch_size=200,
            test_batch_size=10000,
            train_data_size=args.train_data_size,
            val_data_size=args.val_data_size,
            optimizer_kwargs={"lr": args.lr},
            dataloader_num_workers=1,
        )
    else: # AM or MatNet
        model = REINFORCE(
            env=train_env,
            policy=policy,
            baseline=args.baseline,  # Use baseline from args
            batch_size=args.batch_size,
            val_batch_size=200,
            test_batch_size=10000,
            train_data_size=args.train_data_size,
            val_data_size=args.val_data_size,
            optimizer_kwargs={"lr": args.lr},
            dataloader_num_workers=1,
            baseline_kwargs={"bl_alpha": 0.05},  # Keep rollout baseline params
        )

    # 4. Logger
    wandb_logger = WandbLogger(
        project="TDTSP-TW",
        name=f"{args.model}-tdtsp-tw-{args.city}-{args.size}",
    )
    
    # 5. Trainer
    trainer = RL4COTrainer(
        max_epochs=args.epochs,
        accelerator="gpu",
        devices=args.gpus,  
        logger=wandb_logger,
        log_every_n_steps=100,
        check_val_every_n_epoch=5,
    )
    
    # 6. Fit
    trainer.fit(model)
    trainer.save_checkpoint(f"ckpt/tdtsptw/{args.model}/{args.city}_{args.size}.ckpt")
    
    # 7. SwanLab Logging (Optional validation summary)
    # Note: RL4COTrainer handles validation logging to WandB automatically.
    # We can add SwanLab here if needed for specific metrics, but standard trainer uses PL loggers.

if __name__ == "__main__":
    main()
