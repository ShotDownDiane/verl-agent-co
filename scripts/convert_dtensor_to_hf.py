import torch
import os
import argparse
import sys
from transformers import AutoConfig, AutoModelForVision2Seq
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.placement_types import Shard, Replicate, Partial

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()
    
    # Load shards
    # Assuming world_size=2
    path0 = os.path.join(args.checkpoint_path, "model_world_size_2_rank_0.pt")
    path1 = os.path.join(args.checkpoint_path, "model_world_size_2_rank_1.pt")
    
    if not os.path.exists(path0) or not os.path.exists(path1):
        print(f"Checkpoint files not found in {args.checkpoint_path}")
        print(f"Expected model_world_size_2_rank_0.pt and model_world_size_2_rank_1.pt")
        sys.exit(1)
        
    print(f"Loading rank 0 from {path0}...")
    state0 = torch.load(path0, map_location="cpu", weights_only=False)
    print(f"Loading rank 1 from {path1}...")
    state1 = torch.load(path1, map_location="cpu", weights_only=False)
    
    full_state_dict = {}
    
    keys = list(state0.keys())
    print(f"Merging {len(keys)} tensors...")
    
    for key in keys:
        val0 = state0[key]
        val1 = state1[key]
        
        # Check type
        if isinstance(val0, DTensor):
            # Check placement
            placements = val0.placements
            # Assuming 1D mesh for FSDP
            placement = placements[0]
            
            t0 = val0.to_local()
            t1 = val1.to_local()
            
            if isinstance(placement, Shard):
                dim = placement.dim
                # Concatenate
                # Note: rank 0 is usually first? Mesh is [0, 1].
                full_tensor = torch.cat([t0, t1], dim=dim)
            elif isinstance(placement, Replicate):
                # They should be equal
                full_tensor = t0
            elif isinstance(placement, Partial):
                # Partial usually means it needs reduction (sum)
                # But for parameters, it shouldn't be Partial unless gradients?
                print(f"Warning: Partial placement for {key}. Summing.")
                full_tensor = t0 + t1 # Assuming Sum
            else:
                print(f"Unknown placement for {key}: {placement}")
                full_tensor = t0 # Fallback
                
            full_state_dict[key] = full_tensor
        else:
            # Regular tensor
            full_state_dict[key] = val0
            
    print("Merging complete. Saving...")
    
    # Load config and save
    try:
        config = AutoConfig.from_pretrained(args.checkpoint_path, trust_remote_code=True)
    except Exception as e:
        print(f"Error loading config: {e}")
        return

    os.makedirs(args.output_path, exist_ok=True)
    
    # Init empty model
    from accelerate import init_empty_weights
    print("Initializing empty model...")
    with init_empty_weights():
        model = AutoModelForVision2Seq.from_config(config, trust_remote_code=True)
    
    print("Saving pretrained...")
    # save_pretrained with state_dict
    model.save_pretrained(args.output_path, state_dict=full_state_dict)
    config.save_pretrained(args.output_path)
    
    # Processor
    try:
        from transformers import AutoProcessor
        proc = AutoProcessor.from_pretrained(args.checkpoint_path, trust_remote_code=True)
        proc.save_pretrained(args.output_path)
        print("Processor saved.")
    except Exception as e:
        print(f"Processor save failed: {e}")
        try:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(args.checkpoint_path, trust_remote_code=True)
            tok.save_pretrained(args.output_path)
            print("Tokenizer saved.")
        except:
            pass
            
    print(f"Conversion complete. Saved to {args.output_path}")

if __name__ == "__main__":
    main()
