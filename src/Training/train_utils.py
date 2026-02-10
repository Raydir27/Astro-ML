import torch
import gc

def nt_xent_loss(z1, z2, temperature=0.5):
    B = z1.size(0)
    z = torch.cat([z1, z2], dim=0)  # (2B, D)
    sim = torch.matmul(z, z.T) / temperature  # (2B, 2B)
    # Remove self-similarity
    mask = torch.eye(2 * B, device=z.device).bool()
    sim.masked_fill_(mask, -9e15)
    # Positive pairs: (i, i+B) and (i+B, i)
    pos = torch.cat([sim.diag(B), sim.diag(-B)])
    # For each anchor, denominator is all except itself
    exp_sim = torch.exp(sim)
    exp_sim_sum = exp_sim.sum(dim=1)
    loss = -torch.log(torch.exp(pos) / exp_sim_sum)
    return loss.mean()


def find_auto_batch_size(model, feature_dim, device, start_batch=2048, safety_margin=0.8):
    """
    Finds the max batch size by simulating a forward and backward pass.
    """
    if device == "cpu":
        return 64
    
    batch_size = start_batch
    model.to(device)
    
    # Use training mode to ensure gradients and activations are tracked
    model.train() 
    # Use a dummy optimizer to simulate memory used by optimizer states (Adam)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    print(f"Searching for optimal batch size starting from {start_batch}...")

    while batch_size > 1:
        try:
            optimizer.zero_grad()
            
            # 1. Simulate the "Double View" of SimCLR
            # (batch_size, dim) -> (2 * batch_size, dim)
            dummy_input = torch.randn(batch_size * 2, feature_dim).to(device)
            
            # 2. Forward pass
            output = model(dummy_input)
            
            # 3. Dummy loss and Backward pass (This is crucial!)
            loss = output.sum()
            loss.backward()
            
            # 4. Cleanup
            del dummy_input, output, loss
            torch.cuda.empty_cache()
            gc.collect()
            
            # Apply safety margin (e.g., 80% of max) to account for 
            # fragmentation and OS background tasks
            final_bs = int(batch_size * safety_margin)
            print(f"✅ Success! Max tested: {batch_size}. Selected with safety margin: {final_bs}")
            return final_bs

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                batch_size //= 2
                print(f"❌ OOM at {batch_size * 2}, trying {batch_size}...")
                torch.cuda.empty_cache()
                gc.collect()
            else:
                raise e
                
    return 1