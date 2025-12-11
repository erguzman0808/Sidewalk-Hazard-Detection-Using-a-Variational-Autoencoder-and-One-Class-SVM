# train_vae_ablation.py
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split, Subset
from tqdm import tqdm
import wandb
import argparse
import os
import json
import numpy as np
from datetime import datetime

from VAE import CNN_VAE  # Ensure this is the correct import

def vae_loss(recon_x, x, mu, logvar, kld_weight=4.0):
    """
    VAE loss with configurable KLD weight for beta-VAE ablation
    """
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction='mean')
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kld_weight * kld_loss

def evaluate_model(model, val_loader, device, kld_weight=1.0):
    """
    Evaluate model on validation set
    """
    model.eval()
    val_loss = 0
    recon_loss_total = 0
    kld_loss_total = 0
    
    with torch.no_grad():
        for data, _ in val_loader:
            data = data.to(device)
            recon_batch, mu, logvar, _,  _= model(data) #        return decoded, mu, logvar, self.reparameterize(mu, logvar), encoded
            
            recon_loss = nn.functional.mse_loss(recon_batch, data, reduction='mean')
            kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            
            loss = recon_loss + kld_weight * kld_loss
            val_loss += loss.item()
            recon_loss_total += recon_loss.item()
            kld_loss_total += kld_loss.item()
    
    avg_val_loss = val_loss / len(val_loader.dataset)
    avg_recon_loss = recon_loss_total / len(val_loader.dataset)
    avg_kld_loss = kld_loss_total / len(val_loader.dataset)
    
    return avg_val_loss, avg_recon_loss, avg_kld_loss

def get_model_name(args):
    """
    Generate model name based on hyperparameters
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"vae_latent{args.latent_dim}_lr{args.learning_rate}_bs{args.batch_size}_kld{args.kld_weight}_data{args.data_percentage}_{timestamp}"
    return name

def main(args):
    # Generate model name
    model_name = get_model_name(args)
    
    # Initialize wandb with unique run name
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=model_name,
        config={
            "latent_dim": args.latent_dim,
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "kld_weight": args.kld_weight,
            "data_percentage": args.data_percentage,
            "num_epochs": args.num_epochs,
            #"encoder_channels": args.encoder_channels
        }
    )
    
    # Set device
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize VAE model with custom latent dimension
    vae = CNN_VAE(latent_dim=args.latent_dim)
    
    # DataParallel if multiple GPUs
    num_gpus = torch.cuda.device_count()
    print(f"Number of available GPUs: {num_gpus}")
    if device.type == 'cuda' and num_gpus > 1:
        vae = nn.DataParallel(vae)
        print(f"Using {num_gpus} GPUs for training")
    
    vae = vae.to(device)
    
    # Optimizer
    optimizer = optim.Adam(vae.parameters(), lr=args.learning_rate)
    
    # Data transforms
    transform = transforms.Compose([
        transforms.Resize((320, 240)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4500, 0.4552, 0.4417], std=[0.0930, 0.0883, 0.0936])
    ])
    
    # Load full dataset
    full_dataset = ImageFolder(root=args.data_path, transform=transform)
    
    # Calculate dataset sizes based on data_percentage ablation
    total_size = len(full_dataset)
    use_size = int(total_size * (args.data_percentage / 100.0))
    
    # First, select subset based on data_percentage
    if args.data_percentage < 100:
        indices = np.random.choice(total_size, use_size, replace=False)
        subset_dataset = Subset(full_dataset, indices)
    else:
        subset_dataset = full_dataset
    
    # Then split into train/val (80/20)
    train_size = int(0.8 * len(subset_dataset))
    val_size = len(subset_dataset) - train_size
    train_dataset, val_dataset = random_split(
        subset_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"Total dataset size: {total_size}")
    print(f"Using {args.data_percentage}% of data: {use_size} samples")
    print(f"Training samples: {train_size}")
    print(f"Validation samples: {val_size}")
    
    # Data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    # Training loop
    best_val_loss = float('inf')
    results = {
        'config': vars(args),
        'train_losses': [],
        'val_losses': [],
        'recon_losses': [],
        'kld_losses': [],
        'best_epoch': 0,
        'best_val_loss': float('inf')
    }
    
    for epoch in range(args.num_epochs):
        # Training phase
        vae.train()
        train_loss = 0
        train_recon_loss = 0
        train_kld_loss = 0
        
        with tqdm(train_loader, unit="batch") as tepoch:
            for data, _ in tepoch:
                tepoch.set_description(f"Epoch {epoch + 1}/{args.num_epochs}")
                data = data.to(device)
                optimizer.zero_grad()
                
                recon_batch, mu, logvar, latent_vector,_ = vae(data)
                
                # Calculate individual losses for logging
                recon_loss = nn.functional.mse_loss(recon_batch, data, reduction='mean')
                kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                loss = recon_loss + args.kld_weight * kld_loss
                
                loss.backward()
                train_loss += loss.item()
                train_recon_loss += recon_loss.item()
                train_kld_loss += kld_loss.item()
                optimizer.step()
                
                tepoch.set_postfix(loss=loss.item() / len(data))
        
        avg_train_loss = train_loss / len(train_loader.dataset)
        avg_train_recon = train_recon_loss / len(train_loader.dataset)
        avg_train_kld = train_kld_loss / len(train_loader.dataset)
        
        # Validation phase
        avg_val_loss, avg_val_recon, avg_val_kld = evaluate_model(
            vae, val_loader, device, args.kld_weight
        )
        
        # Store results
        results['train_losses'].append(avg_train_loss)
        results['val_losses'].append(avg_val_loss)
        results['recon_losses'].append(avg_val_recon)
        results['kld_losses'].append(avg_val_kld)
        
        # Log to wandb
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "train_recon_loss": avg_train_recon,
            "train_kld_loss": avg_train_kld,
            "val_loss": avg_val_loss,
            "val_recon_loss": avg_val_recon,
            "val_kld_loss": avg_val_kld,
        })
        
        print(f'Epoch {epoch + 1}/{args.num_epochs}:')
        print(f'  Train Loss: {avg_train_loss:.4f} (Recon: {avg_train_recon:.4f}, KLD: {avg_train_kld:.4f})')
        print(f'  Val Loss: {avg_val_loss:.4f} (Recon: {avg_val_recon:.4f}, KLD: {avg_val_kld:.4f})')
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            results['best_epoch'] = epoch + 1
            results['best_val_loss'] = best_val_loss
            
            # Unwrap DataParallel if necessary
            model_to_save = vae.module if isinstance(vae, nn.DataParallel) else vae
            
            # Save model
            model_path = os.path.join(args.output_dir, f'{model_name}_best.pth')
            torch.save(model_to_save.state_dict(), model_path)
            print(f'  → New best model saved! Val Loss: {best_val_loss:.4f}')
    
    # Save final model
    model_to_save = vae.module if isinstance(vae, nn.DataParallel) else vae
    final_model_path = os.path.join(args.output_dir, f'{model_name}_final.pth')
    torch.save(model_to_save.state_dict(), final_model_path)
    
    # Save results JSON
    results_path = os.path.join(args.output_dir, f'{model_name}_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f'\nTraining complete!')
    print(f'Best validation loss: {best_val_loss:.4f} at epoch {results["best_epoch"]}')
    print(f'Models saved to: {args.output_dir}')
    
    wandb.finish()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='VAE Ablation Study')
    
    # Model architecture parameters
    parser.add_argument('--latent-dim', type=int, default=1024, 
                        help='Latent dimension size (128, 256, 512, 1024, 2048)')
    
    # Training parameters
    parser.add_argument('--learning-rate', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=25,
                        help='Batch size')
    parser.add_argument('--num-epochs', type=int, default=100,
                        help='Number of epochs')
    parser.add_argument('--kld-weight', type=float, default=1.0,
                        help='KLD loss weight (beta-VAE parameter)')
    
    # Data parameters
    parser.add_argument('--data-percentage', type=float, default=100.0,
                        help='Percentage of data to use (25, 50, 75, 100)')
    parser.add_argument('--data-path', type=str, 
                        default='/n/netscratch/howe_lab_seas/Lab/eguzman/Data/output_images/Outdoor_Training/color',
                        help='Path to training data')
    parser.add_argument('--num-workers', type=int, default=8,
                        help='Number of data loading workers')
    
    # Output parameters
    parser.add_argument('--output-dir', type=str, default='./ablation_models',
                        help='Directory to save models and results')
    
    # Wandb parameters
    parser.add_argument('--wandb-project', type=str, default='VAE-Ablation-Study',
                        help='Wandb project name')
    parser.add_argument('--wandb-entity', type=str, default='erguzman',
                        help='Wandb entity name')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    main(args)
