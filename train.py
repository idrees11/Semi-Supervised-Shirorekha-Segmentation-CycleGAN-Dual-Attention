
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import yaml
import os
import gc
from models.cyclegan import SemiSupervisedCycleGAN
from datasets.semi_supervised_dataset import SemiSupervisedDataset
from utils.metrics import evaluate
from utils.visualization import visualize_samples

def weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def save_checkpoint(state, filename, save_backup=True):
    """Save checkpoint with backup"""
    try:
        torch.save(state, filename)
        if save_backup:
            backup_file = filename.replace('.pth', '_backup.pth')
            torch.save(state, backup_file)
        print(f"Checkpoint saved to {filename}")
    except Exception as e:
        print(f"Error saving checkpoint: {str(e)}")

def to_gray_mask(logits):
    """Convert logits to grayscale mask"""
    return logits.argmax(dim=1, keepdim=True).float()

def train_cyclegan(config_path='configs/default_config.yaml'):
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Hyperparameters
    batch_size = config['batch_size']
    epochs = config['epochs']
    lr = config['learning_rate']
    lambda_cycle = config['lambda_cycle']
    lambda_seg = config['lambda_seg']
    
    # Device setup
    device = torch.device(config['device'] if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Transformations
    transform = transforms.Compose([
        transforms.Resize((config['image_size'], config['image_size'])),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # Dataset
    dataset = SemiSupervisedDataset(
        img_dir=config['data']['image_dir'],
        mask_dir=config['data']['mask_dir'],
        paired_ratio=config['paired_ratio'],
        transform=transform
    )
    
    # Data splitting
    torch.manual_seed(42)
    train_val_size = int(config['train_split'] * len(dataset))
    test_size = len(dataset) - train_val_size
    train_val_dataset, test_dataset = random_split(
        dataset, [train_val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    train_size = int(config['val_split'] / (config['train_split'] + config['val_split']) * len(train_val_dataset))
    val_size = len(train_val_dataset) - train_size
    train_dataset, val_dataset = random_split(
        train_val_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # DataLoaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        pin_memory=True, num_workers=2
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        pin_memory=True, num_workers=2
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        pin_memory=True, num_workers=2
    )
    
    # Model
    model = SemiSupervisedCycleGAN(
        lambda_cycle=lambda_cycle,
        lambda_seg=lambda_seg
    ).to(device)
    model.apply(weights_init)
    
    # Optimizers
    optimizer_G = optim.Adam(
        list(model.G_A2B.parameters()) + list(model.G_B2A.parameters()),
        lr=lr, betas=(config['optimizer']['generator']['beta1'], 
                      config['optimizer']['generator']['beta2'])
    )
    optimizer_D = optim.Adam(
        list(model.D_A.parameters()) + list(model.D_B.parameters()),
        lr=lr, betas=(config['optimizer']['discriminator']['beta1'],
                      config['optimizer']['discriminator']['beta2'])
    )
    
    # Schedulers
    scheduler_G = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_G, mode='max', factor=config['scheduler_generator']['factor'],
        patience=config['scheduler_generator']['patience'], min_lr=config['scheduler_generator']['min_lr']
    )
    scheduler_D = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_D, mode='min', factor=config['scheduler_discriminator']['factor'],
        patience=config['scheduler_discriminator']['patience'], min_lr=config['scheduler_discriminator']['min_lr']
    )
    
    # Training setup
    scaler = torch.cuda.amp.GradScaler()
    early_stop_counter = 0
    best_mIoU = 0.0
    
    for epoch in range(epochs):
        model.train()
        epoch_train_loss_G = 0.0
        epoch_train_loss_D = 0.0
        
        for batch_idx, batch in enumerate(train_loader):
            try:
                real_A, real_B, is_paired = batch
                real_A = real_A.to(device, non_blocking=True)
                is_paired = is_paired.to(device, non_blocking=True)
                
                if is_paired.any():
                    real_B = real_B.to(device, non_blocking=True)
                else:
                    real_B = None
            except Exception as e:
                print(f"Data loading error: {e}")
                continue
            
            paired_mask = is_paired.any() if is_paired is not None else False
            
            # Generator Update
            optimizer_G.zero_grad(set_to_none=True)
            
            with torch.cuda.amp.autocast():
                fake_B = model.G_A2B(real_A)
                pred_fake_B = model.D_B(to_gray_mask(fake_B).detach())
                loss_G_adv = model.criterion_gan(
                    pred_fake_B, 
                    torch.ones_like(pred_fake_B, device=device) * 0.9
                )
                
                loss_seg = 0.0
                if paired_mask and real_B is not None:
                    target_B = real_B.squeeze(1).long()
                    loss_seg = model.criterion_seg(fake_B, target_B)
                
                cycle_A = model.G_B2A(to_gray_mask(fake_B))
                loss_cycle = model.criterion_cycle(cycle_A, real_A)
                
                loss_G = loss_G_adv + lambda_cycle * loss_cycle
                if paired_mask:
                    loss_G += lambda_seg * loss_seg
            
            scaler.scale(loss_G).backward()
            torch.nn.utils.clip_grad_norm_(model.G_A2B.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(model.G_B2A.parameters(), 1.0)
            scaler.step(optimizer_G)
            scaler.update()
            
            # Discriminator Update
            optimizer_D.zero_grad(set_to_none=True)
            
            with torch.cuda.amp.autocast():
                fake_B_gray = to_gray_mask(fake_B).detach()
                loss_D_real, loss_D_fake = 0.0, 0.0
                
                if paired_mask and real_B is not None:
                    pred_real = model.D_B(real_B.float())
                    loss_D_real = model.criterion_gan(
                        pred_real, 
                        torch.ones_like(pred_real, device=device) * 0.9
                    )
                
                pred_fake = model.D_B(fake_B_gray)
                loss_D_fake = model.criterion_gan(
                    pred_fake, 
                    torch.zeros_like(pred_fake, device=device)
                )
                loss_D_B = 0.5 * (loss_D_real + loss_D_fake)
                
                # Discriminator A
                if paired_mask and real_B is not None:
                    fake_A = model.G_B2A(real_B.float())
                else:
                    fake_A = model.G_B2A(fake_B_gray)
                
                fake_A_gray = to_gray_mask(fake_A).detach()
                pred_fake_A = model.D_A(fake_A_gray)
                pred_real_A = model.D_A(real_A)
                
                loss_D_A_real = model.criterion_gan(
                    pred_real_A, 
                    torch.ones_like(pred_real_A, device=device) * 0.9
                )
                loss_D_A_fake = model.criterion_gan(
                    pred_fake_A, 
                    torch.zeros_like(pred_fake_A, device=device)
                )
                loss_D_A = 0.5 * (loss_D_A_real + loss_D_A_fake)
                
                loss_D = loss_D_A + loss_D_B
            
            scaler.scale(loss_D).backward()
            torch.nn.utils.clip_grad_norm_(model.D_B.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(model.D_A.parameters(), 1.0)
            scaler.step(optimizer_D)
            scaler.update()
            
            epoch_train_loss_G += loss_G.item()
            epoch_train_loss_D += loss_D.item()
            
            if batch_idx % config['log_freq'] == 0:
                print(f"Epoch {epoch}/{epochs} Batch {batch_idx}/{len(train_loader)} "
                      f"G Loss: {loss_G.item():.4f} D Loss: {loss_D.item():.4f}")
                
                with torch.no_grad():
                    vis_real_A = real_A.detach().cpu()
                    vis_fake_B = fake_B.detach().cpu()
                    vis_real_B = real_B.detach().cpu() if paired_mask else None
                    visualize_samples(vis_real_A, vis_fake_B, vis_real_B, epoch, batch_idx)
        
        # Validation
        model.eval()
        val_mIoU, _ = evaluate(model, val_loader, num_classes=3, device=device)
        
        # Update schedulers
        scheduler_G.step(val_mIoU)
        scheduler_D.step(epoch_train_loss_D / len(train_loader))
        
        # Checkpointing
        if val_mIoU > best_mIoU:
            best_mIoU = val_mIoU
            early_stop_counter =
