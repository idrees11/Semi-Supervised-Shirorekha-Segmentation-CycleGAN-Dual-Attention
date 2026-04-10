
import torch
import torch.nn as nn
from .generator import Generator
from .discriminator import Discriminator

class SemiSupervisedCycleGAN(nn.Module):
    def __init__(self, lambda_cycle=10, lambda_seg=50.0, num_classes=3):
        super().__init__()
        # Generators
        self.G_A2B = Generator(in_channels=1, out_channels=num_classes)  # Image → Segmentation
        self.G_B2A = Generator(in_channels=1, out_channels=1)             # Segmentation → Image
        
        # Discriminators
        self.D_A = Discriminator(in_channels=1)  # Real/fake images
        self.D_B = Discriminator(in_channels=1)  # Real/fake segmentations
        
        # Loss functions
        self.criterion_gan = nn.BCEWithLogitsLoss()
        self.criterion_cycle = nn.L1Loss()
        self.criterion_seg = nn.CrossEntropyLoss()

        # Weighting factors
        self.lambda_cycle = lambda_cycle
        self.lambda_seg = lambda_seg
        self.num_classes = num_classes

        # Initialize weights
        self.apply(self.init_weights)

    @staticmethod
    def init_weights(m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.normal_(m.weight, 0, 0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight, 1.0, 0.02)
            nn.init.constant_(m.bias, 0)

    def forward(self, real_A, real_B=None, is_paired=None, mode='train'):
        if mode == 'inference':
            return self.G_A2B(real_A).argmax(dim=1, keepdim=True)

        # Forward passes
        fake_B_logits = self.G_A2B(real_A)
        fake_B_classes = fake_B_logits.argmax(dim=1, keepdim=True).float()
        recon_A = self.G_B2A(fake_B_classes)

        # Loss calculations
        losses = {
            'gan': self.compute_generator_loss(self.D_B(fake_B_classes)),
            'cycle': self.criterion_cycle(recon_A, real_A)
        }

        # Paired data losses
        if is_paired is not None and is_paired.any():
            losses.update({
                'seg': self.criterion_seg(fake_B_logits, real_B.squeeze(1).long()),
                'cycle_B': self.criterion_cycle(
                    self.G_A2B(self.G_B2A(real_B.float())), 
                    real_B
                ) if real_B is not None else 0
            })

        # Weighted total loss
        total_loss = losses['gan'] + self.lambda_cycle * losses['cycle']
        if is_paired is not None and is_paired.any():
            total_loss += self.lambda_seg * losses['seg']
            total_loss += self.lambda_cycle * losses.get('cycle_B', 0)

        return {
            'total_loss': total_loss,
            **losses,
            'fake_B': fake_B_classes.detach(),
            'recon_A': recon_A.detach()
        }

    def compute_generator_loss(self, pred):
        return self.criterion_gan(pred, torch.ones_like(pred))

    def compute_discriminator_loss(self, pred_real, pred_fake):
        real_loss = self.criterion_gan(pred_real, torch.ones_like(pred_real)*0.9)
        fake_loss = self.criterion_gan(pred_fake, torch.zeros_like(pred_fake))
        return (real_loss + fake_loss) * 0.5
