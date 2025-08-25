import torch
import torch.nn as nn
# # Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class TransformerGenerator(nn.Module):
    def __init__(self, latent_dim, output_dim=2048, max_length=512, nhead=8, num_layers=6):
        super().__init__()
        self.max_length = max_length
        self.output_dim = output_dim
        
        # Positional encoding
        self.pos_embedding = nn.Embedding(max_length, latent_dim)
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim, nhead=nhead, dim_feedforward=latent_dim*4,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output projection
        self.output_proj = nn.Linear(latent_dim, output_dim)
        
        # Length prediction
        self.length_predictor = nn.Linear(latent_dim, 1)

    def forward(self, z, target_lengths=None):
        b = z.size(0)
        
        if target_lengths is None:
            # Predict lengths from noise
            target_lengths = (torch.sigmoid(self.length_predictor(z)) * self.max_length).int() + 1
        
        # Expand noise to sequence
        z_expanded = z.unsqueeze(1).repeat(1, self.max_length, 1)  # (b, max_length, latent_dim)
        
        # Add positional encoding
        positions = torch.arange(self.max_length, device=z.device).unsqueeze(0).repeat(b, 1)
        pos_emb = self.pos_embedding(positions)
        transformer_input = z_expanded + pos_emb
        
        # Apply transformer
        transformer_output = self.transformer(transformer_input)
        
        # Project to output dimension
        raw_output = self.output_proj(transformer_output)  # (b, max_length, output_dim)
        
        
            
        
        return raw_output
    
# generator = TransformerGenerator(latent_dim=512).to(device)
# batch_size = 64
# min_len, max_len = 512 , 1024
# latent_dim = 512
# for i in range(100):
#     # In the generator training phase:
#     z = torch.randn(batch_size, latent_dim).to(device)
#     target_lengths = torch.randint(min_len, max_len, (batch_size,)).to(device)
    
#     sequences, mask = generator(z, target_lengths)
#     print(i)