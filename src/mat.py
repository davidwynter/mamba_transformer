import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# From arxiv.org/2409.08530v1

# Reversible Instance Normalization (RevIN)
class RevIN(nn.Module):
    def __init__(self, input_dim):
        super(RevIN, self).__init__()
        self.mean = nn.Parameter(torch.zeros(1, input_dim), requires_grad=False)
        self.std = nn.Parameter(torch.ones(1, input_dim), requires_grad=False)

    def forward(self, x, reverse=False):
        if reverse:
            return x * self.std + self.mean
        else:
            self.mean = x.mean(0, keepdim=True)
            self.std = x.std(0, keepdim=True)
            return (x - self.mean) / (self.std + 1e-5)

# Two-Stage Embedding
class EmbeddingLayer(nn.Module):
    def __init__(self, input_dim, n1, n2):
        super(EmbeddingLayer, self).__init__()
        self.embedding = nn.Sequential(
            nn.Linear(input_dim, n1),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(n1, n2),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

    def forward(self, x):
        return self.embedding(x)

# Transformer Module (for short-range dependency)
class TransformerModule(nn.Module):
    def __init__(self, input_dim, num_heads, hidden_dim, dropout=0.1):
        super(TransformerModule, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)

    def forward(self, x):
        attn_output, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_output)
        ff_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x

# Mamba (State-Space Model) Module (for long-range dependency)
class MambaModule(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(MambaModule, self).__init__()
        self.A = nn.Parameter(torch.randn(hidden_dim, hidden_dim))
        self.B = nn.Parameter(torch.randn(hidden_dim, input_dim))
        self.C = nn.Parameter(torch.randn(input_dim, hidden_dim))
        self.D = nn.Parameter(torch.randn(input_dim, input_dim))

    def forward(self, x):
        seq_len, batch_size, input_dim = x.size()
        h = torch.zeros(batch_size, self.A.size(0)).to(x.device)  # Initialize the state vector
        outputs = []

        for t in range(seq_len):
            h = torch.matmul(h, self.A) + torch.matmul(x[t], self.B)
            y = torch.matmul(h, self.C) + torch.matmul(x[t], self.D)
            outputs.append(y)

        outputs = torch.stack(outputs, dim=0)
        return outputs

# Multi-Scale Context Extraction
class MultiScaleContext(nn.Module):
    def __init__(self, input_dim, scales):
        super(MultiScaleContext, self).__init__()
        self.scales = scales
        self.conv_layers = nn.ModuleList([nn.Conv1d(input_dim, input_dim, kernel_size=s, padding=s//2) for s in scales])

    def forward(self, x):
        # x shape: (seq_len, batch_size, input_dim)
        x = x.permute(1, 2, 0)  # (batch_size, input_dim, seq_len)
        outputs = [conv(x) for conv in self.conv_layers]
        outputs = torch.cat(outputs, dim=1)
        return outputs.permute(2, 0, 1)  # (seq_len, batch_size, new_input_dim)

# MAT Model with all components
class MATModel(nn.Module):
    def __init__(self, input_dim, mamba_hidden_dim, transformer_hidden_dim, num_heads, n1, n2, mamba_model):
        super(MATModel, self).__init__()
        self.revin = RevIN(input_dim)  # Reversible Instance Normalization
        self.embedding = EmbeddingLayer(input_dim, n1, n2)  # Two-stage embedding
        # Choose which Mamba variant we use
        if mamba_model == 'Mamba':
            self.mamba = MambaModule(input_dim, mamba_hidden_dim)  # Mamba long-range dependency
        elif mamba_model = 'MambaSelective':
            self.mamba = MambaModuleSelective(input_dim, mamba_hidden_dim)  # Mamba long-range dependency
        self.transformer = TransformerModule(input_dim, num_heads, transformer_hidden_dim)  # Transformer short-range dependency
        self.multi_scale_context = MultiScaleContext(input_dim, scales=[2, 3, 5])  # Multi-scale context extraction
        self.proj1 = nn.Linear(input_dim, n1)
        self.proj2 = nn.Linear(n1, input_dim)

    def forward(self, x):
        # Normalize input
        x = self.revin(x)
        
        # Embedding
        x = self.embedding(x)

        # Multi-scale context extraction
        x = self.multi_scale_context(x)

        # Mamba and Transformer for long and short range dependencies
        mamba_output = self.mamba(x)
        transformer_output = self.transformer(x)

        # Fuse the outputs (add residual connection)
        fused_output = x + mamba_output + transformer_output

        # Project outputs back to input dimensions
        x_proj = self.proj1(fused_output)
        output = self.proj2(x_proj)

        # Reverse normalization
        output = self.revin(output, reverse=True)
        return output

# Custom Loss Function (optional)
class MATLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super(MATLoss, self).__init__()
        self.mse = nn.MSELoss()
        self.mae = nn.L1Loss()
        self.alpha = alpha

    def forward(self, output, target):
        return self.alpha * self.mse(output, target) + (1 - self.alpha) * self.mae(output, target)

# Training function
def train(model, train_loader, criterion, optimizer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")


class MambaModuleSelective(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(MambaModuleSelective, self).__init__()
        self.A = nn.Parameter(torch.randn(hidden_dim, hidden_dim))
        self.B = nn.Parameter(torch.randn(hidden_dim, input_dim))
        self.C = nn.Parameter(torch.randn(input_dim, hidden_dim))
        self.D = nn.Parameter(torch.randn(input_dim, input_dim))
        self.gate = nn.Sigmoid()  # Gate mechanism for selective scanning

    def forward(self, x):
        seq_len, batch_size, input_dim = x.size()
        h = torch.zeros(batch_size, self.A.size(0)).to(x.device)
        outputs = []

        for t in range(seq_len):
            # Selectively filter input through a gate
            g = self.gate(x[t])  # Learn which parts of the input to keep
            h = torch.matmul(h, self.A) + torch.matmul(x[t] * g, self.B)
            y = torch.matmul(h, self.C) + torch.matmul(x[t] * g, self.D)
            outputs.append(y)

        outputs = torch.stack(outputs, dim=0)
        return outputs


# Example dataset setup and training loop
if __name__ == "__main__":
    # Hyperparameters
    input_dim = 10  # Number of input features
    mamba_hidden_dim = 64
    transformer_hidden_dim = 64
    num_heads = 4
    n1, n2 = 128, 64  # Embedding 
    mamba_model='Mamba'

    # Initialize the model
    model = MATModel(input_dim, mamba_hidden_dim, transformer_hidden_dim, num_heads, n1, n2, mamba_model)

    # Define loss and optimizer
    criterion = MATLoss(alpha=0.7)  # Custom loss with MSE + MAE
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # Assume train_loader is defined and provides (input_sequence, target_sequence)
    # num_epochs = 50
    # train(model, train_loader, criterion, optimizer, num_epochs)
