import torch
import pytest
from src.mat import RevIN, EmbeddingLayer, TransformerModule, MambaModule, MultiScaleContext, MATModel

# Set a random seed for reproducibility
torch.manual_seed(42)

# Device to run tests on
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Test data parameters
input_dim = 10
seq_len = 50
batch_size = 32
hidden_dim = 64


@pytest.fixture
def sample_input():
    """Fixture to create sample input tensor."""
    return torch.randn(seq_len, batch_size, input_dim).to(device)


def test_revin(sample_input):
    """Test the Reversible Instance Normalization (RevIN)."""
    revin = RevIN(input_dim).to(device)
    normalized = revin.forward(sample_input)
    reversed_input = revin.forward(normalized, reverse=True)

    assert normalized.shape == sample_input.shape, "RevIN output shape should match input shape."
    assert torch.allclose(reversed_input, sample_input, atol=1e-5), "Reversed input should match original input."


def test_embedding_layer(sample_input):
    """Test the two-stage embedding layer."""
    embedding = EmbeddingLayer(input_dim, 128, 64).to(device)
    embedded_output = embedding(sample_input)

    assert embedded_output.shape == (seq_len, batch_size, 64), "Embedding output shape should be (seq_len, batch_size, 64)."


def test_transformer_module(sample_input):
    """Test the Transformer module."""
    transformer = TransformerModule(64, 4, hidden_dim).to(device)
    sample_input_transformed = torch.randn(seq_len, batch_size, 64).to(device)  # Adjust input shape
    transformer_output = transformer(sample_input_transformed)

    assert transformer_output.shape == sample_input_transformed.shape, "Transformer output shape should match input."


def test_mamba_module(sample_input):
    """Test the Mamba module (State-Space Model)."""
    mamba = MambaModule(input_dim, hidden_dim).to(device)
    mamba_output = mamba(sample_input)

    assert mamba_output.shape == sample_input.shape, "Mamba output shape should match input."


def test_multiscale_context(sample_input):
    """Test the Multi-Scale Context extraction."""
    multi_scale_context = MultiScaleContext(input_dim, [2, 3, 5]).to(device)
    multi_scale_output = multi_scale_context(sample_input)

    assert multi_scale_output.shape == (seq_len, batch_size, input_dim * 3), "Multi-Scale output shape should be (seq_len, batch_size, input_dim * 3)."


def test_mat_model_forward(sample_input):
    """Test the full MAT model forward pass."""
    mat_model = MATModel(input_dim, hidden_dim, hidden_dim, 4, 128, 64).to(device)
    mat_output = mat_model(sample_input)

    assert mat_output.shape == sample_input.shape, "MAT model output shape should match input."


def test_custom_loss():
    """Test the custom loss function (MSE + MAE combination)."""
    from src.mat import MATLoss

    output = torch.randn(batch_size, input_dim).to(device)
    target = torch.randn(batch_size, input_dim).to(device)

    loss_fn = MATLoss(alpha=0.7).to(device)
    loss = loss_fn(output, target)

    assert loss.item() >= 0, "Loss should be a non-negative value."
    assert isinstance(loss.item(), float), "Loss should be a float."


def test_model_training_step(sample_input):
    """Test a single training step of the MAT model."""
    mat_model = MATModel(input_dim, hidden_dim, hidden_dim, 4, 128, 64).to(device)
    optimizer = torch.optim.Adam(mat_model.parameters(), lr=0.0001)
    criterion = torch.nn.MSELoss()

    # Forward pass
    output = mat_model(sample_input)
    target = torch.randn_like(output).to(device)

    # Calculate loss
    loss = criterion(output, target)

    # Backward pass and optimization step
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    assert loss.item() > 0, "Loss should be greater than 0 during training step."


if __name__ == "__main__":
    pytest.main()
