import math
import torch
import torch.nn as nn


class PriceLSTM(nn.Module):
    """PriceLSTM

    A 2-layer LSTM for processing price sequences.

    Args:
        input_size (int): number of input features per time step (e.g., 1 for price)
        hidden_size (int): LSTM hidden dimension (default 128)
        num_layers (int): number of LSTM layers (default 2)

    Forward:
        x: (batch, seq_len, input_size)

    Returns:
        output: (batch, seq_len, hidden_size)
        (hn, cn): final hidden and cell states
    """

    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # batch_first=True makes input/output shapes (batch, seq, feature)
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        output, (hn, cn) = self.lstm(x)
        # output: (batch, seq_len, hidden_size)
        return output, (hn, cn)


class WeatherLSTM(nn.Module):
    """WeatherLSTM

    A 2-layer LSTM for processing weather sequences.

    Args:
        input_size (int): number of weather features per time step (e.g., rainfall, temp...)
        hidden_size (int): LSTM hidden dimension (default 64)
        num_layers (int): number of LSTM layers (default 2)

    Forward:
        x: (batch, seq_len, input_size)

    Returns:
        output: (batch, seq_len, hidden_size)
        (hn, cn): final hidden and cell states
    """

    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)

    def forward(self, x):
        output, (hn, cn) = self.lstm(x)
        return output, (hn, cn)


class CrossAttention(nn.Module):
    """CrossAttention

    Standard scaled dot-product attention where Query comes from the price LSTM
    output and Key/Value come from the weather LSTM output.

    This implementation uses learnable linear projections to map Q, K, V to
    a common dimension and computes single-head attention.

    Inputs:
        Q: (batch, q_len, q_dim)
        K: (batch, k_len, k_dim)
        V: (batch, k_len, v_dim)

    Output:
        attn_output: (batch, q_len, proj_dim)
    """

    def __init__(self, q_dim: int, k_dim: int, v_dim: int, proj_dim: int = 128):
        super().__init__()
        # linear projections to common dimension
        self.q_proj = nn.Linear(q_dim, proj_dim)
        self.k_proj = nn.Linear(k_dim, proj_dim)
        self.v_proj = nn.Linear(v_dim, proj_dim)
        self.scale = math.sqrt(proj_dim)

    def forward(self, Q, K, V, mask=None):
        # Project inputs
        # Q: (batch, q_len, q_dim)
        q = self.q_proj(Q)  # (batch, q_len, proj_dim)
        k = self.k_proj(K)  # (batch, k_len, proj_dim)
        v = self.v_proj(V)  # (batch, k_len, proj_dim)

        # Compute attention scores
        # scores: (batch, q_len, k_len)
        scores = torch.bmm(q, k.transpose(1, 2)) / self.scale

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn_weights = torch.softmax(scores, dim=-1)  # (batch, q_len, k_len)

        # Weighted sum of values
        attn_output = torch.bmm(attn_weights, v)  # (batch, q_len, proj_dim)
        return attn_output


class AgriFlowPipelineA(nn.Module):
    """AgriFlowPipelineA

    Combines a PriceLSTM, WeatherLSTM and CrossAttention module. Runs the
    price and weather sequences through their LSTMs, applies cross-attention
    (Q=price, K=weather, V=weather) and uses a regression head to predict
    future prices for 7, 14 and 30 days ahead.

    Args:
        price_input_size (int): features per price time step (usually 1)
        weather_input_size (int): features per weather time step
    """

    def __init__(self, price_input_size: int = 1, weather_input_size: int = 6):
        super().__init__()
        # LSTMs
        self.price_lstm = PriceLSTM(input_size=price_input_size, hidden_size=128, num_layers=2)
        self.weather_lstm = WeatherLSTM(input_size=weather_input_size, hidden_size=64, num_layers=2)

        # Cross-attention: project to common dim (use price hidden size)
        self.attention = CrossAttention(q_dim=128, k_dim=64, v_dim=64, proj_dim=128)

        # Regression head: take last time step of attention output and predict 3 values
        self.regressor = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
        )

    def forward(self, price_seq, weather_seq, mask=None):
        """Forward pass

        Args:
            price_seq: (batch, p_seq_len, price_input_size)
            weather_seq: (batch, w_seq_len, weather_input_size)
            mask: optional attention mask (batch, p_seq_len, w_seq_len)

        Returns:
            preds: (batch, 3) predictions for [7d, 14d, 30d]
        """
        # Run through LSTMs
        Hp, _ = self.price_lstm(price_seq)  # (batch, p_seq_len, 128)
        Hw, _ = self.weather_lstm(weather_seq)  # (batch, w_seq_len, 64)

        # Cross-attention (Q=Hp, K=Hw, V=Hw)
        attn_out = self.attention(Hp, Hw, Hw, mask=mask)  # (batch, p_seq_len, 128)

        # Pool: use last time step from attention output (assumes recent prices matter most)
        pooled = attn_out[:, -1, :]  # (batch, 128)

        preds = self.regressor(pooled)  # (batch, 3)
        return preds


def count_parameters(model: nn.Module):
    """Print total trainable parameters in the model."""
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total:,}")
    return total


if __name__ == "__main__":
    # Quick test to verify shapes
    batch = 2
    p_seq_len = 90
    w_seq_len = 67
    price_features = 1
    weather_features = 6

    model = AgriFlowPipelineA(price_input_size=price_features, weather_input_size=weather_features)
    count_parameters(model)

    # Fake data
    price_x = torch.randn(batch, p_seq_len, price_features)
    weather_x = torch.randn(batch, w_seq_len, weather_features)

    with torch.no_grad():
        out = model(price_x, weather_x)
    print("Weather encoder: 67 timesteps (60 historical + 7 forecast)")
    print("Output shape (batch, 3):", out.shape)
    # Ensure outputs correspond to 7,14,30 day forecasts per batch
    assert out.shape == (batch, 3), "Output shape mismatch"
    print("Quick shape test passed.")
