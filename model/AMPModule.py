from turtle import window_width
import torch
import torch.nn as nn
import torch.nn.functional as F

class LocalCrossAttentionModule(nn.Module):
    def __init__(self, in_dim, out_dim, expansion_factor, heads=1 , dim_feedforward=64, dropout=0.1):
        super(LocalCrossAttentionModule, self).__init__()
        self.expansion_factor = expansion_factor
        self.heads = heads
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dim_feedforward = dim_feedforward
        self.scale = (in_dim // heads + 1e-6) ** -0.5

        self.query_linear = nn.Linear(75, dim_feedforward)
        self.key_linear = nn.Linear(3200, dim_feedforward)
        self.value_linear = nn.Linear(3200, dim_feedforward)

        self.out_conv = nn.Conv2d(dim_feedforward, 2*out_dim, kernel_size=1)
        self.norm1 = nn.LayerNorm(in_dim)
        self.norm2 = nn.LayerNorm(2*out_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def extract_local_features(self, x, center, expansion_factor):
        """
        Extract local features around the center, expanding the range by N times.
        Automatically pads the edges to handle border cases.
        """
        _, _, height, width = x.size()
        pad_size = expansion_factor * 1  # Padding size, adjust as needed

        # Pad all boundaries
        x_padded = F.pad(x, (pad_size, pad_size, pad_size, pad_size), mode='constant', value=0)

        # Update height and width
        height_padded, width_padded = x_padded.shape[2], x_padded.shape[3]

        # Adjust the center point to account for padding
        center_padded = (center[0] + pad_size, center[1] + pad_size)

        # Calculate the start and end indices for the extraction region
        start_x = max(center_padded[0] - pad_size, 0)
        end_x = min(center_padded[0] + pad_size + 1, width_padded)  # +1 because the range is inclusive
        start_y = max(center_padded[1] - pad_size, 0)
        end_y = min(center_padded[1] + pad_size + 1, height_padded)

        return x_padded[:, :, start_y:end_y, start_x:end_x]

    def forward(self, x_query, x_kv):
        batch_size, channels, height, width = x_kv.size()

        # Initialize output tensor
        output = torch.zeros_like(torch.zeros(batch_size, 64, height, width))

        stride = 9 * self.expansion_factor * 2

        for i in range(0, height, stride):
            for j in range(0, width, stride):
                # Extract local features for query and key-value
                local_query = self.extract_local_features(x_query, (j, i), self.expansion_factor)
                local_kv = self.extract_local_features(x_kv, (j, i), self.expansion_factor)

                local_query_flat = local_query.reshape(batch_size, -1)
                local_kv_flat = local_kv.reshape(batch_size, -1)

                q = self.query_linear(local_query_flat).view(batch_size, -1, self.heads, self.dim_feedforward)
                k = self.key_linear(local_kv_flat).view(batch_size, -1, self.heads, self.dim_feedforward)
                v = self.value_linear(local_kv_flat).view(batch_size, -1, self.heads, self.dim_feedforward)

                # Reshape for attention computation
                q = q.permute(0, 2, 1, 3)
                k = k.permute(0, 2, 1, 3)
                v = v.permute(0, 2, 1, 3)

                # Compute attention scores
                attention_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
                attention_scores = F.softmax(attention_scores, dim=-1)
                attention_scores = self.dropout1(attention_scores)

                # Apply attention to the values
                out = torch.matmul(attention_scores, v)
                out = out.permute(0, 2, 1, 3).contiguous().view(batch_size, -1, local_query.size(2), local_query.size(3))

                # Update output tensor
                output[:, :, i:i+local_query.size(2), j:j+local_query.size(3)] = out

        # Final convolution and dropout
        output = self.out_conv(output)
        output = self.dropout2(output)

        return output

# Example usage
# in_dim = 256  # Example channel dimension of the feature map
# expansion_factor = 2  # Example expansion factor
# cross_attention = LocalCrossAttentionModule(in_dim, expansion_factor)

# Assume x_query and x_kv are your feature maps from the CNN
# x_query = ...
# x_kv = ...
# out = cross_attention(x_query, x_kv)
