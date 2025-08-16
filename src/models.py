import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils

def get_model(model_type, model_params, input_dim, output_dim=5):
    """
    Dynamically instantiate a model based on type and parameters.

    Args:
        model_type (str): "mlp", "lstm", ...
        model_params (dict): hyperparameters for the model
        input_dim (int): input feature dimension
        output_dim (int): number of classes (default: 4)

    Returns:
        torch.nn.Module: model instance
    """
    if model_type == "mlp":
        return MLP(input_dim=input_dim, output_dim=output_dim, **model_params)
    
    elif model_type == "vgg16":
        return VGG16(num_classes=output_dim, **model_params)
    
    elif model_type == "lstm":
        return LSTM(input_dim=input_dim, output_dim=output_dim, **model_params)
    
    elif model_type == "tcnnlstm":
        return TcnnLSTM(input_dim=input_dim, output_dim=output_dim, **model_params)
    
    elif model_type == "fused_tcnnlstm":
        embed_model = TcnnLSTM_Embeddings(
            input_dim=model_params["embed_input_dim"],
            hidden_dim=model_params["embed_hidden_dim"],
            num_layers=model_params["num_layers"],
            output_dim=model_params["output_dim"],
            bidirectional=model_params["bidirectional"],
            target_length=model_params["target_length"]
        )

        mfb_model = TcnnLSTM(
            input_dim=model_params["mfb_input_dim"],
            hidden_dim=model_params["mfb_hidden_dim"],
            num_layers=model_params["num_layers"],
            output_dim=model_params["output_dim"],
            bidirectional=model_params["bidirectional"]
        )

        return FusedTcnnLSTM(
            embed_model,
            mfb_model,
            fused_dim=model_params["fused_hidden_dim"],
            output_dim=model_params["output_dim"]
        )

    elif model_type == "alexnet":
        return AlexNetBN(num_classes=output_dim, **model_params)
    
    elif model_type == "grunet":
        return GRUNet(input_dim=input_dim, output_dim=output_dim, **model_params)
  
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

# vgg16 reg fused
class VGG16_Reg_Fused(nn.Module):
    def __init__(self, input_channels=2, num_classes=4):  # Changed to 2 channels
        super(VGG16_Reg_Fused, self).__init__()
        
        # Add temporal alignment layers for MFCC
        self.mfcc_projector = nn.Conv1d(40, 768, kernel_size=1)  # Project MFCC 40->768
        
        self.temporal_align = nn.Sequential(
            nn.Conv1d(768, 768, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(768, 768, kernel_size=3, stride=2, padding=1),  # 1501 -> ~750
            nn.AdaptiveAvgPool1d(749)  # Force exact match to embedding length
        )
        
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            # Collapse frequency axis, downsample time to 300
            nn.AdaptiveAvgPool2d((1, 300))  # → (B, 512, 1, 300)
        )

        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )
        
        self.regressor = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 1), )


        self._init_regression_layer()
    
    def _init_regression_layer(self):
        """Initialize the final regression layer to output reasonable values"""
        # Initialize final linear layer to output around the middle of target range (1.5)
        final_layer = self.regressor[-1]
        nn.init.normal_(final_layer.weight, mean=0.0, std=0.01)
        nn.init.constant_(final_layer.bias, 1.5)  # Bias = middle of [0,3] range
    
    def forward(self, x, mfcc=None):
        # If we get separate embedding and mfcc inputs
        if mfcc is not None:
            # x: embedding (batch, 749, 768)
            # mfcc: (batch, 1501, 40)
            
            # Project and align MFCC
            mfcc_proj = self.mfcc_projector(mfcc.transpose(1, 2))  # (batch, 40 -> 768, 1501)
            mfcc_aligned = self.temporal_align(mfcc_proj).transpose(1, 2)  # (batch, 1501 -> 749, 768)
            
            # Stack as 2 channels: (batch, 2, 768, 749)
            x = torch.stack([x, mfcc_aligned], dim=1).transpose(2, 3)
        
        # Process through VGG16
        x = self.features(x)  # (batch, 512, 1, 300)
        x = x.squeeze(2).permute(0, 2, 1)  # (batch, 300, 512)
        
        # Regression output
        pred = self.regressor(x).squeeze(-1)  # (batch, 300)
        return pred


# # cls with BN
# class VGG16(nn.Module):
#     def __init__(self, input_channels=1, num_classes=4):
#         super(VGG16, self).__init__()
#         self.features = nn.Sequential(
#             # Block 1
#             nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(64, 64, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2),

#             # Block 2
#             nn.Conv2d(64, 128, kernel_size=3, padding=1),
#             nn.BatchNorm2d(128),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(128, 128, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2),

#             # Block 3
#             nn.Conv2d(128, 256, kernel_size=3, padding=1),
#             nn.BatchNorm2d(256),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(256, 256, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(256, 256, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2),

#             # Block 4
#             nn.Conv2d(256, 512, kernel_size=3, padding=1),
#             nn.BatchNorm2d(512),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(512, 512, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(512, 512, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),

#             # Collapse frequency axis, downsample time to 300
#             nn.AdaptiveAvgPool2d((1, 300))  # → (B, 512, 1, 300)
#         )

#         self.classifier = nn.Sequential(
#             nn.Linear(512, 256),
#             nn.ReLU(inplace=True),
#             nn.Dropout(0.3),
#             nn.Linear(256, num_classes),
#         )

#         self.regressor = nn.Sequential(
#             nn.Linear(512, 256),
#             nn.ReLU(inplace=True),
#             nn.Dropout(0.3),
#             nn.Linear(256, 1),
#         )

#         self._init_regression_layer()

#     def _init_regression_layer(self):
#         final_layer = self.regressor[-1]
#         nn.init.normal_(final_layer.weight, mean=0.0, std=0.01)
#         nn.init.constant_(final_layer.bias, 1.5)

#     def forward(self, x):
#         x = self.features(x)                      # (B, 512, 1, 300)
#         x = x.squeeze(2).permute(0, 2, 1)         # (B, 300, 512)
#         logits = self.classifier(x)               # (B, 300, 4)
#         return logits


# cls
class VGG16(nn.Module):
    def __init__(self, input_channels=1, num_classes=5):
        super(VGG16, self).__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            # Collapse frequency axis, downsample time to 300
            nn.AdaptiveAvgPool2d((1, 300))  # → (B, 512, 1, 300)
        )

        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )
        
        self.regressor = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 1), )


        self._init_regression_layer()
    
    def _init_regression_layer(self):
        """Initialize the final regression layer to output reasonable values"""
        # Initialize final linear layer to output around the middle of target range (1.5)
        final_layer = self.regressor[-1]
        nn.init.normal_(final_layer.weight, mean=0.0, std=0.01)
        nn.init.constant_(final_layer.bias, 1.5)  # Bias = middle of [0,3] range
    
    def forward(self, x):
        x = self.features(x)                      
        x = x.squeeze(2).permute(0, 2, 1)         
        logits = self.classifier(x)             
        return logits # (B, 300, num_classes)
        
        # # regression (cont vals pred)
        # pred = self.regressor(x).squeeze(-1)  # (B, T)
        # return pred

    
class AttentionLayer(nn.Module):
    def __init__(self, input_dim):
        super(AttentionLayer, self).__init__()
        self.attention_weights = nn.Parameter(torch.randn(input_dim))

    def forward(self, x):
        u = torch.tanh(x)
        scores = torch.matmul(u, self.attention_weights)  # (B, T)
        attention_weights = torch.softmax(scores, dim=1)  # (B, T)
        attended = x * attention_weights.unsqueeze(-1)    # (B, T, F)
        return attended


class GRUNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, attention_dim=None, output_dim=5, bidirectional=True, num_layers=1, use_attention=False, target_length=300):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.attention_dim = attention_dim
        self.bidirectional = bidirectional
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.use_attention = use_attention
        self.target_length = target_length

        # Add temporal alignment for embeddings
        if input_dim == 768:  # Embedding dimension
            self.temporal_align = nn.AdaptiveAvgPool1d(target_length)
        else:
            self.temporal_align = None

        self.gru = nn.GRU(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional
        )

        self.dropout = nn.Dropout(0.3)
        gru_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.bn_gru = nn.BatchNorm1d(gru_output_dim)

        if attention_dim is not None and use_attention:
            self.proj = nn.Linear(gru_output_dim, attention_dim)
            self.bn_proj = nn.BatchNorm1d(attention_dim)
            self.classifier = nn.Linear(attention_dim, output_dim)
        else:
            self.proj = None
            self.bn_proj = None
            self.classifier = nn.Linear(gru_output_dim, output_dim)

    def forward(self, x, return_repr=True):
        # Temporal alignment for embeddings
        if self.temporal_align is not None:
            # x: (B, T, F) -> (B, F, T) for pooling
            x = x.transpose(1, 2)
            x = self.temporal_align(x)  # (B, F, target_length)
            x = x.transpose(1, 2)  # (B, target_length, F)
        
        out, _ = self.gru(x)  # out: (B, T, H)
        B, T, H = out.shape

        out = self.bn_gru(out.contiguous().view(B * T, H)).view(B, T, H)
        out = self.dropout(out)

        if self.proj is not None:
            feat = self.proj(out)  # (B, T, attention_dim)
            feat = self.bn_proj(feat.view(B * T, -1)).view(B, T, -1)
        else:
            feat = out  # (B, T, H)

        if return_repr:
            return feat  # for fusion
        else:
            return self.classifier(feat)  # for prediction

    

class MLP(nn.Module):
    # Processes each timestep independently through fully connected layers.
    # Input: (batch, features) → Output: (batch, 4)
    def __init__(self, input_dim, hidden_dim=64, output_dim=4, num_layers=2, dropout=0.0, activation=nn.ReLU):
        super(MLP, self).__init__()
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(activation())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(activation())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class SimpleLSTM(nn.Module):
    def __init__(self, 
                 input_dim, 
                 hidden_dim=128, 
                 attention_dim=64, 
                 output_dim=4, 
                 num_layers=2, 
                 dropout=0.5, 
                 bidirectional=True,
                 activation=None):
        super(SimpleLSTM, self).__init__()

        self.num_directions = 2 if bidirectional else 1
        self.activation = activation() if activation is not None else None

        # LSTM block
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0, 
            batch_first=True,
            bidirectional=bidirectional
        )

        self.bn_lstm = nn.BatchNorm1d(hidden_dim * self.num_directions)
        self.attention = AttentionLayer(hidden_dim * self.num_directions)

        # FC layers
        self.fc1 = nn.Linear(hidden_dim * self.num_directions, attention_dim)
        self.bn1 = nn.BatchNorm1d(attention_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(attention_dim, output_dim)

    def forward(self, x, lengths):
        # Pack and run LSTM
        packed_x = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        packed_out, _ = self.lstm(packed_x)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)

        B, T, F = out.shape

        # Batch norm over LSTM output
        out = self.bn_lstm(out.contiguous().view(B * T, F)).view(B, T, F)

        # # Attention
        # out = self.attention(out)

        # FC1 + BN + Dropout
        out = self.fc1(out)
        out = self.bn1(out.contiguous().view(-1, out.shape[-1])).view(B, T, -1)
        out = self.dropout(out)

        # Final output
        logits = self.fc2(out)

        if self.activation is not None:
            logits = self.activation(logits)

        return logits


class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, attention_dim, output_dim,
                 num_layers=2, use_attention=True, bidirectional=True, target_length=300):
        super(LSTM, self).__init__()
        self.use_attention = use_attention
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.target_length = target_length

        # Add temporal alignment for embeddings
        if input_dim == 768:  # Embedding dimension
            self.temporal_align = nn.AdaptiveAvgPool1d(target_length)
        else:
            self.temporal_align = None

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional
        )
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.lstm_output_dim = lstm_output_dim
        self.bn_lstm = nn.BatchNorm1d(lstm_output_dim)

        if self.use_attention:
            self.attention = AttentionLayer(self.lstm_output_dim)

        self.fc1 = nn.Linear(self.lstm_output_dim, attention_dim)
        self.bn1 = nn.BatchNorm1d(attention_dim)
        self.fc2 = nn.Linear(attention_dim, output_dim)

    def forward(self, x, lengths=None):
        x = x.squeeze(1)
        
        # Temporal alignment for embeddings
        if self.temporal_align is not None:
            # x: (B, T, F) -> (B, F, T) for pooling
            x = x.transpose(1, 2)
            x = self.temporal_align(x)  # (B, F, target_length)
            x = x.transpose(1, 2)  # (B, target_length, F)
            if lengths is not None:
                lengths = torch.full((x.size(0),), self.target_length, dtype=torch.long, device=x.device)

        lstm_out, _ = self.lstm(x)
        B, T, F = lstm_out.shape

        out = self.bn_lstm(lstm_out.contiguous().view(B * T, F)).view(B, T, F)

        if self.use_attention:
            out = self.attention(out)

        out = self.fc1(out)
        out = self.bn1(out.view(B * T, -1)).view(B, T, -1)
        out = self.relu(out)
        out = self.dropout(out)

        logits = self.fc2(out)
        return logits


class TcnnLSTM(nn.Module):
    def __init__(self, 
                 input_dim,  # input_dim = output_dim_conv
                 hidden_dim=128,
                 num_layers=2,
                 output_dim=5,
                 embedding_dim=128,
                 bidirectional=True,
                 kernel_size_conv=3,
                 padding_conv=1,
                 stride_conv=1,
                 bias_conv=True,
                 dilation_conv=1,
                 groups_conv=40,
                 target_length=300,
                 return_padded = False):
        super(TcnnLSTM, self).__init__()

        self.bidirectional = bidirectional
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.target_length = target_length
        
        # Input normalization
        self.input_norm = nn.InstanceNorm1d(num_features=input_dim, momentum=0.01, affine=True)
        
        # Determine CNN groups based on input type
        if input_dim == 768:  # Embeddings
            self.groups_conv = 1
            self.temporal_align = nn.AdaptiveAvgPool1d(target_length)
        else:  # MFB or other features
            self.groups_conv = groups_conv
            self.temporal_align = None
        
        # Temporal CNN
        self.cnn = nn.Conv1d(input_dim, input_dim, 
                            kernel_size=kernel_size_conv, 
                            stride=stride_conv,
                            padding=padding_conv, 
                            dilation=dilation_conv, 
                            groups=self.groups_conv,
                            bias=bias_conv)
        
        self.lstm_input_dim = input_dim * 2
        self.lstm = nn.LSTM(self.lstm_input_dim, hidden_dim, num_layers, 
                           bidirectional=bidirectional, batch_first=True)
        
        # Simple classifier (same as TcnnLSTM_Embeddings)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),  # Increase dropout
            nn.Linear(lstm_output_dim, 128),  # Much smaller hidden layer
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(128, output_dim))  # Direct to num classes
        
        self.fc1 = nn.Linear(lstm_output_dim, embedding_dim)
    
    def forward(self, x, lengths=None, hidden=None, memory=None, return_padded=False):
        batch_size, seq_len, feat_dim = x.shape
        
        if lengths is None:
            lengths = torch.full((batch_size,), seq_len, dtype=torch.long, device=x.device)
        
        # Process at native resolution first
        x = x.permute(0, 2, 1)  # (B, F, T)
        x = self.input_norm(x)
        x_cnn = self.cnn(x)
        x_comb = torch.cat([x, x_cnn], dim=1)  # (B, 2F, T)
        
        # Temporal alignment for embeddings
        if self.temporal_align is not None:
            x_aligned = self.temporal_align(x_comb)  # (B, 2F, target_length)
            new_lengths = torch.full((batch_size,), self.target_length, 
                                   dtype=torch.long, device=x.device)
        else:
            x_aligned = x_comb
            new_lengths = lengths
        
        x_aligned = x_aligned.permute(0, 2, 1)  # (B, T, 2F)
        
        # LSTM + classification
        packed = rnn_utils.pack_padded_sequence(x_aligned, new_lengths.cpu(), 
                                               batch_first=True, enforce_sorted=False)
        lstm_out, (hidden_out, memory_out) = self.lstm(packed, 
                                                      (hidden, memory) if hidden is not None else None)
        padded, _ = rnn_utils.pad_packed_sequence(lstm_out, batch_first=True)
        logits = self.classifier(padded)
        
        # Last hidden for embedding
        last_hidden = padded[:, -1, :] 
        emb1 = self.fc1(last_hidden)
        
        if return_padded:
            return logits, emb1, hidden_out, memory_out, padded
        return logits, emb1, hidden_out, memory_out




    
class FusedTcnnLSTM(nn.Module):
    def __init__(self, embed_model, mfb_model, fused_dim=128, output_dim=5):
        super().__init__()
        self.embed_model = embed_model
        self.mfb_model = mfb_model

        # LSTM output dim from each model (typically 128 each)
        embed_dim = embed_model.hidden_dim
        mfb_dim = mfb_model.hidden_dim
        fusion_input_dim = embed_dim + mfb_dim  # e.g., 256

        # Project fused to 128, then to logits
        self.fusion_fc = nn.Sequential(
            nn.Linear(fusion_input_dim, fused_dim),  # (B, 300, 128)
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(fused_dim, output_dim)         # (B, 300, num_classes)
        )

    def forward(self, embed_input, mfb_input, lengths):
        # Forward each sub-model and extract padded outputs
        _, _, _, _, padded_embed = self.embed_model(embed_input, return_padded=True)
        _, _, _, _, padded_mfb = self.mfb_model(mfb_input, lengths, return_padded=True)

        # Fuse (B, 300, 256)
        fused = torch.cat([padded_embed, padded_mfb], dim=-1)
        # print("fused_padded shape: ", fused.shape)

        # Pass through fusion FC
        logits = self.fusion_fc(fused)  # (B, 300, 4)
        # print("logits shape: ", logits.shape)

        return logits, None, None, None
    

class AlexNetBN(nn.Module):
    # Treats input as 2D image with 1×1 convolutions
    # Not good for sequential data like acoustic features (MFCC) and embeddings
    # Good for image-like data like spectrogram
    # num of neuron should be adapted based on feature size
    def __init__(self, input_channels=1, num_classes=4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=11, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),  # Add batch normalization here
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),  # Add batch normalization here
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),  # Add batch normalization here
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),  # Add batch normalization here
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),  # Add batch normalization here
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        # Resize to fixed temporal resolution (300 time steps)
        self.temporal_pool = nn.AdaptiveAvgPool2d((300, 1))  # output: (B, 128, 300, 1)

        # Per-frame classifier
        self.classifier = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        # x: (B, T=749, F=768)
        # x = x.unsqueeze(1)  # (B, 1, T, F)
        x = self.features(x)  # (B, 128, T', F')
        x = self.temporal_pool(x)  # (B, 128, 300, 1)
        x = x.squeeze(-1).permute(0, 2, 1)  # (B, 300, 128)
        out = self.classifier(x)  # (B, 300, 4)
        return out



# class GRUNet(nn.Module):
#     # Bidirectional GRU captures forward/backward temporal context
#     # Attention mechanism weighs important timesteps
#     def __init__(self, input_dim, hidden_dim, attention_dim, output_dim, num_layers=2, dropout=0.5):
#         super(GRUNet, self).__init__()
#         self.relu = nn.ReLU()
#         self.dropout = nn.Dropout(dropout)

#         self.gru = nn.GRU(input_dim, hidden_dim, num_layers=num_layers,
#                           batch_first=True, bidirectional=True)
#         self.bn_gru = nn.BatchNorm1d(hidden_dim * 2)
#         self.attention_hr = AttentionLayer(hidden_dim * 2)

#         self.fc1_hr = nn.Linear(hidden_dim * 2, attention_dim)
#         self.bn1_hr = nn.BatchNorm1d(attention_dim)

#         self.fc2_hr = nn.Linear(attention_dim, output_dim)

#     def forward(self, x):
#         x = x.squeeze(1)             # (B, T, D) if input is (B, 1, T, D)
#         gru_out, _ = self.gru(x)
#         B, T, F = gru_out.shape
        
#         # batch normalization
#         gru_out = self.bn_gru(gru_out.contiguous().view(B * T, F)).view(B, T, F) # Output: (batch, 300, hidden_dim * 2)
    
#         gru_out = self.fc1_hr(gru_out)  # (B, T, attention_dim)
#         gru_out = self.bn1_hr(gru_out.view(-1, gru_out.shape[-1])).view(B, T, -1) # (B, T, attention_dim)

#         attention_hr_out = self.attention_hr(gru_out)
        
#         hr = self.fc1_hr(attention_hr_out)
#         hr = self.bn1_hr(hr)  # Apply batch normalization
#         hr = self.relu(hr)
#         hr = self.dropout(hr)
#         logits = self.fc2_hr(hr)  # (B, T, output_dim=4)
        
#         return logits


# class SimpleMLP(nn.Module):
#     def __init__(self, input_dim, hidden_dim=64, output_dim=4):
#         super(SimpleMLP, self).__init__()
#         self.net = nn.Sequential(
#             nn.Linear(input_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, output_dim)
#         )

#     def forward(self, x):
#         return self.net(x)


# class SimpleLSTM(nn.Module):
#     def __init__(self, input_dim, hidden_dim=128, output_dim=4, num_layers=1, bidirectional=False):
#         super(SimpleLSTM, self).__init__()
#         self.hidden_dim = hidden_dim
#         self.num_layers = num_layers
#         self.bidirectional = bidirectional
#         self.num_directions = 2 if bidirectional else 1

#         self.lstm = nn.LSTM(
#             input_dim=input_dim,
#             hidden_size=hidden_dim,
#             num_layers=num_layers,
#             batch_first=True,
#             bidirectional=bidirectional
#         )

#         self.classifier = nn.Linear(hidden_dim * self.num_directions, output_dim)

#     def forward(self, x, lengths):
#         # Pack the padded sequence
#         # x is batched padded input sequences with shape (B, T_max, D)
#         # lengths is a tensor of original sequence lengths
#         # packed_x is unpadded, time-reordered with shape (total_valid_T, D)
#         packed_x = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        
#         # packed_out is the output of LSTM with shape (total_valid_T, hidden_dim * num_directions)
#         packed_out, _ = self.lstm(packed_x)
        
#         # out is padded back to the original shape (B, T_max, hidden_dim * num_directions)
#         out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)

#         # logits with shape (B, T_max, output_dim)
#         logits = self.classifier(out)
#         return logits

