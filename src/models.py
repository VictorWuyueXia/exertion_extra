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
    
    def forward(self, x):
        x = self.features(x)                      
        x = x.squeeze(2).permute(0, 2, 1)         
        logits = self.classifier(x)             
        return logits 


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
            total_T = self.target_length
        else:
            x_aligned = x_comb
            new_lengths = lengths
            total_T = seq_len
        
        x_aligned = x_aligned.permute(0, 2, 1)  # (B, T, 2F)
        
        # LSTM + classification
        packed = rnn_utils.pack_padded_sequence(x_aligned, new_lengths.cpu(), 
                                               batch_first=True, enforce_sorted=False)
        lstm_out, (hidden_out, memory_out) = self.lstm(packed, 
                                                      (hidden, memory) if hidden is not None else None)
        padded, _ = rnn_utils.pad_packed_sequence(lstm_out, batch_first=True, total_length=total_T)
        logits = self.classifier(padded)
        
        # Last hidden for embedding
        last_hidden = padded[:, -1, :] 
        emb1 = self.fc1(last_hidden)
        
        # To be safe with DataParallel gather (concat along dim=0), move batch dimension to dim=0
        hidden_out_b = hidden_out.permute(1, 0, 2).contiguous()   # (B, num_layers*dirs, H)
        memory_out_b = memory_out.permute(1, 0, 2).contiguous()   # (B, num_layers*dirs, H)

        if return_padded:
            return logits, emb1, hidden_out_b, memory_out_b, padded
        return logits, emb1, hidden_out_b, memory_out_b




# class LegacyTcnnLSTM(nn.Module):
#     """Legacy TCNN-LSTM adapted to output per-timestep logits, pooled later by the training loop."""

#     def __init__(self, input_dim=40, hidden_dim=128, n_layers=2, embedding_dim=128, target_dim=1, output_dim_conv=40, kernel_size_conv=3, padding_conv=1, stride_conv=1, bias_conv=True, dilation_conv=1, groups_conv=4-, all_event_len=5):
#         """Init function."""
#         super(tcnnLSTM_classifier, self).__init__()
#         self.cnn = nn.Conv1d(input_dim, output_dim_conv, kernel_size_conv, stride=stride_conv, padding=padding_conv, dilation=dilation_conv, groups=groups_conv, bias=bias_conv)
        
#         lstm_input_dim=2*output_dim_conv
#         self.dropout = nn.Dropout(p = 0.1)


#         self.input_norm = nn.InstanceNorm1d(num_features=input_dim, momentum=0.01, affine=True)

        
#         self.hidden_dim = hidden_dim
#         self.lstm = nn.LSTM(lstm_input_dim, hidden_dim, n_layers)
#         # The linear layer that maps from hidden state space to tag space
#         self.classifier = nn.Sequential(
#             nn.Dropout(p=0.25),
#             nn.Linear(128, 1024),
#             nn.ReLU(inplace=True),
#             nn.Dropout(p=0.25),
#             nn.Linear(1024, 1024),
#             nn.ReLU(inplace=True),
#             nn.Linear(1024, all_event_len),
#         )
#         self.fc1 = nn.Linear(hidden_dim, embedding_dim)
#         #self.dropout = nn.Dropout(p=0.1)
#         self.fc2 = nn.Linear(embedding_dim, target_dim)
        
#         self.hidden = self.init_hidden()

#     def init_hidden(self, batch_size=16):
#         """Serve as initialisation for the hidden states."""
        
#         return (autograd.Variable(torch.zeros(self.lstm.num_layers, batch_size, self.hidden_dim)),
#                 autograd.Variable(torch.zeros(self.lstm.num_layers, batch_size, self.hidden_dim)))

#     def forward(self, x, hidden, memory, sorted_lens):
#         """Forward function."""
#         x1 = x.squeeze(1)
        
#         x_cnn1 = self.cnn(x1)
        
#         x_concat = torch.cat((x1, x_cnn1), dim=1)
        
#         x3 = x_concat.permute(2,0,1) # [L, B, C]
        
#         packed_x = pack_padded_sequence(x3, sorted_lens, batch_first=False, enforce_sorted=False)
        
#         lstm_out, (hidden, memory) = self.lstm(packed_x, (hidden, memory)) 
        
#         padded_x, lens_x = pad_packed_sequence(lstm_out)
        
        
#         sorted_lens = sorted_lens.view(-1,1).unsqueeze(2).repeat(1,1,padded_x.shape[2])
 
#         if lstm_out.data.is_cuda:
#             sorted_lens = sorted_lens.cuda(lstm_out.data.get_device())

#         last_out = padded_x.transpose(0,1).gather(1, sorted_lens-1).squeeze(1)
#         emb1 = self.fc1(last_out) 
        
#         logits = self.classifier(last_out)
#         return logits, emb1, hidden, memory