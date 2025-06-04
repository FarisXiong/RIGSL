import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree, remove_self_loops, softmax
import numpy as np
from torch_geometric.nn import GCNConv
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import zeros
from torch.nn import Parameter






class GraphAttentionLayer(MessagePassing):
    def __init__(self, in_channels, out_channels, gamma=-0.95, zeta=1.05, temperature=2.0, bias=False, aggr='add', denoise=False):
        super(GraphAttentionLayer, self).__init__(aggr='add')  # "Add" aggregation.
        #self.lin = torch.nn.Linear(in_channels, out_channels)
        # self.gate = torch.nn.Linear(2*in_channels, 1)
        self.lin = Linear(in_channels, out_channels, bias=False,
                          weight_initializer='glorot')
        self.att = Linear(2 * out_channels, 1, bias=False)
        self.leakyrelu = nn.LeakyReLU(0.2)
        if bias:
            self.bias = Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
        # Denosing
        self.denoise = denoise
        if self.denoise:
            self.gamma = gamma
            self.zeta = zeta
            self.temperature = temperature
            self.transform_before = nn.Linear(in_channels, in_channels)
            self.transform_after = nn.Linear(in_channels, in_channels)
            self.attention_intra = nn.Linear( 2 * in_channels, 1)
            self.attention_inter = nn.Linear( 2 * in_channels, 1)
            self.attention_crossmodal = nn.Linear( 2 * in_channels, 1)
        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.lin.reset_parameters()
        self.att.reset_parameters()
        if self.bias is not None:
            zeros(self.bias)
        

    def sampling(self, x, edge_index, edge_type, train=False):
        before_features = x[edge_index[0], :]
        after_features = x[edge_index[1], :]
        before_features = self.transform_before(before_features)
        after_features = self.transform_before(after_features)
        concatenated_features = torch.cat([before_features, after_features], dim=1)
        mask_intra = edge_type == 0
        mask_inter = edge_type == 1
        mask_crossmodal = edge_type == 2
        attention_weight = torch.zeros_like(mask_intra, dtype=torch.float32, device=concatenated_features.device)
        attention_weight[mask_intra] = self.attention_intra(concatenated_features[mask_intra]).squeeze()
        attention_weight[mask_inter] = self.attention_inter(concatenated_features[mask_inter]).squeeze()
        attention_weight[mask_crossmodal] = self.attention_crossmodal(concatenated_features[mask_crossmodal]).squeeze()

        # attention_weight_intra = self.attention_intra(concatenated_features)
        # attention_weight_inter = self.attention_inter(concatenated_features)
        # attention_weight_crossmodal = self.attention_crossmodal(concatenated_features)
        # attention_weight = attention_weight_intra * (edge_type.unsqueeze(1) == 0) + \
        #                     attention_weight_inter * (edge_type.unsqueeze(1) == 1) + \
        #                     attention_weight_crossmodal * (edge_type.unsqueeze(1) == 2)
        if train:
            debug_var = 1e-7
            bias = 0.0
            # np_random = np.random.uniform(low=debug_var, high=1.0-debug_var, size=np.shape(attention_weight.cpu().detach().numpy()))
            # random_noise = bias + torch.tensor(np_random)
            random_noise = bias + torch.rand_like(attention_weight) * (1.0 - 2 * debug_var) + debug_var
            gate_inputs = torch.log(random_noise) - torch.log(1.0 - random_noise)
            gate_inputs = (gate_inputs.cuda() + attention_weight) / self.temperature
            gate_inputs = torch.sigmoid(gate_inputs)
        else:
            gate_inputs = torch.sigmoid(attention_weight)

        stretched_values = gate_inputs * (self.zeta-self.gamma) + self.gamma
        cliped = torch.clamp(stretched_values, 0.0, 1.0)
        cliped = cliped.float()

        ## desnosing
        # denosing_adj = torch.sparse.FloatTensor(edge_index, cliped)
        return cliped, attention_weight




    def forward(self, x, edge_index, edge_type, train=False):
        x = self.lin(x)
        if self.denoise:
            denoise_weight, penalty_weight = self.sampling(x, edge_index, edge_type, train=train)
            penalty_loss = self.penalty_weight_loss(penalty_weight, self.temperature, self.gamma, self.zeta)
            out = self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x, denoise_weight=denoise_weight)
        else:
            denoise_weight, penalty_weight, penalty_loss = None, None, None
            out = self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x, denoise_weight=denoise_weight)
    
        if self.bias is not None:
            out = out + self.bias
        return out, penalty_loss

    def message(self, x_i, x_j, edge_index, size, denoise_weight=None):      
        alpha = self.leakyrelu(self.att(torch.cat([x_i, x_j], dim=-1)))
        alpha = softmax(alpha, edge_index[0], num_nodes=size[0]) 
        
        if self.denoise:
            return alpha.view(-1, 1) * (x_j) * denoise_weight.unsqueeze(1)
        else:
            return alpha.view(-1, 1) * (x_j)

    def update(self, aggr_out):
        return aggr_out
    

    def penalty_weight_loss(self, penalty_weight, temperature, gamma, zeta):
        penalty_weight_loss = self.l0_norm(penalty_weight, temperature, gamma=gamma, zeta=zeta)
        return penalty_weight_loss

    def l0_norm(self, log_alpha, beta, gamma, zeta):
        gamma = torch.tensor(gamma).to(log_alpha.device)
        zeta = torch.tensor(zeta).to(log_alpha.device)
        reg_per_weight = torch.sigmoid(log_alpha - beta * torch.log(-gamma/zeta))
        return torch.mean(reg_per_weight)
