import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.nn import  GraphConv
import numpy as np, itertools, random, copy, math
from model_GCN import GCN_2Layers, GCNLayer1, GCNII, TextCNN
# from model_hyper import HyperGCN
from torch_geometric.nn import MessagePassing
from itertools import permutations
from torch_geometric.utils import add_self_loops, degree
from high_fre_conv import GraphConvLayer
from nodeformer import NodeFormer
# from m3net import M3Net
from model_hyper import HyperGCN
from mmgcn import MM_GCN
from gat import GraphAttentionLayer
def print_grad(grad):
    print('the grad is', grad[2][0:5])
    return grad

class FocalLoss(nn.Module):
    def __init__(self, gamma = 2.5, alpha = 1, size_average = True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.size_average = size_average
        self.elipson = 0.000001
    
    def forward(self, logits, labels):
        """
        cal culates loss
        logits: batch_size * labels_length * seq_length
        labels: batch_size * seq_length
        """
        if labels.dim() > 2:
            labels = labels.contiguous().view(labels.size(0), labels.size(1), -1)
            labels = labels.transpose(1, 2)
            labels = labels.contiguous().view(-1, labels.size(2)).squeeze()
        if logits.dim() > 3:
            logits = logits.contiguous().view(logits.size(0), logits.size(1), logits.size(2), -1)
            logits = logits.transpose(2, 3)
            logits = logits.contiguous().view(-1, logits.size(1), logits.size(3)).squeeze()
        labels_length = logits.size(1)
        seq_length = logits.size(0)

        new_label = labels.unsqueeze(1)
        label_onehot = torch.zeros([seq_length, labels_length]).cuda().scatter_(1, new_label, 1)

        log_p = F.log_softmax(logits,-1)
        pt = label_onehot * log_p
        sub_pt = 1 - pt
        fl = -self.alpha * (sub_pt)**self.gamma * log_p
        if self.size_average:
            return fl.mean()
        else:
            return fl.sum()

class MMDFN_FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(MMDFN_FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = input

        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        return loss.mean() if self.size_average else loss.sum()

class PCGNet_FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(PCGNet_FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = input

        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        return loss.mean() if self.size_average else loss.sum()

class MaskedNLLLoss(nn.Module):

    def __init__(self, weight=None):
        super(MaskedNLLLoss, self).__init__()
        self.weight = weight
        self.loss = nn.NLLLoss(weight=weight,
                               reduction='sum')

    def forward(self, pred, target, mask):
        """
        pred -> batch*seq_len, n_classes
        target -> batch*seq_len
        mask -> batch, seq_len
        """
        mask_ = mask.view(-1,1)
        if type(self.weight)==type(None):
            loss = self.loss(pred*mask_, target)/torch.sum(mask)
        else:
            loss = self.loss(pred*mask_, target)\
                            /torch.sum(self.weight[target]*mask_.squeeze())
        return loss


class MaskedMSELoss(nn.Module):

    def __init__(self):
        super(MaskedMSELoss, self).__init__()
        self.loss = nn.MSELoss(reduction='sum')

    def forward(self, pred, target, mask):
        """
        pred -> batch*seq_len
        target -> batch*seq_len
        mask -> batch*seq_len
        """
        loss = self.loss(pred*mask, target)/torch.sum(mask)
        return loss


class UnMaskedWeightedNLLLoss(nn.Module):

    def __init__(self, weight=None):
        super(UnMaskedWeightedNLLLoss, self).__init__()
        self.weight = weight
        self.loss = nn.NLLLoss(weight=weight,
                               reduction='sum')

    def forward(self, pred, target):
        """
        pred -> batch*seq_len, n_classes
        target -> batch*seq_len
        """
        if type(self.weight)==type(None):
            loss = self.loss(pred, target)
        else:
            loss = self.loss(pred, target)\
                            /torch.sum(self.weight[target])
        return loss


class SimpleAttention(nn.Module):

    def __init__(self, input_dim):
        super(SimpleAttention, self).__init__()
        self.input_dim = input_dim
        self.scalar = nn.Linear(self.input_dim,1,bias=False)

    def forward(self, M, x=None):
        """
        M -> (seq_len, batch, vector)
        x -> dummy argument for the compatibility with MatchingAttention
        """
        scale = self.scalar(M)
        alpha = F.softmax(scale, dim=0).permute(1,2,0)
        attn_pool = torch.bmm(alpha, M.transpose(0,1))[:,0,:]
        return attn_pool, alpha


class MatchingAttention(nn.Module):

    def __init__(self, mem_dim, cand_dim, alpha_dim=None, att_type='general'):
        super(MatchingAttention, self).__init__()
        assert att_type!='concat' or alpha_dim!=None
        assert att_type!='dot' or mem_dim==cand_dim
        self.mem_dim = mem_dim
        self.cand_dim = cand_dim
        self.att_type = att_type
        if att_type=='general':
            self.transform = nn.Linear(cand_dim, mem_dim, bias=False)
        if att_type=='general2':
            self.transform = nn.Linear(cand_dim, mem_dim, bias=True)
        elif att_type=='concat':
            self.transform = nn.Linear(cand_dim+mem_dim, alpha_dim, bias=False)
            self.vector_prod = nn.Linear(alpha_dim, 1, bias=False)

    def forward(self, M, x, mask=None):
        """
        M -> (seq_len, batch, mem_dim)
        x -> (batch, cand_dim) cand_dim == mem_dim?
        mask -> (batch, seq_len)
        """
        if type(mask)==type(None):
            mask = torch.ones(M.size(1), M.size(0)).type(M.type())

        if self.att_type=='dot':
            M_ = M.permute(1,2,0)
            x_ = x.unsqueeze(1)
            alpha = F.softmax(torch.bmm(x_, M_), dim=2)
        elif self.att_type=='general':
            M_ = M.permute(1,2,0)
            x_ = self.transform(x).unsqueeze(1)
            alpha = F.softmax(torch.bmm(x_, M_), dim=2)
        elif self.att_type=='general2':
            M_ = M.permute(1,2,0)
            x_ = self.transform(x).unsqueeze(1)
            mask_ = mask.unsqueeze(2).repeat(1, 1, self.mem_dim).transpose(1, 2)
            M_ = M_ * mask_
            alpha_ = torch.bmm(x_, M_)*mask.unsqueeze(1)
            alpha_ = torch.tanh(alpha_)
            alpha_ = F.softmax(alpha_, dim=2)
            alpha_masked = alpha_*mask.unsqueeze(1)
            alpha_sum = torch.sum(alpha_masked, dim=2, keepdim=True)
            alpha = alpha_masked/alpha_sum
        else:
            M_ = M.transpose(0,1)
            x_ = x.unsqueeze(1).expand(-1,M.size()[0],-1)
            M_x_ = torch.cat([M_,x_],2)
            mx_a = F.tanh(self.transform(M_x_))
            alpha = F.softmax(self.vector_prod(mx_a),1).transpose(1,2)

        attn_pool = torch.bmm(alpha, M.transpose(0,1))[:,0,:]
        return attn_pool, alpha


class Attention(nn.Module):
    def __init__(self, embed_dim, hidden_dim=None, out_dim=None, n_head=1, score_function='dot_product', dropout=0):
        ''' Attention Mechanism
        :param embed_dim:
        :param hidden_dim:
        :param out_dim:
        :param n_head: num of head (Multi-Head Attention)
        :param score_function: scaled_dot_product / mlp (concat) / bi_linear (general dot)
        :return (?, q_len, out_dim,)
        '''
        super(Attention, self).__init__()
        if hidden_dim is None:
            hidden_dim = embed_dim // n_head
        if out_dim is None:
            out_dim = embed_dim
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.n_head = n_head
        self.score_function = score_function
        self.w_k = nn.Linear(embed_dim, n_head * hidden_dim)
        self.w_q = nn.Linear(embed_dim, n_head * hidden_dim)
        self.proj = nn.Linear(n_head * hidden_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        if score_function == 'mlp':
            self.weight = nn.Parameter(torch.Tensor(hidden_dim*2))
        elif self.score_function == 'bi_linear':
            self.weight = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        else:
            self.register_parameter('weight', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.hidden_dim)
        if self.weight is not None:
            self.weight.data.uniform_(-stdv, stdv)

    def forward(self, k, q):
        if len(q.shape) == 2:
            q = torch.unsqueeze(q, dim=1)
        if len(k.shape) == 2:
            k = torch.unsqueeze(k, dim=1)
        mb_size = k.shape[0]
        k_len = k.shape[1]
        q_len = q.shape[1]
        # k: (?, k_len, embed_dim,)
        # q: (?, q_len, embed_dim,)
        # kx: (n_head*?, k_len, hidden_dim)
        # qx: (n_head*?, q_len, hidden_dim)
        # score: (n_head*?, q_len, k_len,)
        # output: (?, q_len, out_dim,)
        kx = self.w_k(k).view(mb_size, k_len, self.n_head, self.hidden_dim)
        kx = kx.permute(2, 0, 1, 3).contiguous().view(-1, k_len, self.hidden_dim)
        qx = self.w_q(q).view(mb_size, q_len, self.n_head, self.hidden_dim)
        qx = qx.permute(2, 0, 1, 3).contiguous().view(-1, q_len, self.hidden_dim)
        if self.score_function == 'dot_product':
            kt = kx.permute(0, 2, 1)
            score = torch.bmm(qx, kt)
        elif self.score_function == 'scaled_dot_product':
            kt = kx.permute(0, 2, 1)
            qkt = torch.bmm(qx, kt)
            score = torch.div(qkt, math.sqrt(self.hidden_dim))
        elif self.score_function == 'mlp':
            kxx = torch.unsqueeze(kx, dim=1).expand(-1, q_len, -1, -1)
            qxx = torch.unsqueeze(qx, dim=2).expand(-1, -1, k_len, -1)
            kq = torch.cat((kxx, qxx), dim=-1)
            score = torch.tanh(torch.matmul(kq, self.weight))
        elif self.score_function == 'bi_linear':
            qw = torch.matmul(qx, self.weight)
            kt = kx.permute(0, 2, 1)
            score = torch.bmm(qw, kt)
        else:
            raise RuntimeError('invalid score_function')
        score = F.softmax(score, dim=0)
        output = torch.bmm(score, kx)
        output = torch.cat(torch.split(output, mb_size, dim=0), dim=-1)
        output = self.proj(output)
        output = self.dropout(output)
        return output, score



class GRUModel(nn.Module):

    def __init__(self, D_m, D_e, D_h, n_classes=7, dropout=0.5):
        
        super(GRUModel, self).__init__()
        
        self.n_classes = n_classes
        self.dropout   = nn.Dropout(dropout)
        self.gru = nn.GRU(input_size=D_m, hidden_size=D_e, num_layers=2, bidirectional=True, dropout=dropout)
        self.matchatt = MatchingAttention(2*D_e, 2*D_e, att_type='general2')
        self.linear = nn.Linear(2*D_e, D_h)
        self.smax_fc = nn.Linear(D_h, n_classes)
        
    def forward(self, U, qmask, umask, att2=True):
        """
        U -> seq_len, batch, D_m
        qmask -> seq_len, batch, party
        """
        emotions, hidden = self.gru(U)
        alpha, alpha_f, alpha_b = [], [], []
        
        if att2:
            att_emotions = []
            alpha = []
            for t in emotions:
                att_em, alpha_ = self.matchatt(emotions,t,mask=umask)
                att_emotions.append(att_em.unsqueeze(0))
                alpha.append(alpha_[:,0,:])
            att_emotions = torch.cat(att_emotions,dim=0)
            hidden = F.relu(self.linear(att_emotions))
        else:
            hidden = F.relu(self.linear(emotions))
        
        hidden = self.dropout(hidden)
        log_prob = F.log_softmax(self.smax_fc(hidden), 2)
        return log_prob, alpha, alpha_f, alpha_b, emotions


class LSTMModel(nn.Module):

    def __init__(self, D_m, D_e, D_h, n_classes=7, dropout=0.5):
        
        super(LSTMModel, self).__init__()
        
        self.n_classes = n_classes
        self.dropout   = nn.Dropout(dropout)
        self.lstm = nn.LSTM(input_size=D_m, hidden_size=D_e, num_layers=2, bidirectional=True, dropout=dropout)
        self.matchatt = MatchingAttention(2*D_e, 2*D_e, att_type='general2')
        self.linear = nn.Linear(2*D_e, D_h)
        self.smax_fc = nn.Linear(D_h, n_classes)

    def forward(self, U, qmask, umask, att2=True):
        """
        U -> seq_len, batch, D_m
        qmask -> seq_len, batch, party
        """
        emotions, hidden = self.lstm(U)
        alpha, alpha_f, alpha_b = [], [], []
        
        if att2:
            att_emotions = []
            alpha = []
            for t in emotions:
                att_em, alpha_ = self.matchatt(emotions,t,mask=umask)
                att_emotions.append(att_em.unsqueeze(0))
                alpha.append(alpha_[:,0,:])
            att_emotions = torch.cat(att_emotions,dim=0)
            hidden = F.relu(self.linear(att_emotions))
        else:
            hidden = F.relu(self.linear(emotions))
        
        hidden = self.dropout(hidden)
        log_prob = F.log_softmax(self.smax_fc(hidden), 2)
        return log_prob, alpha, alpha_f, alpha_b, emotions


class MaskedEdgeAttention(nn.Module):

    def __init__(self, input_dim, max_seq_len, no_cuda):
        """
        Method to compute the edge weights, as in Equation 1. in the paper. 
        attn_type = 'attn1' refers to the equation in the paper.
        For slightly different attention mechanisms refer to attn_type = 'attn2' or attn_type = 'attn3'
        """

        super(MaskedEdgeAttention, self).__init__()
        
        self.input_dim = input_dim
        self.max_seq_len = max_seq_len
        self.scalar = nn.Linear(self.input_dim, self.max_seq_len, bias=False)
        self.matchatt = MatchingAttention(self.input_dim, self.input_dim, att_type='general2')
        self.simpleatt = SimpleAttention(self.input_dim)
        self.att = Attention(self.input_dim, score_function='mlp')
        self.no_cuda = no_cuda

    def forward(self, M, lengths, edge_ind):
        """
        M -> (seq_len, batch, vector)
        lengths -> length of the sequences in the batch
        edge_idn 
        """
        attn_type = 'attn1'

        if attn_type == 'attn1':

            scale = self.scalar(M)
            alpha = F.softmax(scale, dim=0).permute(1, 2, 0)
            if not self.no_cuda:
                mask = Variable(torch.ones(alpha.size()) * 1e-10).detach().cuda()
                mask_copy = Variable(torch.zeros(alpha.size())).detach().cuda()
                
            else:
                mask = Variable(torch.ones(alpha.size()) * 1e-10).detach()
                mask_copy = Variable(torch.zeros(alpha.size())).detach()
            
            edge_ind_ = []
            for i, j in enumerate(edge_ind):
                for x in j:
                    edge_ind_.append([i, x[0], x[1]])

            edge_ind_ = np.array(edge_ind_).transpose()
            mask[edge_ind_] = 1
            mask_copy[edge_ind_] = 1
            masked_alpha = alpha * mask
            _sums = masked_alpha.sum(-1, keepdim=True)
            scores = masked_alpha.div(_sums) * mask_copy

            return scores

        elif attn_type == 'attn2':
            scores = torch.zeros(M.size(1), self.max_seq_len, self.max_seq_len, requires_grad=True)

            if not self.no_cuda:
                scores = scores.cuda()


            for j in range(M.size(1)):
            
                ei = np.array(edge_ind[j])

                for node in range(lengths[j]):
                
                    neighbour = ei[ei[:, 0] == node, 1]

                    M_ = M[neighbour, j, :].unsqueeze(1)
                    t = M[node, j, :].unsqueeze(0)
                    _, alpha_ = self.simpleatt(M_, t)
                    scores[j, node, neighbour] = alpha_

        elif attn_type == 'attn3':
            scores = torch.zeros(M.size(1), self.max_seq_len, self.max_seq_len, requires_grad=True)

            if not self.no_cuda:
                scores = scores.cuda()

            for j in range(M.size(1)):

                ei = np.array(edge_ind[j])

                for node in range(lengths[j]):

                    neighbour = ei[ei[:, 0] == node, 1]

                    M_ = M[neighbour, j, :].unsqueeze(1).transpose(0, 1)
                    t = M[node, j, :].unsqueeze(0).unsqueeze(0).repeat(len(neighbour), 1, 1).transpose(0, 1)
                    _, alpha_ = self.att(M_, t)
                    scores[j, node, neighbour] = alpha_[0, :, 0]

        return scores


def pad(tensor, length, no_cuda):
    if isinstance(tensor, Variable):
        var = tensor
        if length > var.size(0):
            if not no_cuda:
                return torch.cat([var, torch.zeros(length - var.size(0), *var.size()[1:]).cuda()])
            else:
                return torch.cat([var, torch.zeros(length - var.size(0), *var.size()[1:])])
        else:
            return var
    else:
        if length > tensor.size(0):
            if not no_cuda:
                return torch.cat([tensor, torch.zeros(length - tensor.size(0), *tensor.size()[1:]).cuda()])
            else:
                return torch.cat([tensor, torch.zeros(length - tensor.size(0), *tensor.size()[1:])])
        else:
            return tensor


def simple_batch_graphify(features, lengths, no_cuda):
    edge_index, edge_norm, edge_type, node_features = [], [], [], []
    batch_size = features.size(1)
    for j in range(batch_size):
        node_features.append(features[:lengths[j], j, :])

    node_features = torch.cat(node_features, dim=0)  

    if not no_cuda:
        node_features = node_features.cuda()
    return node_features, None, None, None, None



class MMGatedAttention(nn.Module):

    def __init__(self, mem_dim, cand_dim, att_type='general'):
        super(MMGatedAttention, self).__init__()
        self.mem_dim = mem_dim
        self.cand_dim = cand_dim
        self.att_type = att_type
        self.dropouta = nn.Dropout(0.5)
        self.dropoutv = nn.Dropout(0.5)
        self.dropoutl = nn.Dropout(0.5)
        if att_type=='av_bg_fusion':
            self.transform_al = nn.Linear(mem_dim*2, cand_dim, bias=True)
            self.scalar_al = nn.Linear(mem_dim, cand_dim)
            self.transform_vl = nn.Linear(mem_dim*2, cand_dim, bias=True)
            self.scalar_vl = nn.Linear(mem_dim, cand_dim)
        elif att_type=='general':
            self.transform_l = nn.Linear(mem_dim, cand_dim, bias=True)
            self.transform_v = nn.Linear(mem_dim, cand_dim, bias=True)
            self.transform_a = nn.Linear(mem_dim, cand_dim, bias=True)
            self.transform_av = nn.Linear(mem_dim*3,1)
            self.transform_al = nn.Linear(mem_dim*3,1)
            self.transform_vl = nn.Linear(mem_dim*3,1)

    def forward(self, a, v, l, modals=None):
        a = self.dropouta(a) if len(a) !=0 else a
        v = self.dropoutv(v) if len(v) !=0 else v
        l = self.dropoutl(l) if len(l) !=0 else l
        if self.att_type == 'av_bg_fusion':
            if 'a' in modals:
                fal = torch.cat([a, l],dim=-1)
                Wa = torch.sigmoid(self.transform_al(fal))
                hma = Wa*(self.scalar_al(a))
            if 'v' in modals:
                fvl = torch.cat([v, l],dim=-1)
                Wv = torch.sigmoid(self.transform_vl(fvl))
                hmv = Wv*(self.scalar_vl(v))
            if len(modals) == 3:
                hmf = torch.cat([l,hma,hmv], dim=-1)
            elif 'a' in modals:
                hmf = torch.cat([l,hma], dim=-1)
            elif 'v' in modals:
                hmf = torch.cat([l,hmv], dim=-1)
            return hmf
        elif self.att_type == 'general':
            ha = torch.tanh(self.transform_a(a)) if 'a' in modals else a
            hv = torch.tanh(self.transform_v(v)) if 'v' in modals else v
            hl = torch.tanh(self.transform_l(l)) if 'l' in modals else l

            if 'a' in modals and 'v' in modals:
                z_av = torch.sigmoid(self.transform_av(torch.cat([a,v,a*v],dim=-1)))
                h_av = z_av*ha + (1-z_av)*hv
                if 'l' not in modals:
                    return h_av
            if 'a' in modals and 'l' in modals:
                z_al = torch.sigmoid(self.transform_al(torch.cat([a,l,a*l],dim=-1)))
                h_al = z_al*ha + (1-z_al)*hl
                if 'v' not in modals:
                    return h_al
            if 'v' in modals and 'l' in modals:
                z_vl = torch.sigmoid(self.transform_vl(torch.cat([v,l,v*l],dim=-1)))
                h_vl = z_vl*hv + (1-z_vl)*hl
                if 'a' not in modals:
                    return h_vl
            return torch.cat([h_av, h_al, h_vl],dim=-1)



class Model(nn.Module):
    def __init__(self, base_model, D_m, D_g, D_m_v, D_m_a, n_speakers, max_seq_len, window_past, window_future, 
                 n_classes=7, dropout=0.5, 
                 no_cuda=False, use_residue=True, modals='avl', av_using_lstm=False, dataset='IEMOCAP',
                 backbone='GCN', use_speaker=True, norm='None', 
                 num_L=4, use_residue_denoise=True, denoise_dropout = 0, gamma=-0.95, zeta=1.05, temperature=2.0,
                 num_K=4, use_gumbel=True, gumbel_k=30, nodeformer_heads=4, nb_features_dim=256, tau=0.25, nodeformer_dropout=0, use_residue_nodeformer=True, use_jk_nodeformer=False):
        
        super(Model, self).__init__()
        # choice = ['wo_nodeformer', 'wo_denoise', 'None']
        # self.ablation = 'wo_nodeformer'
        # self.ablation = 'wo_denoise'
        self.ablation = 'None'
        self.use_concat = False
        if backbone == 'MMGCN':
            self.use_concat = True
        self.use_gate_fusion = True
        self.base_model = base_model
        self.no_cuda = False
        self.dropout = dropout
        self.denoise_dropout = denoise_dropout
        self.use_residue_denoise = use_residue_denoise
        self.use_residue_nodeformer = use_residue_nodeformer
        self.use_residue = use_residue
        self.return_feature = True
        self.modals = [x for x in modals]  # a, v, l
        self.use_speaker = use_speaker
        self.n_speakers = n_speakers
        self.backbone = backbone
        self.tau = tau
        self.gumbel_k = gumbel_k
        self.dataset = dataset
        self.use_gumbel = use_gumbel
        if norm == 'LN':            
            self.normLNa = nn.LayerNorm(1024, elementwise_affine=True)
            self.normLNb = nn.LayerNorm(1024, elementwise_affine=True)
            self.normLNc = nn.LayerNorm(1024, elementwise_affine=True)
            self.normLNd = nn.LayerNorm(1024, elementwise_affine=True)     
        elif norm == 'BN':
            self.normBNa = nn.BatchNorm1d(1024, affine=True)
            self.normBNb = nn.BatchNorm1d(1024, affine=True)
            self.normBNc = nn.BatchNorm1d(1024, affine=True)
            self.normBNd = nn.BatchNorm1d(1024, affine=True)

        self.norm_strategy = norm
        self.av_using_lstm = av_using_lstm
        self.use_bert_seq = False
        self.dataset = dataset
        if 'a' in self.modals:
            hidden_a = D_g
            self.linear_a = nn.Linear(D_m_a, hidden_a)
            if self.av_using_lstm:
                self.gru_a = nn.GRU(input_size=hidden_a, hidden_size=D_g//2, num_layers=2, bidirectional=True, dropout=dropout)
        if 'v' in self.modals:
            hidden_v = D_g
            self.linear_v = nn.Linear(D_m_v, hidden_v)
            if self.av_using_lstm:
                self.gru_v = nn.GRU(input_size=hidden_v, hidden_size=D_g//2, num_layers=2, bidirectional=True, dropout=dropout)
        if 'l' in self.modals:
            hidden_l = D_g
            if self.use_bert_seq:
                self.txtCNN = TextCNN(input_dim=D_m, emb_size=hidden_l)
            else:
                self.linear_l = nn.Linear(D_m, hidden_l)
            self.gru_l = nn.GRU(input_size=hidden_l, hidden_size=D_g//2, num_layers=2, bidirectional=True, dropout=dropout)
        
        self.speaker_weights_l1 = nn.Linear(hidden_l, hidden_l)
        self.speaker_weights_l2 = nn.Linear(hidden_l, hidden_l)
        self.speaker_weights_a1 = nn.Linear(hidden_a, hidden_a)
        self.speaker_weights_a2 = nn.Linear(hidden_a, hidden_a)
        self.speaker_weights_v1 = nn.Linear(hidden_v, hidden_v)
        self.speaker_weights_v2 = nn.Linear(hidden_v, hidden_v)

        self.rnn_parties_l = nn.GRU(input_size=hidden_l, hidden_size=int(hidden_l/2), num_layers=2, bidirectional=True, dropout=dropout)
        self.rnn_parties_a = nn.GRU(input_size=hidden_a, hidden_size=int(hidden_a/2), num_layers=2, bidirectional=True, dropout=dropout)
        self.rnn_parties_v = nn.GRU(input_size=hidden_v, hidden_size=int(hidden_v/2), num_layers=2, bidirectional=True, dropout=dropout)


        self.conv_layer = num_L
        if self.backbone == 'GCN':
            for k in range(self.conv_layer):
                # setattr(self,'conv{}'.format(k), GraphConvLayer(hidden_l, hidden_l, denoise=True, denoise_type=denoise_type))
                setattr(self,'conv{}'.format(k), GraphConvLayer(hidden_l, hidden_l, gamma=gamma, zeta=zeta, temperature=temperature, denoise=True))
                # self.gate_denoise = nn.Linear(hidden_l, 1)

        elif self.backbone == 'GAT':
            for k in range(self.conv_layer):
                setattr(self,'conv{}'.format(k), GraphAttentionLayer(hidden_l, hidden_l, gamma=gamma, zeta=zeta, temperature=temperature, denoise=True))
                # self.gate_denoise = nn.Linear(hidden_l, 1)
        elif self.backbone == 'MMGCN':
            if self.dataset == 'IEMOCAP':
                setattr(self,'conv', MM_GCN(a_dim=hidden_l, v_dim=hidden_l, l_dim=hidden_l, n_dim=200, nlayers=64, nhidden=100, nclass=6, dropout=0.4, lamda=0.5, alpha=0.1, variant=True, return_feature=True, use_residue=True, n_speakers=2, modals=['a', 'v', 'l'], use_speaker=True, use_modal=False))
            else:
                pass


        elif self.backbone == 'M3Net':
            if self.dataset == 'IEMOCAP':
                setattr(self,'hypergraph', HyperGCN(a_dim=512, v_dim=512, l_dim=512, n_dim=hidden_l, nlayers=64,  nhidden=hidden_l, nclass=6, 
                                        dropout=0.5, lamda=0.5, alpha=0.1, variant=True, return_feature=True, use_residue=False, 
                                        n_speakers=2, modals=['a', 'v', 'l'], use_speaker=True, use_modal=False, num_L=3, num_K=self.conv_layer))
            else:
                setattr(self,'hypergraph', HyperGCN(a_dim=1024, v_dim=1024, l_dim=1024, n_dim=hidden_l, nlayers=64, nhidden=hidden_l, nclass=7, 
                                        dropout=0.4, lamda=0.5, alpha=0.1, variant=True, return_feature=True, use_residue=False, 
                                        n_speakers=9, modals=['a', 'v', 'l'], use_speaker=True, use_modal=False, num_L=3, num_K=self.conv_layer))

        


        self.node_former_layer = num_K
        self.nodeformer = NodeFormer(hidden_l, hidden_l, num_layers=self.node_former_layer, dropout=nodeformer_dropout,
                    num_heads=nodeformer_heads, use_bn=True, nb_random_features=nb_features_dim,
                    use_gumbel=self.use_gumbel, use_residual=self.use_residue_nodeformer, use_act=False, use_jk=use_jk_nodeformer,
                    nb_gumbel_sample=self.gumbel_k, rb_trans='sigmoid')
        
        # self.nodeformer = NodeFormerFast(hidden_l, hidden_l, num_layers=self.node_former_layer, dropout=nodeformer_dropout,
        #             num_heads=nodeformer_heads, use_bn=True, nb_random_features=nb_features_dim,
        #             use_gumbel=True, use_residual=True, use_act=False, use_jk=False,
        #             nb_gumbel_sample=self.gumbel_k, rb_trans='sigmoid',
        #             projection_matrix_input_dim=hidden_l, seed=1475)
        
        
        # self.gate_nodeformer = nn.Linear(hidden_l, 1)
        
        #self.gatedatt = MMGatedAttention(D_g + graph_hidden_size, graph_hidden_size, att_type='general')

        self.dropout_fc = nn.Dropout(self.dropout)
        
        if self.backbone == 'M3Net':
            if self.use_concat:
                if self.use_residue:
                    self.smax_fc = nn.Linear(hidden_l*len(self.modals) * 4, n_classes)
                else:
                    self.smax_fc = nn.Linear(hidden_l*len(self.modals) * 3, n_classes)
            else:
                self.smax_fc = nn.Linear(hidden_l*len(self.modals), n_classes)
                if self.use_gate_fusion:
                    self.gate_fusion_origin = nn.Linear(hidden_l, hidden_l)
                    self.gate_fusion_denoise = nn.Linear(hidden_l, hidden_l)
                    self.gate_fusion_nodeformer = nn.Linear(hidden_l, hidden_l)
        elif self.backbone == 'MMGCN':
            if self.dataset == 'IEMOCAP':
                self.smax_fc = nn.Linear(hidden_l*len(self.modals) * 2 + 300*3, n_classes)
            else:
                pass
        else:
            if self.use_concat:
                if self.ablation == 'None':
                    if self.use_residue:
                        self.smax_fc = nn.Linear(hidden_l*len(self.modals) * 3, n_classes)
                    else:
                        self.smax_fc = nn.Linear(hidden_l*len(self.modals) * 2, n_classes)
                else:
                    if self.use_residue:
                        self.smax_fc = nn.Linear(hidden_l*len(self.modals) * 2, n_classes)
                    else:
                        self.smax_fc = nn.Linear(hidden_l*len(self.modals) * 1, n_classes)
            else:
                self.smax_fc = nn.Linear(hidden_l*len(self.modals), n_classes)
                if self.use_gate_fusion:
                    self.gate_fusion_origin = nn.Linear(hidden_l, hidden_l)
                    self.gate_fusion_denoise = nn.Linear(hidden_l, hidden_l)
                    self.gate_fusion_nodeformer = nn.Linear(hidden_l, hidden_l)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.linear_l.weight)
        nn.init.xavier_uniform_(self.linear_a.weight)
        nn.init.xavier_uniform_(self.linear_v.weight)
        nn.init.xavier_uniform_(self.speaker_weights_l1.weight)
        nn.init.xavier_uniform_(self.speaker_weights_a1.weight)
        nn.init.xavier_uniform_(self.speaker_weights_v1.weight)
        nn.init.xavier_uniform_(self.speaker_weights_l2.weight)
        nn.init.xavier_uniform_(self.speaker_weights_a2.weight)
        nn.init.xavier_uniform_(self.speaker_weights_v2.weight)
        nn.init.xavier_uniform_(self.speaker_weights_l1.weight)
        nn.init.xavier_uniform_(self.speaker_weights_l2.weight)
        nn.init.xavier_uniform_(self.speaker_weights_a1.weight)
        nn.init.xavier_uniform_(self.speaker_weights_a2.weight)
        nn.init.xavier_uniform_(self.speaker_weights_v1.weight)
        nn.init.xavier_uniform_(self.speaker_weights_v2.weight)
        nn.init.xavier_uniform_(self.smax_fc.weight)

    def _reverse_seq(self, X, mask):
        X_ = X.transpose(0,1)
        mask_sum = torch.sum(mask, 1).int()

        xfs = []
        for x, c in zip(X_, mask_sum):
            xf = torch.flip(x[:c], [0])
            xfs.append(xf)

        return pad_sequence(xfs)


    def forward(self, U, qmask, umask, seq_lengths, U_a=None, U_v=None, epoch=None, label=None, train=False):

        #=============roberta features
        [r1,r2,r3,r4]=U
        seq_len, _, feature_dim = r1.size()
        if self.norm_strategy == 'LN':
            r1 = self.normLNa(r1.transpose(0, 1).reshape(-1, feature_dim)).reshape(-1, seq_len, feature_dim).transpose(1, 0)
            r2 = self.normLNb(r2.transpose(0, 1).reshape(-1, feature_dim)).reshape(-1, seq_len, feature_dim).transpose(1, 0)
            r3 = self.normLNc(r3.transpose(0, 1).reshape(-1, feature_dim)).reshape(-1, seq_len, feature_dim).transpose(1, 0)
            r4 = self.normLNd(r4.transpose(0, 1).reshape(-1, feature_dim)).reshape(-1, seq_len, feature_dim).transpose(1, 0)
            U = (r1 + r2 + r3 + r4)/4
        elif self.norm_strategy == 'BN':
            r1 = self.normBNa(r1.transpose(0, 1).reshape(-1, feature_dim)).reshape(-1, seq_len, feature_dim).transpose(1, 0)
            r2 = self.normBNb(r2.transpose(0, 1).reshape(-1, feature_dim)).reshape(-1, seq_len, feature_dim).transpose(1, 0)
            r3 = self.normBNc(r3.transpose(0, 1).reshape(-1, feature_dim)).reshape(-1, seq_len, feature_dim).transpose(1, 0)
            r4 = self.normBNd(r4.transpose(0, 1).reshape(-1, feature_dim)).reshape(-1, seq_len, feature_dim).transpose(1, 0)
            U = (r1 + r2 + r3 + r4)/4
        elif self.norm_strategy == 'LN2':
            norm2 = nn.LayerNorm((seq_len, feature_dim), elementwise_affine=False)
            r1 = norm2(r1.transpose(0, 1)).transpose(0, 1)
            r2 = norm2(r2.transpose(0, 1)).transpose(0, 1)
            r3 = norm2(r3.transpose(0, 1)).transpose(0, 1)
            r4 = norm2(r4.transpose(0, 1)).transpose(0, 1)
            U = (r1 + r2 + r3 + r4)/4
        elif self.norm_strategy == 'mean':
            U = (r1 + r2 + r3 + r4)/4
        else:
            if self.dataset == 'IEMOCAP':
                U = r1
            else:
                U = (r1 + r2 + r3 + r4)/4
                # U = r1
 
        # U = self.dropout_forward(U)
        # U_a = self.dropout_forward(U_a) if U_a is not None else None
        # U_v = self.dropout_forward(U_v) if U_v is not None else None
        emotions_a, emotions_v, emotions_l = None, None, None
        if 'l' in self.modals:
            U = self.linear_l(U)
            emotions_l, hidden_l = self.gru_l(U)
            U_, qmask_ = U.transpose(0, 1), qmask.transpose(0, 1)
            U_p_ = torch.zeros(U_.size()[0], U_.size()[1], emotions_l.shape[-1]).type(U.type())
            U_parties_ = [torch.zeros_like(U_).type(U_.type()) for _ in range(self.n_speakers)]  # default 2
            for b in range(U_.size(0)):
                for p in range(len(U_parties_)):
                    index_i = torch.nonzero(qmask_[b][:, p]).squeeze(-1)
                    if index_i.size(0) > 0:
                        U_parties_[p][b][:index_i.size(0)] = U_[b][index_i]
            E_parties_ = [self.rnn_parties_l(U_parties_[p].transpose(0, 1))[0].transpose(0, 1) for p in range(len(U_parties_))]
            for b in range(U_p_.size(0)):
                for p in range(len(U_parties_)):
                    index_i = torch.nonzero(qmask_[b][:, p]).squeeze(-1)
                    if index_i.size(0) > 0: U_p_[b][index_i] = E_parties_[p][b][:index_i.size(0)]
            U_p = U_p_.transpose(0, 1)
            emotions_l = torch.nn.Sigmoid()(self.speaker_weights_l1(emotions_l)) * emotions_l + torch.nn.Sigmoid()(self.speaker_weights_l2(U_p)) * U_p

        if 'a' in self.modals:
            U_a = self.linear_a(U_a)
            emotions_a = U_a
            if self.av_using_lstm:
                emotions_a, hidden_a = self.gru_a(U_a)
            U_, qmask_ = U_a.transpose(0, 1), qmask.transpose(0, 1)
            U_p_ = torch.zeros(U_.size()[0], U_.size()[1], emotions_a.shape[-1]).type(U_a.type())
            U_parties_ = [torch.zeros_like(U_).type(U_.type()) for _ in range(self.n_speakers)]  # default 2
            for b in range(U_.size(0)):
                for p in range(len(U_parties_)):
                    index_i = torch.nonzero(qmask_[b][:, p]).squeeze(-1)
                    if index_i.size(0) > 0:
                        U_parties_[p][b][:index_i.size(0)] = U_[b][index_i]
            E_parties_ = [self.rnn_parties_a(U_parties_[p].transpose(0, 1))[0].transpose(0, 1) for p in range(len(U_parties_))]
            for b in range(U_p_.size(0)):
                for p in range(len(U_parties_)):
                    index_i = torch.nonzero(qmask_[b][:, p]).squeeze(-1)
                    if index_i.size(0) > 0: U_p_[b][index_i] = E_parties_[p][b][:index_i.size(0)]
            U_p = U_p_.transpose(0, 1)
            emotions_a = torch.nn.Sigmoid()(self.speaker_weights_a1(emotions_a)) * emotions_a + torch.nn.Sigmoid()(self.speaker_weights_a2(U_p)) * U_p

        if 'v' in self.modals:
            U_v = self.linear_v(U_v)
            emotions_v = U_v
            if self.av_using_lstm:
                emotions_v, hidden_v = self.gru_v(U_v)
            U_, qmask_ = U_v.transpose(0, 1), qmask.transpose(0, 1)
            U_p_ = torch.zeros(U_.size()[0], U_.size()[1], emotions_v.shape[-1]).type(U_v.type())
            U_parties_ = [torch.zeros_like(U_).type(U_.type()) for _ in range(self.n_speakers)]  # default 2
            for b in range(U_.size(0)):
                for p in range(len(U_parties_)):
                    index_i = torch.nonzero(qmask_[b][:, p]).squeeze(-1)
                    if index_i.size(0) > 0:
                        U_parties_[p][b][:index_i.size(0)] = U_[b][index_i]
            E_parties_ = [self.rnn_parties_v(U_parties_[p].transpose(0, 1))[0].transpose(0, 1) for p in range(len(U_parties_))]
            for b in range(U_p_.size(0)):
                for p in range(len(U_parties_)):
                    index_i = torch.nonzero(qmask_[b][:, p]).squeeze(-1)
                    if index_i.size(0) > 0: U_p_[b][index_i] = E_parties_[p][b][:index_i.size(0)]
            U_p = U_p_.transpose(0, 1)
            emotions_v = torch.nn.Sigmoid()(self.speaker_weights_v1(emotions_v)) * emotions_v + torch.nn.Sigmoid()(self.speaker_weights_v2(U_p)) * U_p

        if 'l' in self.modals:
            features_l, edge_index, edge_norm, edge_type, edge_index_lengths = simple_batch_graphify(emotions_l, seq_lengths, self.no_cuda)
        else:
            features_l = []
        if 'a' in self.modals:
            features_a, edge_index, edge_norm, edge_type, edge_index_lengths = simple_batch_graphify(emotions_a, seq_lengths, self.no_cuda)
        else:
            features_a = []
        if 'v' in self.modals:
            features_v, edge_index, edge_norm, edge_type, edge_index_lengths = simple_batch_graphify(emotions_v, seq_lengths, self.no_cuda)
        else:
            features_v = []
        

        speakers = torch.nonzero(torch.cat([qmask[:seq_lengths[dia_len_index],dia_len_index,:] for dia_len_index in range(len(seq_lengths))], dim=0), as_tuple=True)[1].detach()
        # create features and dialogue_index
        graph_features, dialogue_index = self.create_graph_features(features_l, features_a, features_v, seq_lengths, modals=self.modals)
        penalty_weight_loss = 0
        link_loss = 0

        # denoise module
        if self.ablation != 'wo_denoise':
            denoise_gnn_features = graph_features
            if self.backbone == 'GCN' or self.backbone == 'GAT':
                denoise_gnn_edge_index, denoise_gnn_edge_type = self.create_gnn_index(dia_len=seq_lengths, modals=self.modals, speakers=speakers)
                denoise_layer = [graph_features]
                for k in range(self.conv_layer):
                    denoise_gnn_features = F.dropout(denoise_gnn_features, p=self.denoise_dropout, training=train)
                    denoise_gnn_output = getattr(self,'conv{}'.format(k))(denoise_gnn_features, denoise_gnn_edge_index, edge_type=denoise_gnn_edge_type, train=train)
                    denoise_gnn_features = denoise_gnn_output[0] 
                    if self.use_residue_denoise:
                        denoise_gnn_features += denoise_layer[k]
                    penalty_weight_loss += denoise_gnn_output[1]
                    denoise_layer.append(denoise_gnn_features)
            elif self.backbone == 'MMGCN':
                denoise_gnn_features, penalty_weight_loss = getattr(self,'conv')(features_a, features_v, features_l, seq_lengths, qmask)
                denoise_gnn_features, _ = self.create_graph_features(denoise_gnn_features[0], denoise_gnn_features[1], denoise_gnn_features[2], seq_lengths, modals=self.modals)
                
            elif self.backbone == 'M3Net':
                denoise_gnn_features, penalty_weight_loss = self.hypergraph(features_a, features_v, features_l, seq_lengths, qmask, epoch, speakers)

        # node former
        if self.ablation != 'wo_nodeformer':
            nodeformer_gnn_edge_index, nodeformer_dialogue_index = self.create_nodeformer_edges(dia_len=seq_lengths, modals=self.modals)
            nodeformer_gnn_features = []
            for i, length in enumerate(seq_lengths):
                current_dialogue_nodeformer_features, current_link_loss = self.nodeformer(graph_features[dialogue_index==i], nodeformer_gnn_edge_index[:,nodeformer_dialogue_index==i], tau=self.tau)
                link_loss += sum(current_link_loss) / len(current_link_loss)
                nodeformer_gnn_features.append(current_dialogue_nodeformer_features)        
            nodeformer_gnn_features = torch.cat(nodeformer_gnn_features, dim=0)

        
        # features = graph_features + denoise_gnn_features
        # features = graph_features + nodeformer_gnn_features
        # features = graph_features + denoise_gnn_features + nodeformer_gnn_features

        if self.ablation == 'None':
            if self.use_concat:
                if self.use_residue:
                    features = torch.cat([graph_features, denoise_gnn_features, nodeformer_gnn_features], -1)
                else:
                    features = torch.cat([denoise_gnn_features, nodeformer_gnn_features], -1)
            else:
                if self.use_residue:
                    features = nn.Sigmoid()(self.gate_fusion_origin(graph_features)) * graph_features + \
                                nn.Sigmoid()(self.gate_fusion_denoise(denoise_gnn_features)) * denoise_gnn_features + \
                                nn.Sigmoid()(self.gate_fusion_nodeformer(nodeformer_gnn_features)) * nodeformer_gnn_features
                else:
                    features = nn.Sigmoid()(self.gate_fusion_denoise(denoise_gnn_features)) * denoise_gnn_features + \
                                nn.Sigmoid()(self.gate_fusion_nodeformer(nodeformer_gnn_features)) * nodeformer_gnn_features
        elif self.ablation == 'wo_denoise':
            if self.use_concat:
                if self.use_residue:
                    features = torch.cat([graph_features, nodeformer_gnn_features], -1)
                else:
                    features = nodeformer_gnn_features
            else:
                if self.use_residue:
                    features = nn.Sigmoid()(self.gate_fusion_origin(graph_features)) * graph_features + \
                                nn.Sigmoid()(self.gate_fusion_nodeformer(nodeformer_gnn_features)) * nodeformer_gnn_features
                else:
                    features = nn.Sigmoid()(self.gate_fusion_nodeformer(nodeformer_gnn_features)) * nodeformer_gnn_features
        elif self.ablation == 'wo_nodeformer':
            if self.use_concat:
                if self.use_residue:
                    features = torch.cat([graph_features, denoise_gnn_features], -1)
                else:
                    features = denoise_gnn_features
            else:
                if self.use_residue:
                    features = nn.Sigmoid()(self.gate_fusion_origin(graph_features)) * graph_features + \
                                nn.Sigmoid()(self.gate_fusion_denoise(denoise_gnn_features)) * denoise_gnn_features
                else:
                    features = nn.Sigmoid()(self.gate_fusion_denoise(denoise_gnn_features)) * denoise_gnn_features 
        else:
            raise NotImplementedError

        features = self.reverse_features(seq_lengths, features)

        # features = torch.cat([features_l, features_a, features_v], dim=-1)
        features = nn.ReLU()(self.dropout_fc(features))
        log_prob = F.log_softmax(self.smax_fc(features), 1)

        return log_prob, edge_index, edge_norm, edge_type, edge_index_lengths, penalty_weight_loss, link_loss

    def create_graph_features(self, l, a, v, dia_len, modals):
        features = []
        dialogue_index = []
        temp = 0
        for i, length in enumerate(dia_len):
            current_dialogue_features = []
            if 'l' in modals:
                current_dialogue_features.append(l[temp: temp + length])
            if 'a' in modals:
                current_dialogue_features.append(a[temp: temp + length])
            if 'v' in modals:
                current_dialogue_features.append(v[temp: temp + length])
            dialogue_index.extend([i] * (len(modals) * length))
            features.append(torch.cat(current_dialogue_features, dim=0))
            temp += length
        features = torch.cat(features, dim=0)
        dialogue_index = torch.tensor(dialogue_index)
        return features, dialogue_index


    def create_nodeformer_edges(self, dia_len, modals):
        index = []
        dialogues_index = []
        for i, length in enumerate(dia_len):
            current_index = []
            nodes = list(range(length*len(modals)))
            current_node_count = 0
            if 'l' in modals:
                nodes_l = nodes[current_node_count: current_node_count + length]
                current_node_count += length
                current_index += list(permutations(nodes_l, 2))
            if 'a' in modals:
                nodes_a = nodes[current_node_count: current_node_count + length]
                current_node_count += length
                current_index += list(permutations(nodes_a, 2))
            if 'v' in modals:
                nodes_v = nodes[current_node_count: current_node_count + length]
                current_node_count += length
                current_index += list(permutations(nodes_v, 2))
            for _ in range(length):
                current_index = current_index + list(permutations([nodes_l[_]] + [nodes_a[_]] + [nodes_v[_]],2))
            dialogues_index.extend([i] * len(current_index))
            index = index + current_index
        edge_index = torch.LongTensor(index).T.cuda()
        dialogues_index = torch.tensor(dialogues_index).cuda()
        return edge_index, dialogues_index

    # def create_gnn_index(self, dia_len, modals, speakers):
    #     num_modality = len(modals)
    #     node_count = 0
    #     indexes_context = []
    #     indexes_modal = []
    #     categorys_context = []
    #     categorys_modal = []

    #     # Precompute node speakers once
    #     node_speakers = torch.cat([speakers[start: start + length].repeat(len(modals)) 
    #                             for start, length in zip(list(itertools.accumulate([0] + dia_len[:-1])), dia_len)])
    #     for length in dia_len:
    #         nodes_base = list(range(node_count, node_count + length * num_modality))
    #         node_count += length * num_modality

    #         # Separate nodes for each modality
    #         modal_nodes = {modal: nodes_base[current_node_count:current_node_count + length]
    #                    for modal, current_node_count in zip(sorted(modals, key={'l': 0, 'a': 1, 'v': 2}.get), itertools.accumulate([0] + [length for _ in range(len(modals)-1)]))} #TODO: 处理modals的顺序
    #         # Temporal contextual modeling
    #         for modal, nodes in modal_nodes.items():
    #             if len(nodes) != 1:
    #                 index_context_modal = torch.tensor(list(permutations(nodes, 2))).cuda()
    #                 category_context_modal = (node_speakers[index_context_modal[:, 0]] == node_speakers[index_context_modal[:, 1]]).int().cuda()
    #                 assert len(index_context_modal) == len(category_context_modal)
    #                 indexes_context.append(index_context_modal)
    #                 categorys_context.append(category_context_modal)
    #         # Cross-modal modeling
    #         for i in range(length):
    #             index_cross_modal = torch.tensor(list(permutations([modal_nodes[modal][i] for modal in modals], 2))).cuda()
    #             indexes_modal.append(index_cross_modal)
    #             categorys_modal.append(torch.ones(index_cross_modal.shape[0]).cuda())
        
    #     edge_index = torch.cat([torch.cat(indexes_context, dim=0), torch.cat(indexes_modal, dim=0)], dim=0).T
    #     category = torch.cat([torch.cat(categorys_context, dim=0), torch.cat(categorys_modal, dim=0)], dim=0)
    #     assert edge_index.shape[1] == category.shape[0]
    #     return edge_index, category

    # def create_gnn_index(self, dia_len, modals, speakers):
    #     num_modality = len(modals)
    #     node_count = 0
    #     index = []
    #     category = []
    #     tmp = []

    #     # Precompute node speakers once
    #     node_speakers = torch.cat([speakers[start: start + length].repeat(len(modals)) 
    #                             for start, length in zip(list(itertools.accumulate([0] + dia_len[:-1])), dia_len)])
    #     for length in dia_len:
    #         nodes_base = list(range(node_count, node_count + length * num_modality))
    #         node_count += length * num_modality

    #         # Separate nodes for each modality
    #         modal_nodes = {modal: nodes_base[current_node_count:current_node_count + length]
    #                    for modal, current_node_count in zip(sorted(modals, key={'l': 0, 'a': 1, 'v': 2}.get), itertools.accumulate([0] + [length for _ in range(len(modals)-1)]))} #TODO: 处理modals的顺序
    #         # Temporal contextual modeling
    #         for modal, nodes in modal_nodes.items():
    #             if len(nodes) != 1:
    #                 index_modal = torch.tensor(list(permutations(nodes, 2))).cuda()
    #                 category_modal = (node_speakers[index_modal[:, 0]] == node_speakers[index_modal[:, 1]]).int().cuda()
    #                 assert len(index_modal) == len(category_modal)
    #                 index.append(index_modal)
    #                 category.append(category_modal)
    #         # Cross-modal modeling
    #         for i in range(length):
    #             index_ms = torch.tensor(list(permutations([modal_nodes[modal][i] for modal in modals], 2))).cuda()
    #             tmp.append(index_ms)
    #             category.append(torch.ones(len(index_ms)).cuda())
        
    #     edge_index = torch.cat([torch.cat(index, dim=0), torch.cat(tmp, dim=0)], dim=0).T
    #     category = torch.cat(category, dim=0)
    #     return edge_index, category


    def create_gnn_index(self, dia_len, modals, speakers):
        num_modality = len(modals)
        node_count = 0
        index =[]
        tmp = []
        category = []
        category_tmp = []

        # type1
        node_speakers = torch.cat([torch.cat([speakers[length_count: length_count + length]] * len(modals)) for length_count, length in zip(itertools.accumulate([0] + dia_len[:-1]), dia_len)])

        for length in dia_len:
            nodes = list(range(length*num_modality))
            nodes = [j + node_count for j in nodes]
            current_node_count = 0
            if 'l' in modals:
                nodes_l = nodes[current_node_count: current_node_count + length]
                current_node_count += length
                index_l = list(permutations(nodes_l,2))
                category_l = (node_speakers[torch.tensor(index_l)[:, 0]] == node_speakers[torch.tensor(index_l)[:, 1]]).int().tolist() if len(index_l) != 0 else []
                assert len(index_l) == len(category_l)
                index = index + index_l
                category = category + category_l

            if 'a' in modals:
                nodes_a = nodes[current_node_count: current_node_count + length]
                current_node_count += length
                index_a = list(permutations(nodes_a,2))
                category_a = (node_speakers[torch.tensor(index_a)[:, 0]] == node_speakers[torch.tensor(index_a)[:, 1]]).int().tolist() if len(index_a) != 0 else []
                assert len(index_a) == len(category_a)
                index = index + index_a
                category = category + category_a
            if 'v' in modals:
                nodes_v = nodes[current_node_count: current_node_count + length]
                current_node_count += length
                index_v = list(permutations(nodes_v,2))
                category_v = (node_speakers[torch.tensor(index_v)[:, 0]] == node_speakers[torch.tensor(index_v)[:, 1]]).int().tolist() if len(index_v) != 0 else []
                assert len(index_v) == len(category_v)
                index = index + index_v
                category = category + category_v

            for _ in range(length):
                index_ms = list(permutations([nodes_l[_]] + [nodes_a[_]] + [nodes_v[_]], 2))
                tmp = tmp + index_ms
                category_tmp = category_tmp + [2] * len(index_ms)
                
            node_count = node_count + length * num_modality
        edge_index = torch.cat([torch.LongTensor(index).T,torch.LongTensor(tmp).T],1).cuda()
        edge_category = torch.cat([torch.tensor(category), torch.tensor(category_tmp)]).cuda()
        return edge_index, edge_category
    


    def reverse_features(self, dia_len, features):
        l=[]
        a=[]
        v=[]
        for i in dia_len:
            ll = features[0:1*i]
            aa = features[1*i:2*i]
            vv = features[2*i:3*i]
            features = features[3*i:]
            l.append(ll)
            a.append(aa)
            v.append(vv)
        tmpl = torch.cat(l,dim=0)
        tmpa = torch.cat(a,dim=0)
        tmpv = torch.cat(v,dim=0)
        features = torch.cat([tmpl, tmpa, tmpv], dim=-1)
        return features




    def create_drop_gnn_index(self, l, a, v, dia_len, modals, label=None, drop_rate=0, drop_positive_rate=0, drop_negative_rate=0):
        self_loop = False
        num_modality = len(modals)
        node_count = 0
        batch_count = 0
        index =[]
        tmp = []
        # print(drop_rate)
        # print(drop_positive_rate)
        # print(drop_negative_rate)

        for dia_len_index in range(len(dia_len)):
            d_len = dia_len[dia_len_index]
            nodes = list(range(d_len*num_modality))

            nodes_l = nodes[0:d_len*num_modality//3]
            nodes_a = nodes[d_len*num_modality//3:d_len*num_modality*2//3]
            nodes_v = nodes[d_len*num_modality*2//3:]
            #### 获得l模态的边，a模态的边，v模态的边
            conversation_index = list(permutations(nodes_l,2)) + list(permutations(nodes_a,2)) + list(permutations(nodes_v,2))

            Gnodes=[]
            for _ in range(d_len):
                Gnodes.append([nodes_l[_]] + [nodes_a[_]] + [nodes_v[_]])
            for ii, _ in enumerate(Gnodes):
                modal_index = list(permutations(_,2))
            if node_count == 0:
                ll = l[0:0+d_len]
                aa = a[0:0+d_len]
                vv = v[0:0+d_len]
                current_label = label[0:0+d_len].tolist()
                features = torch.cat([ll,aa,vv],dim=0)
                temp = 0+d_len
            else:
                ll = l[temp:temp+d_len]
                aa = a[temp:temp+d_len]
                vv = v[temp:temp+d_len]
                current_label = label[temp:temp+d_len].tolist()
                features_temp = torch.cat([ll,aa,vv],dim=0)
                features =  torch.cat([features,features_temp],dim=0)
                temp = temp+d_len
            current_label = current_label + current_label + current_label

            if drop_rate == 0 and drop_positive_rate == 0 and drop_negative_rate == 0:
                indexes = conversation_index + modal_index
            else:
                if drop_rate > 0:
                    indexes = drop_random_samples(conversation_index + modal_index, drop_rate)
                else:
                    index_positive_tobesample, index_negative_tobesample = parallel_process_indices(conversation_index + modal_index, current_label)
                    indexes = drop_random_samples(index_positive_tobesample, drop_positive_rate) + drop_random_samples(index_negative_tobesample, drop_negative_rate)

            #### 加上位置偏移
            index = index + [(_[0] + node_count, _[1] + node_count) for _ in indexes]
            node_count = node_count + d_len*num_modality
            
        # edge_index = torch.cat([torch.LongTensor(index).T,torch.LongTensor(tmp).T],1).cuda()
        edge_index = torch.LongTensor(index).T.cuda()

        return edge_index, features
    



def drop_random_samples(data_list, drop_ratio):
    total_samples = len(data_list)
    num_to_drop = int(total_samples * drop_ratio)  # 将比例转换为需要丢弃的元素数量
    indices_to_drop = random.sample(range(total_samples), num_to_drop)
    new_list = [data_list[i] for i in range(total_samples) if i not in indices_to_drop]
    return new_list


from concurrent.futures import ThreadPoolExecutor

def process_indices(batch, current_label):
    index_positive_tobesample = []
    index_negative_tobesample = []
    for cur_index in batch:
        ################## 处理positive 和 negative的边 ################################
        if current_label[cur_index[0]] == current_label[cur_index[1]]:
            index_positive_tobesample.append(cur_index)
        else:
            index_negative_tobesample.append(cur_index)
    
    return index_positive_tobesample, index_negative_tobesample

def split_list(a_list, chunks):
    """分割列表为指定的块数"""
    k, m = divmod(len(a_list), chunks)
    return [a_list[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(chunks)]

def parallel_process_indices(indexes, current_label, num_threads=100):
    # 将索引列表分割为多个批次
    batches = split_list(indexes, num_threads)

    # 使用线程池处理每个批次
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(process_indices, batch, current_label) for batch in batches]
        results_tobesample = []
        results_processed = []
        # 收集结果
        for future in futures:
            tobesample, processed = future.result()
            results_tobesample.extend(tobesample)
            results_processed.extend(processed)
    return results_tobesample, results_processed


