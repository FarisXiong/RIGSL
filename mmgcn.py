
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.nn import RGCNConv, GraphConv
from torch.nn.parameter import Parameter
import numpy as np, itertools, random, copy, math
import math
import scipy.sparse as sp
# import ipdb
 
class GCNII_lyc(nn.Module):
    def __init__(self, nfeat, nlayers, nhidden, nclass, dropout, lamda, alpha, variant, return_feature, use_residue, new_graph=False):
        super(GCNII_lyc, self).__init__()
        self.return_feature = return_feature
        self.use_residue = use_residue
        self.new_graph = new_graph
        self.convs = nn.ModuleList()
        for _ in range(nlayers):
            self.convs.append(GraphConvolution(nhidden, nhidden,variant=variant))
        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(nfeat, nhidden))
        if not return_feature:
            self.fcs.append(nn.Linear(nfeat+nhidden, nclass))
        self.params1 = list(self.convs.parameters())
        self.params2 = list(self.fcs.parameters())
        self.act_fn = nn.ReLU()
        self.dropout = dropout
        self.alpha = alpha
        self.lamda = lamda

    def cossim(self, x, y):
        a = torch.matmul(x, y)
        b = torch.sqrt(torch.matmul(x, x)) * torch.sqrt(torch.matmul(y, y))
        if b == 0:
            return 0
        else:
            return (a / b)

    def forward(self, x, dia_len, topicLabel, adj=None):
        if adj is None:
            if self.new_graph:
                adj = self.message_passing_relation_graph(x, dia_len)
            else:
                adj = self.message_passing_wo_speaker(x, dia_len, topicLabel)
        else:
            adj = adj
        _layers = []
        x = F.dropout(x, self.dropout, training=self.training)
        layer_inner = self.act_fn(self.fcs[0](x))
        _layers.append(layer_inner)
        for i,con in enumerate(self.convs):
            layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
            layer_inner = self.act_fn(con(layer_inner,adj,_layers[0],self.lamda,self.alpha,i+1))
        layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
        if self.use_residue:
            layer_inner = torch.cat([x, layer_inner], dim=-1)
        if not self.return_feature:
            layer_inner = self.fcs[-1](layer_inner)
            layer_inner = F.log_softmax(layer_inner, dim=1)
        return layer_inner

    def message_passing_wo_speaker(self, x,dia_len, topicLabel):
        adj = torch.zeros((x.shape[0], x.shape[0]))
        start = 0
        for i in range(len(dia_len)):
            sub_adj = torch.zeros((dia_len[i], dia_len[i]))
            temp = x[start:start+dia_len[i]]
            vec_length = torch.sqrt(torch.sum(temp.mul(temp), dim=1))
            norm_temp = (temp.permute(1, 0) / vec_length) 
            cos_sim_matrix = torch.sum(torch.matmul(norm_temp.unsqueeze(2), norm_temp.unsqueeze(1)), dim=0)
            cos_sim_matrix = cos_sim_matrix * 0.99999
            sim_matrix = torch.acos(cos_sim_matrix)

            d = sim_matrix.sum(1)
            D = torch.diag(torch.pow(d, -0.5))

            sub_adj[:dia_len[i], :dia_len[i]] = D.mm(sim_matrix).mm(D)
            adj[start:start+dia_len[i], start:start+dia_len[i]] = sub_adj
            start+=dia_len[i]

        adj = adj.cuda()

        return adj

    def atom_calculate_edge_weight(self, x, y):
        f = self.cossim(x, y)
        if f >1 and f <1.05:
            f = 1
        elif f< -1 and f>-1.05:
            f = -1
        elif f>=1.05 or f<=-1.05:
            print('cos = {}'.format(f))
        return f

    def message_passing_directed_speaker(self, x, dia_len, qmask):
        total_len = sum(dia_len)
        adj = torch.zeros((total_len, total_len))+torch.eye(total_len)
        start = 0
        use_utterance_edge=False
        for (i, len_) in enumerate(dia_len):
            speaker0 = []
            speaker1 = []
            for (j, speaker) in enumerate(qmask[i][0:len_]):
                if speaker[0] == 1:
                    speaker0.append(j)
                else:
                    speaker1.append(j)
            if use_utterance_edge:
                for j in range(len_-1):
                    f = self.atom_calculate_edge_weight(x[start+j], x[start+j+1])
                    Aij = 1-math.acos(f) / math.pi
                    adj[start+j][start+j+1] = Aij
                    adj[start+j+1][start+j] = Aij
            for k in range(len(speaker0)-1):
                f = self.atom_calculate_edge_weight(x[start+speaker0[k]], x[start+speaker0[k+1]])
                Aij = 1-math.acos(f) / math.pi
                adj[start+speaker0[k]][start+speaker0[k+1]] = Aij
                adj[start+speaker0[k+1]][start+speaker0[k]] = Aij
            for k in range(len(speaker1)-1):
                f = self.atom_calculate_edge_weight(x[start+speaker1[k]], x[start+speaker1[k+1]])
                Aij = 1-math.acos(f) / math.pi
                adj[start+speaker1[k]][start+speaker1[k+1]] = Aij
                adj[start+speaker1[k+1]][start+speaker1[k]] = Aij

            start+=dia_len[i]

        d = adj.sum(1)
        D=torch.diag(torch.pow(d,-0.5))
        adj = D.mm(adj).mm(D).cuda()
        
        return adj.cuda()

    def message_passing_relation_graph(self, x, dia_len):
        total_len = sum(dia_len)
        adj = torch.zeros((total_len, total_len))+torch.eye(total_len)
        window_size = 10
        start = 0
        for (i, len_) in enumerate(dia_len):
            edge_set = []
            for k in range(len_):
                left = max(0, k-window_size)
                right = min(len_-1, k+window_size)
                edge_set = edge_set + [str(i)+'_'+str(j) for i in range(left, right) for j in range(i+1, right+1)]
            edge_set = [[start+int(str_.split('_')[0]),start+int(str_.split('_')[1])] for str_ in list(set(edge_set))]
            for left, right in edge_set:
                f = self.atom_calculate_edge_weight(x[left], x[right])
                Aij = 1-math.acos(f) / math.pi
                adj[left][right] = Aij
                adj[right][left] = Aij
            start+=dia_len[i]

        d = adj.sum(1)
        D=torch.diag(torch.pow(d,-0.5))
        adj = D.mm(adj).mm(D).cuda()
        
        return adj.cuda()








class GraphConvolution(nn.Module):

    def __init__(self, in_features, out_features, residual=False, variant=False,  gamma=-0.95, zeta=1.05, temperature=2.0):
        super(GraphConvolution, self).__init__() 
        self.variant = variant
        if self.variant:
            self.in_features = 2*in_features 
        else:
            self.in_features = in_features

        self.out_features = out_features
        self.residual = residual
        self.weight = Parameter(torch.FloatTensor(self.in_features,self.out_features))
        self.reset_parameters()

        self.gamma = gamma
        self.zeta = zeta
        self.temperature = temperature
        self.transform_before = nn.Linear(200, 200)
        self.transform_after = nn.Linear(200, 200)
        self.attention_intra = nn.Linear( 2 * 200, 1)
        self.attention_inter = nn.Linear( 2 * 200, 1)
        self.attention_crossmodal = nn.Linear( 2 * 200, 1)

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_features)
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input, adj , h0 , lamda, alpha, l):
        theta = math.log(lamda/l+1)
        hi = torch.spmm(adj, input)
        if self.variant:
            support = torch.cat([hi,h0],1)
            r = (1-alpha)*hi+alpha*h0
        else:
            support = (1-alpha)*hi+alpha*h0
            r = support
        output = theta*torch.mm(support, self.weight)+(1-theta)*r
        if self.residual:
            output = output+input
        return output



class MM_GCN(nn.Module):
    def __init__(self, a_dim, v_dim, l_dim, n_dim, nlayers, nhidden, nclass, dropout, lamda, alpha, variant, return_feature, use_residue, new_graph='full',n_speakers=2, modals=['a','v','l'], use_speaker=True, use_modal=False):
        super(MM_GCN, self).__init__()
        self.return_feature = return_feature
        self.use_residue = use_residue
        self.new_graph = new_graph

        self.graph_net = GCNII_lyc(nfeat=n_dim, nlayers=nlayers, nhidden=nhidden, nclass=nclass,
                               dropout=dropout, lamda=lamda, alpha=alpha, variant=variant,
                               return_feature=return_feature, use_residue=use_residue)
        self.a_fc = nn.Linear(a_dim, n_dim)
        self.v_fc = nn.Linear(v_dim, n_dim)
        self.l_fc = nn.Linear(l_dim, n_dim)
        if self.use_residue:
            self.feature_fc = nn.Linear(n_dim*3+nhidden*3, nhidden)
        else:
            self.feature_fc = nn.Linear(nhidden * 3, nhidden)
        self.final_fc = nn.Linear(nhidden, nclass)
        self.act_fn = nn.ReLU()
        self.dropout = dropout
        self.alpha = alpha
        self.lamda = lamda
        self.modals = modals
        self.modal_embeddings = nn.Embedding(3, n_dim)
        self.speaker_embeddings = nn.Embedding(n_speakers, n_dim)
        self.a_spk_embs = nn.Embedding(n_speakers, n_dim)
        self.v_spk_embs = nn.Embedding(n_speakers, n_dim)
        self.l_spk_embs = nn.Embedding(n_speakers, n_dim)
        self.use_speaker = use_speaker
        self.use_modal = use_modal

    def forward(self, a, v, l, dia_len, qmask):
        a = self.a_fc(a)
        v = self.v_fc(v)
        l = self.l_fc(l)
        qmask = torch.cat([qmask[:x,i,:] for i,x in enumerate(dia_len)],dim=0)
        spk_idx = torch.argmax(qmask, dim=-1)
        spk_emb_vector = self.speaker_embeddings(spk_idx)
        if self.use_speaker:
            if 'l' in self.modals:
                l += spk_emb_vector
        if self.use_modal:
            emb_idx = torch.LongTensor([0, 1, 2]).cuda()
            emb_vector = self.modal_embeddings(emb_idx)

            if 'a' in self.modals:
                a += emb_vector[0].reshape(1, -1).expand(a.shape[0], a.shape[1])
            if 'v' in self.modals:
                v += emb_vector[1].reshape(1, -1).expand(v.shape[0], v.shape[1])
            if 'l' in self.modals:
                l += emb_vector[2].reshape(1, -1).expand(l.shape[0], l.shape[1])

        adj = self.create_big_adj(a, v, l, dia_len, self.modals)

        if len(self.modals) == 3:
            features = torch.cat([a, v, l], dim=0).cuda()
        elif 'a' in self.modals and 'v' in self. modals:
            features = torch.cat([a, v], dim=0).cuda()
        elif 'a' in self.modals and 'l' in self.modals:
            features = torch.cat([a, l], dim=0).cuda()
        elif 'v' in self.modals and 'l' in self.modals:
            features = torch.cat([v, l], dim=0).cuda()
        else:
            return NotImplementedError
        features = self.graph_net(features, None, qmask, adj)
        all_length = l.shape[0] if len(l)!=0 else a.shape[0] if len(a) != 0 else v.shape[0]
        # if len(self.modals) == 3:
        #     features = torch.cat([features[:all_length], features[all_length:all_length * 2], features[all_length * 2:all_length * 3]], dim=-1)
        # else:
        #     features = torch.cat([features[:all_length], features[all_length:all_length * 2]], dim=-1)

        return (features[all_length * 2:all_length * 3], features[:all_length], features[all_length:all_length * 2]), 1


    def create_big_adj(self, a, v, l, dia_len, modals):
        modal_num = len(modals)
        all_length = l.shape[0] if len(l)!=0 else a.shape[0] if len(a) != 0 else v.shape[0]
        adj = torch.zeros((modal_num*all_length, modal_num*all_length)).cuda()
        if len(modals) == 3:
            features = [a, v, l]
        elif 'a' in modals and 'v' in modals:
            features = [a, v]
        elif 'a' in modals and 'l' in modals:
            features = [a, l]
        elif 'v' in modals and 'l' in modals:
            features = [v, l]
        else:
            return NotImplementedError
        start = 0
        for i in range(len(dia_len)):
            sub_adjs = []
            for j, x in enumerate(features):
                if j < 0:
                    sub_adj = torch.zeros((dia_len[i], dia_len[i])) + torch.eye(dia_len[i])
                else:
                    sub_adj = torch.zeros((dia_len[i], dia_len[i]))
                    temp = x[start:start + dia_len[i]]
                    vec_length = torch.sqrt(torch.sum(temp.mul(temp), dim=1))
                    norm_temp = (temp.permute(1, 0) / vec_length)
                    cos_sim_matrix = torch.sum(torch.matmul(norm_temp.unsqueeze(2), norm_temp.unsqueeze(1)), dim=0)  # seq, seq
                    cos_sim_matrix = cos_sim_matrix * 0.99999
                    sim_matrix = 1 - torch.acos(cos_sim_matrix)/np.pi
                    sub_adj[:dia_len[i], :dia_len[i]] = sim_matrix
                sub_adjs.append(sub_adj)
            dia_idx = np.array(np.diag_indices(dia_len[i]))
            for m in range(modal_num):
                for n in range(modal_num):
                    m_start = start + all_length*m
                    n_start = start + all_length*n
                    if m == n:
                        adj[m_start:m_start+dia_len[i], n_start:n_start+dia_len[i]] = sub_adjs[m]
                    else:
                        modal1 = features[m][start:start+dia_len[i]] #length, dim
                        modal2 = features[n][start:start+dia_len[i]]
                        normed_modal1 = modal1.permute(1, 0) / torch.sqrt(torch.sum(modal1.mul(modal1), dim=1)) #dim, length
                        normed_modal2 = modal2.permute(1, 0) / torch.sqrt(torch.sum(modal2.mul(modal2), dim=1)) #dim, length
                        dia_cos_sim = torch.sum(normed_modal1.mul(normed_modal2).permute(1, 0), dim=1) #length
                        dia_cos_sim = dia_cos_sim * 0.99999
                        dia_sim = 1 - torch.acos(dia_cos_sim)/np.pi
                        idx =dia_idx.copy()
                        idx[0] += m_start
                        idx[1] += n_start
                        adj[idx] = dia_sim

            start += dia_len[i]
        d = adj.sum(1)
        D = torch.diag(torch.pow(d, -0.5))
        adj = D.mm(adj).mm(D)

        return adj

