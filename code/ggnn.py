import torch
import torch.nn as nn
from constants import Constants
from torch.autograd import Variable
import numpy as np

class AttrProxy(object):
    """
    Translates index lookups into attribute lookups.
    To implement some trick which able to use list of nn.Module in a nn.Module
    see https://discuss.pytorch.org/t/list-of-nn-module-in-a-nn-module/219/2
    """
    def __init__(self, module, prefix):
        self.module = module
        self.prefix = prefix

    def __getitem__(self, i):
        return getattr(self.module, self.prefix + str(i))


class Propogator(nn.Module):
    """
    Gated Propogator for GGNN
    Using LSTM gating mechanism
    """
    def __init__(self, state_dim, n_node, n_edge_types):
        super(Propogator, self).__init__()

        self.n_node = n_node
        self.n_edge_types = n_edge_types

        self.reset_gate = nn.Sequential(
            nn.Linear(state_dim*3, state_dim),
            nn.Sigmoid()
        )
        self.update_gate = nn.Sequential(
            nn.Linear(state_dim*3, state_dim),
            nn.Sigmoid()
        )
        self.tansform = nn.Sequential(
            nn.Linear(state_dim*3, state_dim),
            nn.Tanh()
        )

    def forward(self, state_in, state_out, state_cur, A):
        A_in = A[:, :, :self.n_node*self.n_edge_types]
        A_out = A[:, :, self.n_node*self.n_edge_types:]
    
        # print("a_in "+str(A_in.dtype))
        # print("state_in "+str(state_in.dtype))
        a_in = torch.bmm(A_in, state_in)
        a_out = torch.bmm(A_out, state_out)
        a = torch.cat((a_in, a_out, state_cur), 2)

        r = self.reset_gate(a)
        z = self.update_gate(a)
        joined_input = torch.cat((a_in, a_out, r * state_cur), 2)
        h_hat = self.tansform(joined_input)

        output = (1 - z) * state_cur + z * h_hat

        return output


class GGNN(nn.Module):
    """
    Gated Graph Sequence Neural Networks (GGNN)
    Mode: SelectNode
    Implementation based on https://arxiv.org/abs/1511.05493
    """
    def __init__(self):
        super(GGNN, self).__init__()

        # assert (opt.state_dim >= opt.annotation_dim,  \
        #         'state_dim must be no less than annotation_dim')

        # self.state_dim = opt.state_dim # hidden_dim
        # self.annotation_dim = opt.annotation_dim # feature_dim
        # self.n_edge_types = opt.n_edge_types # max edges num
        # self.n_node = opt.n_node # max node num
        # self.n_steps = opt.n_steps # steps num
        
        self.state_dim = Constants.GGNN_STATE_DIM
        self.annotation_dim = Constants.GGNN_ANNOTATION_DIM
        self.n_edge_types = Constants.MAX_EDGE_NUM
        self.n_node = Constants.MAX_NODE_NUM
        self.n_steps = Constants.GGNN_STEP_NUM

        for i in range(self.n_edge_types):
            # incoming and outgoing edge embedding
            in_fc = nn.Linear(self.state_dim, self.state_dim)
            out_fc = nn.Linear(self.state_dim, self.state_dim)
            self.add_module("in_{}".format(i), in_fc)
            self.add_module("out_{}".format(i), out_fc)

        self.in_fcs = AttrProxy(self, "in_")
        self.out_fcs = AttrProxy(self, "out_")

        # Propogation Model
        self.propogator = Propogator(self.state_dim, self.n_node, self.n_edge_types)

        # Output Model
        self.out = nn.Sequential(
            nn.Linear(self.state_dim + self.annotation_dim, self.state_dim),
            nn.Tanh(),
            nn.Linear(self.state_dim, 5)
        )

        self._initialization()

    def _initialization(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.02)
                m.bias.data.fill_(0)

    def forward(self, prop_state, annotation, A):
        for i_step in range(self.n_steps):
            in_states = []
            out_states = []
            for i in range(self.n_edge_types):
                in_states.append(self.in_fcs[i](prop_state))
                out_states.append(self.out_fcs[i](prop_state))
            in_states = torch.stack(in_states).transpose(0, 1).contiguous()
            in_states = in_states.view(-1, self.n_node*self.n_edge_types, self.state_dim)
            out_states = torch.stack(out_states).transpose(0, 1).contiguous()
            out_states = out_states.view(-1, self.n_node*self.n_edge_types, self.state_dim)

            prop_state = self.propogator(in_states, out_states, prop_state, A)

        join_state = torch.cat((prop_state, annotation), 2)

        output = self.out(join_state)
        # print("output")
        # print(output.shape)
        return output.mean(axis=1)


def create_adjacency_matrix(edges, n_nodes, n_edge_types):
    a = np.zeros([n_nodes, n_nodes * n_edge_types * 2])
    for edge in edges:
        src_idx = edge[0]
        tgt_idx = edge[1]
        a[tgt_idx][src_idx ] =  1
        a[src_idx][n_edge_types* n_nodes + tgt_idx] =  1
    return a

    # init_input, annotation, adj_matrix
    #   padding = torch.zeros(len(annotation), opt.n_node, opt.state_dim - opt.annotation_dim).double()
    #   init_input = torch.cat((annotation, padding), 2)
if(__name__=="__main__"):
     model = GGNN()
     annotation = torch.tensor([[1,1,1],[2,2,2],[3,2,3],[0,0,0],[0,0,0],[1,1,1],[2,2,2],[3,2,3],[0,0,0],[0,0,0]])
     annotation = annotation.view(2,5,3)
     edges = np.array([(0,0),(0,1),(1,1),(1,2),(2,2)])
     adj_matrix = create_adjacency_matrix(edges,Constants.MAX_NODE_NUM,Constants.MAX_EDGE_NUM)
     adj_matrix = torch.tensor([adj_matrix,adj_matrix]).view(2,Constants.MAX_NODE_NUM,Constants.MAX_NODE_NUM*Constants.MAX_EDGE_NUM*2).float()
     padding = torch.zeros(2, Constants.MAX_NODE_NUM, Constants.GGNN_STATE_DIM - Constants.GGNN_ANNOTATION_DIM)
     print(annotation.shape)
     print(adj_matrix.shape)
     print(padding.shape)
     init_input = torch.cat((annotation, padding), 2)
     adj_matrix = Variable(adj_matrix)
     annotation = Variable(annotation)
     init_input = Variable(init_input)
     print(init_input.dtype)
     print(annotation.dtype)
     print(adj_matrix.dtype)
     print(model(init_input,annotation,adj_matrix))