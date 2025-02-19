import torch
import torch.nn as nn
from  constants import Constants 
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

class ResidualBlock_1(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock_1, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class ResNet18_1(nn.Module):
    def __init__(self):
        super(ResNet18_1, self).__init__()
        self.conv1 = nn.Conv2d(Constants.CNN_INPUT_CHANNELS,Constants.CNN_COV_OUTPUT_CHANNELS, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(Constants.CNN_COV_OUTPUT_CHANNELS)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(Constants.CNN_COV_OUTPUT_CHANNELS, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2)
        self.layer3 = self._make_layer(128, 256, 2)
        self.layer4 = self._make_layer(256, 512, 2)
        self.maxpool = nn.AdaptiveMaxPool2d((1, 1))
        self.fc = nn.Linear(512, Constants.CNN_OUTPUT_DIM_CFG)

    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = []
        layers.append(ResidualBlock_1(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(ResidualBlock_1(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.maxpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

class ResidualBlock_2(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock_2, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class ResNet18_2(nn.Module):
    def __init__(self):
        super(ResNet18_2, self).__init__()
        self.conv1 = nn.Conv2d(Constants.CNN_INPUT_CHANNELS,Constants.CNN_COV_OUTPUT_CHANNELS, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(Constants.CNN_COV_OUTPUT_CHANNELS)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(Constants.CNN_COV_OUTPUT_CHANNELS, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2)
        self.layer3 = self._make_layer(128, 256, 2)
        self.layer4 = self._make_layer(256, 512, 2)
        self.maxpool = nn.AdaptiveMaxPool2d((1, 1))
        self.fc = nn.Linear(512, Constants.CNN_OUTPUT_DIM_CG)

    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = []
        layers.append(ResidualBlock_2(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(ResidualBlock_2(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.maxpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x



def truncate_and_pad(input_tensor, max_length):
    """
    Truncate or pad a 1xN tensor to 1xmax_length tensor.
    
    Args:
    input_tensor (torch.Tensor): Input tensor of size 1xN.
    max_length (int): Maximum length for truncation or padding.
    
    Returns:
    torch.Tensor: 1xmax_length tensor with truncated or padded values.
    """
    input_length = input_tensor.size(1)
    if input_length >= max_length:
        # If input length is greater than or equal to max_length, truncate
        output_tensor = input_tensor[:, :max_length]
    else:
        # If input length is less than max_length, pad with zeros
        pad_length = max_length - input_length
        padding = torch.zeros((1, pad_length), dtype=input_tensor.dtype, device=input_tensor.device)
        output_tensor = torch.cat((input_tensor, padding), dim=1)
    return output_tensor


class SimpleCNN(nn.Module):
    def __init__(self,cnn_input_channels,cnn_output_channels):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(cnn_input_channels, cnn_output_channels, kernel_size=3, stride=1, padding=1)
        self.fc = nn.Linear(Constants.CNN_COV_MAX_OUTPUT,Constants.CNN_OUTPUT_DIM).to(Constants.DEVICE)
       # self.fc = nn.Linear(16*N*N, 10)  # 假设输出维度为10

    def forward(self, x):
        x = torch.relu(self.conv1(x)).to(Constants.DEVICE)
        x = x.view(x.size(0), -1).to(Constants.DEVICE)  # 展平
        x = truncate_and_pad(x,Constants.CNN_COV_MAX_OUTPUT).to(Constants.DEVICE)
        x = self.fc(x)
        return x


class BinaryClassificationMLP(nn.Module):
    def __init__(self, input_size):
        super(BinaryClassificationMLP, self).__init__()
        hidden_size = Constants.MLP_HIDDEN_SIZE
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, 2)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.output(x)
        x = torch.softmax(x,dim=0)
        return x


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
            nn.Linear(self.state_dim, Constants.GGNN_OUTPUT_DIM)
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
        output = output.sum(1).squeeze(0)
        # print("output")
        # print(output.shape)
        return output


def expand_tensor(tensor):
    if tensor.shape == (1, 1):
        expanded_tensor = torch.zeros((2, 2))
        expanded_tensor[0, 0] = tensor[0, 0]
        return expanded_tensor
    else:
        return tensor
    
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        
        self.ggnn = GGNN()

        self.cnn1  = ResNet18_1()
        self.cnn2  = ResNet18_2()

        self.mlp  =  BinaryClassificationMLP(Constants.GGNN_OUTPUT_DIM + Constants.CNN_OUTPUT_DIM_CFG + Constants.CNN_OUTPUT_DIM_CG)
    # def forward(self, graph):
    #     # graph_data包含图节点特征和边索引
    #     x = torch.tensor(graph.node_features).squeeze()
    #     edge_index = torch.tensor(graph.edge_index)
    #     feature1 = self.mpnn(x, edge_index)

    #     feature2 = self.cnn(torch.tensor(graph.matrix).unsqueeze(0).unsqueeze(0).float())

    #     feature1 = torch.mean(feature1,dim=0)
    #     feature2 = torch.mean(feature2,dim=0)

    #     # 拼接MPNN和CNN的输出
    #     combined_features = torch.cat((feature1, feature2), dim=0)

    #     # 通过MLP得到最终输出
    #     output = self.mlp(combined_features)
    #     return output
    
    def forward(self, func):
        cfg = func.cfg
        cgs = func.cgs

        ## cfg cgs[0]
        graph1 = cfg
        # graph2 = cgs[0]
        
        # features = torch.empty(2,Constants.GGNN_OUTPUT_DIM)
        annotations = torch.tensor(graph1.node_features).to(Constants.DEVICE)
        annotations = annotations.view(1,Constants.MAX_NODE_NUM ,Constants.WORD2VECTOR_DIM)
        adj_matrix = torch.tensor(graph1.adj_matrix).view(1,Constants.MAX_NODE_NUM,Constants.MAX_NODE_NUM*Constants.MAX_EDGE_NUM*2).to(Constants.DEVICE).float()
        padding = torch.zeros(len(annotations), Constants.MAX_NODE_NUM, Constants.GGNN_STATE_DIM - Constants.GGNN_ANNOTATION_DIM).to(Constants.DEVICE)
        init_input = torch.cat((annotations, padding), 2).to(Constants.DEVICE)
        adj_matrix = Variable(adj_matrix)
        annotations = Variable(annotations)
        init_input = Variable(init_input)
        feature1_1 = self.ggnn(init_input,annotations,adj_matrix)
        # features[0] = torch.relu(feature1_1).to(device)

        # x = torch.tensor(graph2.node_features).squeeze(1).to(device)
        # edge_index = torch.tensor(graph2.edge_index,dtype=int).to(device)
        # feature1_2 = self.mpnn2(x, edge_index)
        # feature1_2 = torch.mean(feature1_2,dim=0).to(device)
        # features[1] = torch.relu(feature1_2).to(device)

        # feature1 = torch.cat((feature1_1, feature1_2), dim=0).to(device)
        # feature1 = feature1_1
        # feature1_1 = 0
        # print(graph1.matrix)
        # print(graph1.matrix)


        adjmatrix2_1 = expand_tensor(torch.tensor(graph1.matrix)).unsqueeze(0).unsqueeze(0).float().to(Constants.DEVICE)
        feature2_1 = self.cnn1(adjmatrix2_1)
        feature2_1 = torch.mean(feature2_1,dim=0)

        adjmatrix2_2 = expand_tensor(torch.tensor(func.cg_matrix)).unsqueeze(0).unsqueeze(0).float().to(Constants.DEVICE)
        feature2_2 = self.cnn2(adjmatrix2_2)
        feature2_2 = torch.mean(feature2_2,dim=0)
        feature2 = torch.cat((feature2_1, feature2_2), dim=0).to(Constants.DEVICE)
        feature2_1 = 0
        feature2_2 = 0
        # feature2 = feature2_1
        # print(feature2)
        # print(feature2.size())

        # 拼接GGNN和CNN的输出
        # print(feature1)
        # print(feature1.shape)
        # print(feature2)
        # print(feature2.shape)
        combined_features = torch.cat((feature1_1, feature2), dim=0).to(Constants.DEVICE)
        feature1 = 0
        feature2 = 0
        #print(combined_features)
        # 通过MLP得到最终输出
        output = self.mlp(combined_features)
        return output

if __name__ == "__main__":
    matrix = [[0,0,1],
              [1,1,1],
              [1,0,0]]
    
    model = ResNet18_1()
    m = expand_tensor(torch.tensor([matrix,matrix])).unsqueeze(1).float()
    print(m.shape)

    print(model(m))