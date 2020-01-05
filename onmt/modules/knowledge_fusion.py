# encoding: utf-8
import math
import torch
import torch.nn as nn
from torch.nn import Parameter


class Fusion(nn.Module):
    '''
    Fuse knowledge into history encoded state
    '''

    def __init__(self, source_size, knowledge_size, hidden_size,batch_first=True, bias=True):
        super(Fusion, self).__init__()
        self.source_size = source_size
        self.knowledge_size = knowledge_size
        self.hidden_size = hidden_size
        self.batch_first = batch_first
        self.bias = bias

        # input gate
        self.i_w_x = Parameter(torch.Tensor(self.hidden_size, self.source_size))
        self.i_w_k = Parameter(torch.Tensor(self.hidden_size, self.knowledge_size))
        if self.bias:
            self.i_b = Parameter(torch.Tensor(self.hidden_size, 1))    # 暂不使用

        # source forget gate
        self.f_w_x = Parameter(torch.Tensor(self.hidden_size, self.source_size))
        if self.bias:
            self.f_b_x = Parameter(torch.Tensor(self.hidden_size, 1 ))

        # knowledge forget gate
        self.f_w_k = Parameter(torch.Tensor(self.hidden_size, self.knowledge_size))
        if self.bias:
            self.f_b_k = Parameter(torch.Tensor(self.hidden_size, 1))
        
        # output gate
        self.o_w_x = Parameter(torch.Tensor(self.hidden_size, self.hidden_size))
        self.o_w_k = Parameter(torch.Tensor(self.hidden_size, self.hidden_size))
        if self.bias:
            self.o_b = Parameter(torch.Tensor(self.hidden_size,1))
        

        # update state
        self.u_w_x = Parameter(torch.Tensor(self.hidden_size, self.source_size))
        self.u_w_k = Parameter(torch.Tensor(self.hidden_size, self.knowledge_size))
        if self.bias:
            self.u_b = Parameter(torch.Tensor(self.hidden_size,1))
        
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        
        self.init_weights()

    def init_weights(self):

        '''
        init weigths
        :return:
        '''

        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            nn.init.uniform_(weight, -stdv, stdv)

    def forward(self, source: torch.Tensor, knowledge: torch.Tensor):
        '''
        fusion forward
        :param source: batch_size * self.source_size    tuple for lstm
        :param knowledge: batch_size * self.knowledge_size     tuple for lstm
        :return: fused Tensor [batch* hidden_size], knowledge forget gate Tensro [hidden_size * 1]

        '''


        # if self.batch_first:
        #     source = source.transpose(0, 1)
        #     knowledge = knowledge.transpose(0, 1)

        label_to_restore = False
        if isinstance(source, tuple) and isinstance(knowledge,tuple):
            h_x, c_x = source
            h_k, c_k = source
            tuple_x_size = h_x.size()
            if h_x.dim() > 2:
                label_to_restore = True
                h_x = h_x.view(-1, self.hidden_size).t()
                c_x = c_x.view(-1, self.hidden_size).t()
                h_k = h_k.view(-1, self.hidden_size).t()
                c_k = c_k.view(-1, self.hidden_size).t()
        else:
            h_x = source
            h_k = knowledge
            if h_x.dim() > 2:
                label_to_restore = True
                h_x = h_x.view(-1, self.hidden_size).t()
                h_k = h_k.view(-1, self.hidden_size).t()

        assert h_x.dim() == 2 and h_k.dim() == 2, "Dim not equal to 2!"

        if self.bias:
            input_gate = self.sigmoid(self.i_w_x @ h_x + self.i_w_k @ h_k + self.i_b)

            forget_x_gate = self.sigmoid(torch.mm(self.f_w_x, h_x)+ self.f_b_x)
            forget_k_gate = self.sigmoid(torch.mm(self.f_w_k, h_k) + self.f_b_k)

            output_gate = self.sigmoid(torch.mm(self.o_w_x, h_x) + torch.mm(self.o_w_k, h_k) + self.o_b)
            if isinstance(source,tuple):
                state = self.tanh(torch.mm(self.u_w_x, c_x) + torch.mm(self.u_w_k, c_k) + self.u_b)
            else:
                state = self.tanh(torch.mm(self.u_w_x, h_x) + torch.mm(self.u_w_k, h_k) + self.u_b)
        else:
            input_gate = self.sigmoid(torch.mm(self.i_w_x, h_x) + torch.mm(self.i_w_k, h_k))
            
            forget_x_gate = self.sigmoid(torch.mm(self.f_w_x, h_x))
            forget_k_gate = self.sigmoid(torch.mm(self.f_w_k, h_k))
    
            output_gate = self.sigmoid(torch.mm(self.o_w_x, h_x) + torch.mm(self.o_w_k, h_k))
            if isinstance(source, tuple):
                state = self.tanh(torch.mm(self.u_w_x, c_x) + torch.mm(self.u_w_k, c_k))
            else:
                state = self.tanh(torch.mm(self.u_w_x, h_x) + torch.mm(self.u_w_k, h_k))
    
        out = input_gate * state + forget_k_gate * c_k + forget_x_gate * c_x
        final = output_gate * out

        if self.batch_first:
            final = final.t()
            forget_k_gate = forget_k_gate.t()
        # lstm
        if label_to_restore:
            final = final.view(tuple_x_size)
            out = out.view(tuple_x_size)
        else:
            final = final.view(source.size())
            out = out.view(source.size())
        if isinstance(source,tuple) and isinstance(knowledge, tuple):
            final = (final, out)

        return final, forget_k_gate
