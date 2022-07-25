import torch
from torch.nn import Parameter
from torch.nn import Module,Sequential,Linear,ReLU,Dropout,LogSoftmax
from torch.nn.modules.module import Module
from torch import nn
import torch.nn.functional as F
import math

class RGCLayer(Module):
    def __init__(self,input_dim,h_dim,num_predcicates,num_base,featureless,drop_prob, devices):
        super(RGCLayer,self).__init__()
        self.num_base=num_base
        self.input_dim=input_dim
        self.h_dim=h_dim
        self.num_predcicates = num_predcicates
        self.featureless=featureless
        self.drop_prob=drop_prob
        self.devices = devices
        if num_base>0:
            self.W=Parameter(torch.empty(input_dim*self.num_base,h_dim,dtype=torch.float32))
            self.W_comp=Parameter(torch.empty(num_predcicates,num_base,dtype=torch.float32))
        else:
            self.W=Parameter(torch.empty(input_dim*num_predcicates,h_dim,dtype=torch.float32))
        self.B=Parameter(torch.FloatTensor(h_dim))
        self.reset_parameters()
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W)
        if self.num_base>0:
            nn.init.xavier_uniform_(self.W_comp)
        self.B.data.fill_(0)
    def forward(self,vertex,A):
        supports=[]
        nodes_num=A[0].shape[0]
        for i,adj in enumerate(A):
            if not self.featureless:
                supports.append(torch.mm(adj,vertex))
            else:
                supports.append(adj)
        supports=torch.cat(supports,dim=1)
        if self.num_base>0:
            V=torch.matmul(self.W_comp,torch.reshape(self.W,(self.num_base,self.input_dim,self.h_dim)).permute(1,0,2))
            V=torch.reshape(V,(self.input_dim*self.num_predcicates,self.h_dim))
            output=torch.mm(supports,V)
        else:
            output=torch.mm(supports,self.W)
        if self.featureless:
            temp=torch.ones(nodes_num).to(self.devices)
            temp_drop=F.dropout(temp,self.drop_prob)
            output=(output.transpose(1,0)*temp_drop).transpose(1,0)
        output+=self.B
        return output

class RGCN(Module):
    def __init__(self,i_dim,h_dim,dropout,num_rels,num_bases,featureless=True,devices=0):
        super(RGCN,self).__init__()
        self.drop_prob=dropout
        self.gc1=RGCLayer(i_dim,h_dim,num_rels,num_bases,featureless,dropout,devices=devices)
        self.gc2=RGCLayer(h_dim,h_dim,num_rels,num_bases,False,dropout,devices=devices)

    def forward(self,vertex,A):
        gc1=F.dropout(F.relu(self.gc1(vertex,A)),self.drop_prob)
        gc2=F.dropout(F.relu(self.gc2(gc1,A)),self.drop_prob)

        return gc2

class RPGCLayer(Module):
    def __init__(self,input_dim,h_dim, num_objects,num_base,featureless,drop_prob, devices):
        super(RPGCLayer,self).__init__()
        self.num_base=num_base
        self.input_dim=input_dim
        self.h_dim=h_dim
        self.num_objects = num_objects
        self.featureless=featureless
        self.drop_prob=drop_prob
        self.devices = devices
        if num_base>0:
            self.W=Parameter(torch.empty(input_dim*self.num_base,h_dim,dtype=torch.float32))
            self.W_comp=Parameter(torch.empty(num_objects,num_base,dtype=torch.float32))
        else:
            self.W=Parameter(torch.empty(input_dim*num_objects,h_dim,dtype=torch.float32))
        self.B=Parameter(torch.FloatTensor(h_dim))
        self.reset_parameters()
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W)
        if self.num_base>0:
            nn.init.xavier_uniform_(self.W_comp)
        self.B.data.fill_(0)
    def forward(self,vertex,A):
        nodes_num=A[0].shape[0]
        supports = torch.matmul(A, vertex)
        supports = supports.reshape(-1, self.input_dim*self.num_objects)
        if self.num_base>0:
            V=torch.matmul(self.W_comp,torch.reshape(self.W,(self.num_base,self.input_dim,self.h_dim)).permute(1,0,2))
            V=torch.reshape(V,(self.input_dim*self.num_objects,self.h_dim))
            output=torch.mm(supports,V)
        else:
            output=torch.mm(supports,self.W)
        if self.featureless:
            temp=torch.ones(nodes_num).to(self.devices)
            temp_drop=F.dropout(temp,self.drop_prob)
            output=(output.transpose(1,0)*temp_drop).transpose(1,0)
        output+=self.B
        return output

class RPGCN(Module):
    def __init__(self,i_dim,h_dim,dropout, num_rels, num_objs,num_bases,featureless=True,devices=0):
        super(RPGCN,self).__init__()
        self.drop_prob=dropout
        self.gc1=RGCLayer(i_dim,i_dim,num_rels,num_bases,featureless,dropout,devices=devices)
        self.gc2=RPGCLayer(i_dim,h_dim,num_objs,num_bases,False,dropout,devices=devices)

    def forward(self,vertex,A):
        gc1=F.dropout(F.relu(self.gc1(vertex,A)),self.drop_prob)
        gc2=F.dropout(F.relu(self.gc2(gc1,A)),self.drop_prob)

        return gc2