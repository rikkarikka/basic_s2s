import sys
import os
import argparse
import torch
from itertools import product
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable

class QA2BModel(nn.Module):
    def __init__(self,args):
        super().__init__()
        self.linin = nn.Linear(args.hsz*3,args.hsz*3,bias=False)
        self.lin = nn.Linear(args.hsz*3,args.hsz*3,bias=False)
        self.sm = nn.Softmax(dim=-1)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.drop = nn.Dropout(args.drop)
    
        #Qlen regr stuff 
        self.regrout = nn.Linear(args.hsz*3,1,bias=False)
    
    def forward(self, inp):
      #input is new hidden output value for current input
      inter = (self.linin(inp))
      inter = (self.lin(inter))
      inter = (self.lin(inter))
      pred = (self.regrout(inter))
      return pred

    
class QlenModel(nn.Module):
    def __init__(self,args):
        super().__init__()
        self.linin = nn.Linear(args.hsz*2,args.hsz*3,bias=False)
        self.lin = nn.Linear(args.hsz*3,args.hsz*3,bias=False)
        self.sm = nn.Softmax(dim=-1)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.drop = nn.Dropout(0.2)
        self.sp = nn.Softplus()
        #Qlen regr stuff 
        self.regrout = nn.Linear(args.hsz,1,bias=False)
        self.clfout = nn.Linear(args.hsz*3,args.maxlen,bias=False)
    
    def forward(self, inp):
      #input is new hidden output value for current input
      inter = self.drop(self.tanh(self.linin(inp)))
      inter = self.drop(self.tanh(self.lin(inter)))
      inter = self.drop(self.tanh(self.lin(inter)))
      inter = self.drop(self.tanh(self.lin(inter)))
      #inter = self.drop(self.lin(inter))
        
      #pred = self.relu(self.regrout(inter))#self.relu
      pred = self.clfout(inter)
      return pred

class RVAEModel(nn.Module):
    def __init__(self,args):
        super().__init__()
        # encoder decoder stuff
        self.args = args
        self.encemb = nn.Embedding(args.svsz,args.hsz,padding_idx=0)
        self.enc = nn.GRU(args.hsz,args.hsz//2,bidirectional=True,num_layers=args.layers,batch_first=True)
        self.decemb = nn.Embedding(args.vsz,args.hsz,padding_idx=0)
        self.dec = nn.GRU(args.hsz,args.hsz,num_layers=args.layers,batch_first=True)
        self.gen = nn.Linear(args.hsz,args.vsz)
        self.to_mu = nn.Linear(args.hsz,args.hsz)
        self.to_logvar = nn.Linear(args.hsz,args.hsz)

        self.linin = nn.Linear(args.hsz,args.hsz,bias=False)
        self.sm = nn.Softmax(dim=-1)
        self.linout = nn.Linear(args.hsz,args.hsz,bias=False)
        self.tanh = nn.Tanh()
        self.drop = nn.Dropout(args.drop)

    def initHidden(self):
        return torch.cuda.FloatTensor(1, 1, self.args.hsz).fill_(0)
    
    def encoder(self,inp):
        h = self.initHidden()
        encenc = self.encemb(inp)
        enc, h = self.enc(encenc)

        #enc hidden has bidirection so switch those to the features dim
        #print("h size before reshape", h.size())
        h = torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2) 
        return enc, h
    
    def decode_step(self, prev, h):
        dembedding = self.decemb(prev)
        #print("dembedding size")
        #print(dembedding.size())
        #print(h.size())
        #asd
        decout, h = self.dec(dembedding, h)
        op = self.drop(self.tanh(self.linout(decout)))
        return op, h
      
    def decoder(self, inp, h, out=None, val=False): #h is encoded context
        outputs = []
        if out is None:
            seqlen = self.args.maxlen
        else:
            seqlen = out.size(1)
        for i in range(seqlen): 
            if i == 0:
                prev = Variable(torch.cuda.LongTensor(inp.size(0),1).fill_(3))
            else:
                if out is None or val:
                    prev = self.gen(op).max(2)
                    prev = prev[1]
                else:
                    prev = out[:,i-1].unsqueeze(1)
            op, h = self.decode_step(prev, h)
        
            outputs.append(self.gen(op))
        return outputs, h

    def forward(self, inp, out=None, val=False):
        #print(inp.size())
        enc, h = self.encoder(inp)
        #print("encoded enc and h")
        #print(enc.size())
        #print(h.size())
        mu = self.to_mu(h)    
        logvar = self.to_logvar(h)    
        std = torch.exp(0.5 * logvar)
        #print("mu and logvar",mu.size(),logvar.size())
        
        # reparametrization
        z = Variable(torch.randn([inp.size(0), self.args.hsz]))
        #if use_cuda:
        z = z.cuda()
        z = z * std + mu

        kld = (-0.5 * torch.sum(logvar - torch.pow(mu, 2) - torch.exp(logvar) + 1, 1)).mean().squeeze()
        #print("z size",z.size())
        #print("outsize",out.size())
        outputs, final_state = self.decoder(inp, z, out, val)
        #print(len(outputs))
        outputs = torch.cat(outputs,1)
        return outputs, final_state, kld    
        
    
    def getKLdiv(self,inp):
        enc, h = self.encoder(inp, self.args)
        
        mu = self.to_mu(h)    
        logvar = self.to_logvar(context)    
        std = t.exp(0.5 * logvar)

        # reparametrization
        z = Variable(torch.randn([inp.size(0), self.args.hsz]))
        #if use_cuda:
        z = z.cuda()
        z = z * std + mu

        kld = (-0.5 * torch.sum(logvar - torch.pow(mu, 2) - torch.exp(logvar) + 1, 1)).mean().squeeze()
        
        return kld