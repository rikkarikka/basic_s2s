import sys
import os
import argparse
import torch
from itertools import product
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
from qfunc import qFunc

class Beam():
  def __init__(self,args):
    super().__init__()
    self.args = args
    self.beamsize = args.beamsize

  def search(self,M,qfunc):
    pass

  #def beamsearch(self,inp, M, qfunc, scorewholevocab=None):
  def beamsearch(self,inp, M, QM=None, goldL=None):
    #inp = inp.unsqueeze(0)
    encenc = M.encemb(inp)
    enc,(h,c) = M.enc(encenc)

    #enc hidden has bidirection so switch those to the features dim
    h = [torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2) for i in range(self.beamsize)]
    c = [torch.cat([c[0:c.size(0):2], c[1:c.size(0):2]], 2) for i in range(self.beamsize)]
    encbeam = [enc for i in range(self.beamsize)]

    ops = [Variable(torch.cuda.FloatTensor(1,1,self.args.hsz).zero_()) for i in range(self.beamsize)]
    prev = [Variable(torch.cuda.LongTensor(1,1).fill_(3)) for i in range(self.beamsize)]
    beam = [[] for x in range(self.beamsize)]
    scores = [0]*self.beamsize
    sents = [0]*self.beamsize
    done = []
    donescores = []
    for i in range(self.args.maxlen):
      tmp = []
      for j in range(len(beam)):
        op, (hx,cx) = M.decode_step(prev[j], ops[j].squeeze(1), enc, h[j], c[j])
        op2 = M.gen(op)
        op2 = op2.squeeze()
        probs = F.log_softmax(op2)
        vals = 0
        pidx = 0
        if self.args.scoreqfunc: ## and sampling condition aswell?
            if not QM:
                raise("QModel not defined")
            qf = qFunc(self.args)
            vals, pidx = qf.scoreqfunc(QM, M, beam, probs, goldL, op, hx, cx)
        else:
            asd
            vals, pidx = probs.topk(self.beamsize*2,0)
        vals = vals.squeeze()
        pidx = pidx.squeeze()
        donectr = 0
        k = -1
        while donectr<self.beamsize:
          k+=1
          pdat = pidx[k].data[0]
          if pdat == 2:
            continue
          else:
            #if not scorewholevocab and scoreqfunc:
            #    ??
            #    tmp.append((vals[k].data[0]+scores[j],pidx[k],j,hx,cx,op))
            #    donectr+=1
            #else:
            tmp.append((vals[k].data[0],pidx[k],j,hx,cx,op))
            donectr+=1
        #for k in range(self.beamsize):
        #  tmp.append((vals[k].data[0]+scores[j],pidx[k],j,hx,cx,op))
      tmp.sort(key=lambda x: x[0],reverse=True)
      newbeam = []
      newscore = []
      newsents = []
      newh = []
      newc = []
      newops = []
      newprev = []
      added = 0
      j = 0
      while added < len(beam):
        v,pidx,beamidx,hx,cx,op = tmp[j]
        pdat = pidx.data[0]
        new = beam[beamidx]+[pdat]
        if pdat in M.punct:
          newsents.append(sents[beamidx]+1)
        else:
          newsents.append(sents[beamidx])
        if pdat == M.endtok or newsents[-1]>4:
          if new not in done:
            done.append(new)
            donescores.append(v)
            added += 1
        else:
          if new not in newbeam:
            newbeam.append(new)
            newscore.append(v)
            newh.append(hx)
            newc.append(cx)
            newops.append(op)
            newprev.append(pidx)
            added += 1
        j+=1
      beam = newbeam 
      prev = newprev
      scores = newscore
      sents = newsents
      h = newh
      c = newc
      ops = newops
    if len(done)==0:
      done.extend(beam)
      donescores.extend(scores)
    donescores = [x/len(done[i]) for i,x in enumerate(donescores)]
    topscore =  donescores.index(max(donescores))
    return done, donescores#[topscore]
    