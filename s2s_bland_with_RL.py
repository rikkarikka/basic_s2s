
# coding: utf-8

# In[ ]:


import sys
import os
import argparse
import torch
from itertools import product
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu, sentence_bleu
from preprocess_new import load_data
from arguments import s2s_bland as parseParams
import numpy as np
import math
import torch.distributions as dt


# In[ ]:


class model(nn.Module):
  def __init__(self,args):
    super().__init__()
    self.args = args
    
    # encoder decoder stuff
    self.encemb = nn.Embedding(args.svsz,args.hsz,padding_idx=0)
    self.enc = nn.LSTM(args.hsz,args.hsz//2,bidirectional=True,num_layers=args.layers,batch_first=True)
    self.decemb = nn.Embedding(args.vsz,args.hsz,padding_idx=0)
    self.dec = nn.LSTM(args.hsz*2,args.hsz,num_layers=args.layers,batch_first=True)
    self.gen = nn.Linear(args.hsz,args.vsz)

    # attention stuff
    self.linin = nn.Linear(args.hsz,args.hsz,bias=False)
    self.sm = nn.Softmax(dim=-1)
    self.linout = nn.Linear(args.hsz*2,args.hsz,bias=False)
    self.tanh = nn.Tanh()
    self.drop = nn.Dropout(args.drop)
    self.beamsize = args.beamsize

  def beamsearch(self,inp,trainRL=0,out=None, val=False, anneal=0):
    ##inp = inp.unsqueeze(0)
    #encenc = self.encemb(inp)
    #enc,(h,c) = self.enc(encenc)
    #print(inp)
    #print(enc)
    
    ### edited for RL
    sampled_word_proba = Variable(torch.FloatTensor(1,1))
    sampled_word = Variable(torch.LongTensor(1,1))
    tempouts, enc, h, c, hlist, clist = self.forward(inp, out, val)
    tempsofts = self.sm(tempouts)
    windex = math.floor(tempsofts.size()[1]/2 - anneal)
    if windex <= 1:
        windex = 1
    #enc hidden has bidirection so switch those to the features dim
    h = [torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2) for i in range(self.beamsize)]
    c = [torch.cat([c[0:c.size(0):2], c[1:c.size(0):2]], 2) for i in range(self.beamsize)]
    encbeam = [enc for i in range(self.beamsize)]

    ops = [Variable(torch.cuda.FloatTensor(1,1,self.args.hsz).zero_()) for i in range(self.beamsize)]
    ##set prev here according to a condition for RL
    if trainRL and (windex > 0):
        h = [hlist[windex-1] for i in range(self.beamsize)] # choose proper h and c index
        c = [clist[windex-1] for i in range(self.beamsize)] 
        vals, idxs = torch.topk(tempsofts[0][windex],1)
        m = dt.Categorical(tempsofts[0][windex])
        t1 = m.sample()
        sampled_word = t1.view(1,1)
        sampled_word_proba = m.log_prob(t1)
        t1 = t1.view(1,1)
        prev = [t1 for i in range(self.beamsize)]
    else:
        prev = [Variable(torch.cuda.LongTensor(1,1).fill_(3)) for i in range(self.beamsize)]
        
    ### done edited for RL
    beam = [[] for x in range(self.beamsize)]
    scores = [0]*self.beamsize
    sents = [0]*self.beamsize
    done = []
    donescores = []
    for i in range(self.args.maxlen):
      tmp = []
      for j in range(len(beam)):
        dembedding = self.decemb(prev[j].view(1,1))
        decin = torch.cat((dembedding.squeeze(1),ops[j].squeeze(1)),1).unsqueeze(1)
        decout, (hx,cx) = self.dec(decin,(h[j],c[j]))

        #attend on enc 
        q = self.linin(decout.squeeze(1)).unsqueeze(2)
        #q = decout.view(decout.size(0),decout.size(2),decout.size(1))
        w = torch.bmm(enc,q).squeeze(2)
        w = self.sm(w)
        cc = torch.bmm(w.unsqueeze(1),enc)
        op = torch.cat((cc,decout),2)
        op = self.drop(self.tanh(self.linout(op)))
      
        op2 = self.gen(op)
        op2 = op2.squeeze()
        probs = F.log_softmax(op2,dim=-1)
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
            tmp.append((vals[k].data[0]+scores[j],pidx[k],j,hx,cx,op))
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
        if pdat in self.punct:
          newsents.append(sents[beamidx]+1)
        else:
          newsents.append(sents[beamidx])
        if pdat == self.endtok or newsents[-1]>4:
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
    ### edited for RL
    if (not trainRL) or val:
        return done[topscore]
    else:
        return tempsofts, done, donescores, sampled_word_proba, sampled_word, windex
    ### done edited for RL
    
  def encode(self, inp):
    encenc = self.encemb(inp)
    enc,(h,c) = self.enc(encenc)
    return enc,(h,c)
      
  def forward(self, inp, out=None, val=False):
    encenc = self.encemb(inp)
    enc,(h,c) = self.enc(encenc)
    #enc hidden has bidirection so switch those to the features dim
    htemp = h
    ctemp = c
    h = torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2) 
    c = torch.cat([c[0:c.size(0):2], c[1:c.size(0):2]], 2) 
    #decode
    op = Variable(torch.cuda.FloatTensor(inp.size(0),self.args.hsz).zero_())
    outputs = []
    if out is None:
      outp = self.args.maxlen
    else:
      outp = out.size(1)
    hlist = []
    clist = []
    for i in range(outp): 
      if i == 0:
        prev = Variable(torch.cuda.LongTensor(inp.size(0),1).fill_(3))
      else:
        if out is None or val:
          prev = self.gen(op).max(2)
          prev = prev[1]
        else:
          prev = out[:,i-1].unsqueeze(1)
        op = op.squeeze(1)
      ## break if <eos> reached in input feeding technique
      if (out is None) and prev.data[0][0] == 1:
        break
      #print("start here")    
      #print(np.shape(prev))
      dembedding = self.decemb(prev)
      #print(np.shape(dembedding))
      #print(np.shape(op))
      decin = torch.cat((dembedding.squeeze(1),op),1).unsqueeze(1)
      decout, (h,c) = self.dec(decin,(h,c))
      #print(np.shape(decout))
      q = self.linin(decout.squeeze(1)).unsqueeze(2)
      #print(np.shape(q))
      #q = decout.view(decout.size(0),decout.size(2),decout.size(1))
      w = torch.bmm(enc,q).squeeze(2)
      #print(np.shape(w))
      w = self.sm(w)
      cc = torch.bmm(w.unsqueeze(1),enc)
      #print(np.shape(cc))
      op = torch.cat((cc,decout),2)
      #print(np.shape(op))
      op = self.drop(self.tanh(self.linout(op)))
      outputs.append(self.gen(op))
      ##output list of h and c from decoder for RL
      hlist.append(h)
      clist.append(c)
      #print(np.shape(outputs[0])) 
      
    outputs = torch.cat(outputs,1)
    #print(np.shape(outputs))
    return outputs, enc, htemp, ctemp, hlist, clist

def validate(M,DS,args):
  data = DS.new_data(args.valid)
  smth = SmoothingFunction()
  M.eval()
  refs = []
  hyps = []
  trainRL = 0
  print("here")
  j=0
  for sources,targets in data:
    sources = Variable(sources.cuda(),volatile=True)
    M.zero_grad()
    
    #logits = M(sources,None)
    #logits = torch.max(logits.data.cpu(),2)[1]
    #logits = [list(x) for x in logits]
    logits = M.beamsearch(sources, trainRL, None, True)
    hyp = [DS.vocab[x] for x in logits]
    print("hypothesis")
    print(hyp)
    #print("target")
    #print(targets)
    hyps.append(hyp)
    refs.append(targets)
  bleu = corpus_bleu(refs,hyps,emulate_multibleu=True,smoothing_function=smth.method3)
  M.train()
  print(bleu)
  with open(args.savestr+"hyps"+args.epoch,'w') as f:
    hyps = [' '.join(x) for x in hyps]
    f.write('\n'.join(hyps))
  try:
    os.stat(args.savestr+"refs")
  except:
    with open(args.savestr+"refs",'w') as f:
      refstr = []
      for r in refs:
        r = [' '.join(x) for x in r]
        refstr.append('\n'.join(r))
      f.write('\n'.join(refstr))
  return bleu

def train(M,DS,args,optimizer):
  weights = torch.cuda.FloatTensor(args.vsz).fill_(1)
  weights[0] = 0
  criterion = nn.CrossEntropyLoss(weights)
  trainloss = []
  M.train()
  while True:
    x = DS.get_batch()
    if not x:
      break
    sources,targets = x
    sources = Variable(sources.cuda())
    targets = Variable(targets.cuda())
    M.zero_grad()
    logits, _, _, _, _, _ = M(sources,targets)
    logits = logits.view(-1,logits.size(2))
    targets = targets.view(-1)
    loss = criterion(logits, targets)
    loss.backward()
    trainloss.append(loss.data.cpu()[0])
    optimizer.step()

    if len(trainloss)%100==99: print(trainloss[-1])
  return sum(trainloss)/len(trainloss)

def get_rewards(DS, idxs, sampled_word, target, beam_outs, windex,smth):
    
    logits_greedy = [] # part of sentence before 
    for i in range(windex):
        logits_greedy.extend([idxs[i][0]])
    logits = []
    for beams in beam_outs:
        logits.append(logits_greedy + [sampled_word.data.cpu()[0][0]] + beams[:])
    hyps = [[DS.vocab[j] for j in x if j<len(DS.vocab) and j>0] for x in logits]
    print(hyps)
    bleu = 0
    for hyp in hyps:
        bleu += sentence_bleu(target,hyp,emulate_multibleu=True,smoothing_function=smth.method3)
    return bleu/len(hyps)

def trainRL(M,DS,args,optimizer):
    data = DS.new_data(args.train)
    data = data[0:math.floor(len(data)/50)]
    smth = SmoothingFunction()
    M.train()
    refs = []
    hyps = []
    trainRL = 1
    anneal=-1
    for epoch in range(50):
        if epoch%5==0:
            anneal+=1
        greedyop = []
        tloss = 0
        treward = 0
        j=0
        print("epoch")
        print(epoch)
        for sources,targets in data:
            baseline = Variable(torch.cuda.FloatTensor(1).fill_(0), requires_grad=False)
            j=j+1
            M.zero_grad()
            sources = Variable(sources.cuda())
            greedyproba, beamouts, beamscores, sampled_word_proba, sampled_word, windex = M.beamsearch(sources,trainRL, None, False, int(anneal))
            probas = torch.log(greedyproba+eps)
            vals,idxs = torch.topk(probas,1)
            idxs = idxs.squeeze(0).data.cpu()
            greedyop = []
            for i in range(idxs.size()[0]):
                greedyop.extend([idxs[i][0]])
            hyp = [DS.vocab[x] for x in greedyop]
            targets[0].extend([DS.vocab[1]])
            vals = vals.squeeze(0)
            reward = get_rewards(DS, idxs, sampled_word, targets, beamouts, windex,smth)
            i=0
            baseline = sentence_bleu(targets,hyp,emulate_multibleu=True,smoothing_function=smth.method3)
            vals_greedy = sampled_word_proba #torch.sum(vals.squeeze(0)[:windex]) + sampled_word_proba + bsc# part of sentence before 
            loss = 1 * vals_greedy * (reward - baseline)
            loss.backward()
            optimizer.step()
            tloss+=loss.data[0]
            treward+=reward
        print(treward/len(data))
        print(tloss/len(data))


# In[ ]:


def main(args):
  DS = torch.load(args.datafile)
  if args.debug:
    args.bsz=2
    DS.train = DS.train[:2]

  args.vsz = DS.vsz
  args.svsz = DS.svsz
  if args.resume:
    M,optimizer = torch.load(args.resume)
    M.enc.flatten_parameters()
    M.dec.flatten_parameters()
    e = args.resume.split("/")[-1] if "/" in args.resume else args.resume
    e = e.split('_')[0]
    e = int(e)+1
  else:
    M = model(args).cuda()
    optimizer = torch.optim.Adam(M.parameters(), lr=args.lr)
    e=0
  M.endtok = DS.vocab.index("<eos>")
  M.punct = [DS.vocab.index(t) for t in ['.','!','?'] if t in DS.vocab]
  print(M)
  print(args.datafile)
  print(args.savestr)
  for epoch in range(e,args.epochs):
    args.epoch = str(epoch)
    trainloss = train(M,DS,args,optimizer)
    print("train loss epoch",epoch,trainloss)
    b = validate(M,DS,args)
    print("valid bleu ",b)
    torch.save((M,optimizer),args.savestr+args.epoch+"_bleu-"+str(b))
  ## train RL
  trainRL(M,data,args,optimizer)
    
if __name__=="__main__":
  args = parseParams()
  main(args)
  

