import sys
import torch
from collections import Counter
from arguments import s2s_bland as parseParams

class load_data3:
  def __init__(self,args):
    self.args = args
    
    train_sources,train_targets,train_src2 = self.ds(args.train)
    
    #train_targets = [x[0] for x in train_targets]
    ctr = Counter([x for z in train_targets for x in z])
    
    thresh = 1
    self.vocab = ["<pad>","<eos>","<unk>","<start>"]+[x for x in ctr if ctr[x]>thresh and x != "<unk>"]
    self.stoit = {x:i for i,x in enumerate(self.vocab)}
    self.vsz = len(self.vocab)
    
    ctr = Counter([x for z in train_sources for x in z])
    thresh = 1
    self.itos = ["<pad>","<eos>","<unk>","<start>"]+[x for x in ctr if ctr[x]>thresh and x != "<unk>"]
    self.stoi = {x:i for i,x in enumerate(self.itos)}
    self.svsz = len(self.itos)
    
    ctr = Counter([x for z in train_src2 for x in z])
    thresh = 1
    self.itos2 = ["<pad>","<eos>","<unk>","<start>"]+[x for x in ctr if ctr[x]>thresh and x != "<unk>"]
    self.stoi2 = {x:i for i,x in enumerate(self.itos2)}
    self.svsz2 = len(self.itos2)
    
    self.train = list(zip(train_sources,train_targets,train_src2))
    self.train.sort(key=lambda x: len(x[0]),reverse=True)
    self.bctr = 0
    self.bsz = args.bsz
    self.dsz = len(self.train)

  def get_batch(self):
    if self.bctr>=self.dsz:
      self.bctr = 0
      return None
    else:
      data = self.train
      siz = len(data[self.bctr][0])
      k = 0
      srcs,tgts,srcs2 = [],[],[]
      srclen, tgtlen, src2len = [],[],[]
      while k<self.bsz and self.bctr+k<self.dsz:
        src,tgt,src2 = data[self.bctr+k]
        if len(src)<siz:
          break
        srcs.append(src)
        tgts.append(tgt)
        srcs2.append(src2)
        srclen.append(len(src))
        tgtlen.append(len(tgt))
        src2len.append(len(src2))
        k+=1
      self.bctr+=k
    return self.pad_batch((srcs,tgts,srcs2)), srclen, tgtlen, src2len

  def new_data(self,fn,targ=False,src2=False):
    src,tgt,src2 = self.ds(fn)
    new = []
    for i in range(len(src)):
        new.append(self.pad_batch(([src[i]],tgt[i],src2[i]),targ=targ,src2=src2))
    return new

  def val_data(self, fn, targ=True, src2=True):
    src,tgt,src2 = self.ds(fn)

    data = list(zip(src,tgt,src2))
    data.sort(key=lambda x:len(x[0]),reverse=True)
    batches = self.batches(data)
    batches = [self.pad_batch(batch) for batch in batches]
    return batches

  def pad_batch(self,batch,targ=True,src2=True):
    srcs, tgts, srcs2 = batch
    targs = tgts
    sorcs2 = srcs2
   
    srcnums = [[self.stoi[w] if w in self.stoi else 2 for w in x]+[1] for x in srcs]
    m = max([len(x) for x in srcnums])
    srcnums = [x+([0]*(m-len(x))) for x in srcnums]
    tensor = torch.cuda.LongTensor(srcnums)
    
    if targ:
      targtmp = [[self.vocab.index(w) if w in self.vocab else 2 for w in x]+[1] for x in tgts]
      m = max([len(x) for x in targtmp])
      targtmp = [x+([0]*(m-len(x))) for x in targtmp]
      targs = torch.cuda.LongTensor(targtmp)
        
    if src2:
      src2tmp = [[self.stoi2[w] if w in self.stoi2 else 2 for w in x]+[1] for x in srcs2]
      m = max([len(x) for x in src2tmp])
      src2tmp = [x+([0]*(m-len(x))) for x in src2tmp]
      sorcs2 = torch.cuda.LongTensor(src2tmp)
        
    return (tensor,targs,sorcs2)

  def mkbatches(self,bsz):
    self.bsz = bsz
    self.train_batches = self.batches(self.train)
    self.val_batches = self.batches(self.val)

  def batches(self,data):
    ctr = 0
    batches = []
    while ctr<len(data):
      siz = len(data[ctr][0])
      k = 0
      srcs,tgts, srcs2 = [], [], []
      while k<self.bsz and ctr+k<len(data):
        src,tgt,src2 = data[ctr+k]
        if len(src)<siz:
          break
        srcs.append(src)
        tgts.append(tgt)
        srcs2.append(src2)
        k+=1
      ctr+=k
      batches.append((srcs,tgts,srcs2))
    return batches
        
  def ds(self,fn):
    with open(fn) as f:
      sources, targs, sources2 = zip(*[x.strip().split("\t") for x in f.readlines()])
    sources = [x.split(" ") for x in sources]
    
    targets = []
    for t in targs:
      t = t.replace("PERSON","<person>").replace("LOCATION","<location>").lower()
      targets.append(t.split(" "))
    
    src2 = []
    for t in sources2:
      t = t.replace("PERSON","<person>").replace("LOCATION","<location>").lower()
      src2.append(t.split(" "))
    
    return sources, targets, src2

if __name__=="__main__":
  args = parseParams()
  DS = load_data3(args)
  torch.save(DS,args.datafile)
