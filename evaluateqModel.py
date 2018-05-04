import sys
import os
import argparse
import torch
from itertools import product
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
from preprocess_src2_inclusive import load_data3
from preprocess_new import load_data
from Beam import Beam
from arguments import qMargs
from qmodelDefs import *
from tempseq2seq import *

## basically validate function from s2s_bland.py

def testQmodel(args,epoch=0):
    DS = torch.load(args.datafile)
    args.vsz = DS.vsz
    args.svsz = DS.svsz
    print(args.valid)
    QM, _ = torch.load(args.qevaluate)
    M, _ = torch.load(args.fwdseq2seqModel)
    M.enc.flatten_parameters()
    M.dec.flatten_parameters()
    M.endtok = DS.vocab.index("<eos>")
    M.punct = [DS.vocab.index(t) for t in ['.','!','?'] if t in DS.vocab]
    QM.eval()
    M.eval()
    beam = Beam(args)
    data = DS.new_data(args.valid)
    cc = SmoothingFunction()
    refs = []
    hyps = []
    curpos = 0
    goldL=50
    for sources, targets in data:
        curpos+=1
        if curpos%50==0:
            print("processed :",curpos,"valsamples")
        sources = Variable(sources.cuda(),volatile=True)
        #logits = M(sources,None)
        #logits = torch.max(logits.data.cpu(),2)[1]
        #logits = [list(x) for x in logits]
        done, donescores = beam.beamsearch(sources, M, QM,len(targets[0]))
        topscore =  donescores.index(max(donescores))
        logits = done[topscore]
        hyp = [DS.vocab[x] for x in logits]
        hyps.append(hyp)
        refs.append(targets)
    bleu = corpus_bleu(refs,hyps,emulate_multibleu=True,smoothing_function=cc.method3)
    if args.scoreqfunc:
        sv=args.savestr+"hyps_"+args.qfntype
    else:
        sv=args.savestr+"hyps_"+"seq2seq"
    print("saving in",sv)
    with open(sv,'w') as f:
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
   
if __name__ == "__main__":
    args = qMargs()
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
    bleu = testQmodel(args)
    print("bleu is", bleu)
