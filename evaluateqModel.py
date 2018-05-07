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
from s2s_bland import *
from qfunc import qFunc
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu

## basically validate function from s2s_bland.py

def testQmodel(args,epoch=0):
    DS = torch.load(args.datafile)
    args.vsz = DS.vsz
    args.svsz = DS.svsz
    print(args.valid)
    print("source_vocab_size",DS.svsz)
    print("target_vocab_size",DS.vsz)
    QM, _ = torch.load(args.qMtoeval)
    if args.qfntype == "qRVAE" or args.qfntype == "qMMI" :
        QM.enc.flatten_parameters()
        QM.dec.flatten_parameters()
        QM.endtok = DS.vocab.index("<eos>")
        QM.punct = [DS.vocab.index(t) for t in ['.','!','?'] if t in DS.vocab]
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
    print("loaded models")
    print(M)
    print(QM)
    for sources, targets in data:
        curpos+=1
        if curpos%50==0:
            print("processed :",curpos,"valsamples")
        sources = Variable(sources.cuda(),volatile=True)
        #logits = M(sources,None)
        #logits = torch.max(logits.data.cpu(),2)[1]
        #logits = [list(x) for x in logits]
        done, donescores = beam.beamsearch(sources, M, QM, len(targets[0]))
        
        if args.rerankqfunc:
            qf = qFunc(args)
            donescores = qf.rerankqfunc(QM, M, sources, done, donescores) #donescores returned after rescoring
        
        topscore =  donescores.index(max(donescores))
        logits = done[topscore]
        try:
            hyp = [DS.vocab[x] if x < len(DS.vocab) else DS.vocab[2] for x in logits]
        except:
            print(logits)
            print("somthing's wrong")
            asd
        hyps.append(hyp)
        refs.append(targets)
    bleu = corpus_bleu(refs,hyps,emulate_multibleu=True,smoothing_function=cc.method3)
    if args.scoreqfunc or args.rerankqfunc:
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
