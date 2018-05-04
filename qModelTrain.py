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
import qmodelDefs as qm
from qfunc import qModelTrainVal
# need trained seq2seqModel
from arguments import qMargs
from qmodelDefs import *
from tempseq2seq import *

def main(args):
    # raise errors by checking if proper datafiles assigned. eg. A2B needs 3 file and qlen needs 2 file
    DS = torch.load(args.datafile)
    args.vsz = DS.vsz
    args.svsz = DS.svsz
    if args.debug:
        args.bsz=2
        DS.train = DS.train[:2]

    if args.resume:
        QM,Qoptimizer = torch.load(args.qresume)
        if args.qfntype == qRVAE:
            QM.enc.flatten_parameters()
            QM.dec.flatten_parameters()
        e = args.resume.split("/")[-1] if "/" in args.resume else args.resume
        e = e.split('_')[0]
        e = int(e)+1
    else:
        if args.qfntype == "qlen":
            QM = qm.QlenModel(args).cuda()
            Qoptimizer = torch.optim.Adam(QM.parameters(), lr=0.001)
        elif args.qfntype == "qA2BM":
            QM = qm.QA2BModel(args).cuda()
            Qoptimizer = torch.optim.Adadelta(QAM.parameters(), lr=.5)
        elif args.qfntype == "qRVAE":
            QM = qm.RVAEModel(args).cuda()
            Qoptimizer = torch.optim.Adam(QAM.parameters(), lr=.001)
        else:
            raise("unknown qfunctype")
        e=0

    print(QM)
    print(args.qdatafile)
    print(args.qsavestr)    
    print("training qModel")
    trainer = qModelTrainVal(args)
    trainer.qModelTrainer(QM, Qoptimizer, DS)
    
if __name__=="__main__":
    args = qMargs()
    main(args)
  