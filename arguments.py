import os
import argparse
def s2bool(v):
  if v.lower()=='false':
    return False
  else:
    return True

def general():
    parser = argparse.ArgumentParser(description='none')
    # learning
    parser.add_argument('-layers', type=int, default=2, help='min_freq for vocab [default: 1]') #
    parser.add_argument('-drop', type=float, default=0.3, help='min_freq for vocab [default: 1]') #
    parser.add_argument('-bsz', type=int, default=128, help='min_freq for vocab [default: 1]') #
    parser.add_argument('-hsz', type=int, default=500, help='min_freq for vocab [default: 1]') #
    parser.add_argument('-maxlen', type=int, default=50, help='min_freq for vocab [default: 1]') #
    parser.add_argument('-lr', type=float, default=0.001, help='initial learning rate [default: 0.001]') #
    parser.add_argument('-epochs', type=int, default=30, help='number of epochs for train [default: 100]') #
    parser.add_argument('-debug', action="store_true")
    parser.add_argument('-resume', type=str, default=None)
    parser.add_argument('-train',type=str,default="data/train.txt")
    parser.add_argument('-valid',type=str,default="data/valid.txt")
    parser.add_argument('-cuda',type=s2bool,default=True)
    parser.add_argument('-gpu',type=str,default="0")
    parser.add_argument('-burninepoch', type=int,default="10",help='num epochs before decaying considered')
    parser.add_argument('-decayrate', type=int,default="2",help='decay lr by lr/decayrate')
    parser.add_argument('-minlr', type=float,default="1e-6")
    return parser

def mkdir(args,qMsave=0):
  try:
    os.stat(args.savestr)
  except:
    os.mkdir(args.savestr)
  if qMsave:
    try:
      os.stat(args.saveqMstr)  
    except:
      os.mkdir(args.saveqMstr)

def s2s_bland():
    parser = general()
    parser.add_argument('-datafile', type=str, default="data_opensub/bland.pt")
    parser.add_argument('-savestr',type=str,default="saved_models/bland_opensub/")
    parser.add_argument('-beamsize', type=int, default=4, help='min_freq for vocab [default: 1]') #
    parser.add_argument('-vmodel',type=str, default=None)
    parser.add_argument('-modelname',type=str,default="fwd_nounk")
    args = parser.parse_args()
    mkdir(args)
    return args

def qMargs():
    parser = general()
    parser.add_argument('-datafile', type=str, default="data_opensub/opensub_fwd.pt")
    parser.add_argument('-savestr',type=str,default="saved_models/bland_opensub/")
    parser.add_argument('-beamsize', type=int, default=4, help='min_freq for vocab [default: 1]') #
    parser.add_argument('-vmodel',type=str, default=None)    
    parser.add_argument('-qfntype', type=str,default="qlen") # options are qlen, qA2B, qRVAE future - qCRVAE, qAdver
    parser.add_argument('-fwdseq2seqModel', type=str,default="saved_models/bland_opensub/30_bleu-0.0917")
    parser.add_argument('-bwdseq2seqModel',type=str,default="saved_models/bland_opensub/30_bw_bleu-0.0884")
    ## if qfunc requires some other type of model, load it within func
    parser.add_argument('-saveqMstr', type=str,default="saved_models/bland_opensub/")
    parser.add_argument('-qepochs', type=int, default=31)
    parser.add_argument('-qresume', type=str, default="saved_models/bland_opensub/30QM")
    parser.add_argument('-qMtoeval', type=str, default="saved_models/bland_opensub/30QM")
    parser.add_argument('-scorewholevocab', type=s2bool, default=False)
    parser.add_argument('-scoreqfunc', type=s2bool, default=False)
    parser.add_argument('-rerankqfunc',type=s2bool, default=False)
    parser.add_argument('-qfnstoscore', type=str, nargs='*',default=["qlen"])
    parser.add_argument('-qlr', type=float, default=0.001, help='initial learning rate [default: 0.001]')
    args = parser.parse_args()
    mkdir(args,1)
    return args
