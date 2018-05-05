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
from qmodelDefs import *
from s2s_bland import *

class qFunc():    
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.qfntype = args.qfntype
        
    def qlenfunc(self, QM, M, probs, goldL=None,op=None,hx=None,cx=None):
        if goldL is None:
            self.args.maxlen #?? or random number between a and b
        
        if self.args.scorewholevocab:
            print("wha..GTFO")
            raise NotImplementedError
        else:
            vals, pidx = probs.topk(self.args.beamsize*2,0)
            for k in range(len(pidx)):
                #*** need to finish this***
                #or can we do it here, pass pidx through decoder, get hnew, pass through QM
                dem = M.decemb(pidx[k].view(1,1))
                decin = torch.cat((dem.squeeze(1),op.squeeze(1)),1).unsqueeze(1)
                decout, (h1,c1) = M.dec(decin,(hx,cx))
                h1 = torch.cat([h1[0],h1[1]],dim=-1)
                predlen = QM(h1)
                predlen = F.log_softmax(predlen,dim=-1)
                val_pred = predlen.squeeze()#.topk(2) ,idx_pred
                temp_pred,_ = predlen.squeeze().topk(1)
                #for regre - -0.7*abs(goldL-predlen.data[0][0])**2
                lenscore = val_pred[goldL].data[0]
                vals[k] = vals[k] + 5*lenscore                
            return vals, pidx
    
    def qA2Bfunc(self, QM, M, probs,hx):
        if self.args.scorewholevocab:
            tmpwi=[] 
            for kk in range(probs.size(0)): # vocabsize? check in other notebook replace with probs.size(1)
                dem = M.decemb(Variable(torch.cuda.LongTensor(1).fill_(kk).view(1,1)))
                h1 = torch.cat([hx[0],hx[1]],dim=-1)
                inp = torch.cat([h1,dem.squeeze(0)],dim=-1)
                predsc = QM(inp)
                tmpwi.append(predsc)
            probs = probs + torch.stack(tmpwi,dim=-1).squeeze()*0.1
            vals, pidx = probs.topk(self.args.beamsize*2,0)
            return vals,pidx            
        else:
            vals, pidx = probs.topk(self.args.beamsize*2,0)
            for k in range(len(pidx)):
                dem = M.decemb(pidx[k].view(1,1))
                h1 = torch.cat([hx[0],hx[1]],dim=-1)
                inp = torch.cat([h1,dem.squeeze(0)],dim=-1)
                predsc = QM(inp)
                vals[k] = vals[k] - 0.1*predsc
            return vals,pidx           
    
    def qKLdivfunc(self, QM, beam, probs):
        if self.args.scorewholevocab:
            raise NotImplementedError
        else:
            vals, pidx = probs.topk(self.args.beamsize*2,0)
            for k in range(len(pidx)):
                if beam !=[]:
                    beam.append(pidx[k].data[0])
                    tempseq = Variable(torch.cuda.LongTensor(beam)) #evaluate KLdiv of each beam and predicted next word, choose word with lowest KL div 
                else:
                    tempseq = pidx[k]
                klscore = QM.getKLdiv(tempseq.unsqueeze(0))
                vals[k] = vals[k] + klscore
            return vals, pidx
    
    
    
    
    
    
    
    def qMMIfunc(self,M_MMI,M,Blist,MB_scores) # M_MMI means seqseq(A1|B)
        
    
    def scoreqfunc(self, QM, M, beam, probs, goldL=None, op=None, hx=None, cx=None):
        
        if self.args.scorewholevocab:
            raise NotImplementedError
        else:
            vals, pidx = probs.topk(self.args.beamsize*2,0)
            tempscore = 0
            for k in range(len(pidx)):
                if "qlen" in self.args.qfnstoscore:
                
                if "qA2B" in self.args.qfnstoscore:
                    
                if "qRVAE" in self.args.qfnstoscore:
                    
                       
        
        
        
        
        
        
        if self.qfntype == "qlen":
            vals,pidx = self.qlenfunc(QM,M,probs,goldL,op,hx,cx)
        elif self.qfntype == "qA2B":
            vals,pidx = self.qA2Bfunc(QM,M,probs,hx) 
        elif self.qfntype == "qRVAE":
            vals,pidx = self.qKLdivfunc(QM,beam,probs)
        else:
            raise ("unknown qtype:") #change this
        return vals,pidx
    
    def rerankqfunc(self,QM, M, inlist, scores):
        if self.qfntype ==

class qModelTrainVal():
    def __init__(self,args):
        super().__init__()
        self.args = args
        self.qfntype = args.qfntype
                
    def qlenfunc_trainIter(self, QM, M, DS, Qoptimizer):
        QM.train()
        M.eval()
        val=0
        #criterion = nn.MSELoss() # use this when regressing length, this option will be added later
        weights = torch.cuda.FloatTensor(self.args.maxlen).fill_(1)
        weights[0] = 0
        criterion = nn.CrossEntropyLoss(weights)
        trainloss = []
        
        while True:
            x= DS.get_batch()
            if not x:
                break
            (sources,targets),srclen,tgtlen = x
            aclen = [x+1 for x in tgtlen]
            tgtlen = torch.cuda.LongTensor(tgtlen).view(1,-1)+1
            sources = Variable(sources.cuda())
            targets = Variable(targets.cuda()) 
            batchlen = targets.size()[1]
            matrixtgtlen=[]
            matrixtgtlen.append(tgtlen)
            #print(targets.size())
            
            for i in range(batchlen-1):
                tgtlen=tgtlen-1
                for j in range(tgtlen.size(1)):
                    if tgtlen[0][j] < 0:
                        tgtlen[0][j] = 0
                matrixtgtlen.append(tgtlen)
            tgtlen = (torch.stack(matrixtgtlen,dim=-1))
            tgtlen = Variable(tgtlen.squeeze(0))
            Qoptimizer.zero_grad()
            enc, (h,c) = M.encode(sources)
            op = Variable(torch.cuda.FloatTensor(enc.size(0),self.args.hsz).zero_())
            op = op.squeeze(1)
            outputs = []
            
            if targets is None:
                outp = self.args.maxlen
            else:
                outp = targets.size(1)
                
            for i in range(outp): 
                if i == 0:
                    prev = Variable(torch.cuda.LongTensor(enc.size(0),1).fill_(3))
                else:
                    if targets is None or val:
                        prev = M.gen(op).max(2)
                        prev = prev[1]
                    else:
                        prev = targets[:,i-1].unsqueeze(1)
                    op = op.squeeze(1)
                op, (h,c) = M.decode_step(prev,op,enc,h,c)
                h = Variable(h.data)
                h1 = torch.cat([h[0],h[1]],dim=-1)
                preds = QM(h1)
                outputs.append(preds)
            outputs = torch.stack(outputs,dim=1)#.squeeze(1)
            #print(outputs)
            #print(F.log_softmax(outputs))
            #for i,acl in enumerate(aclen):
            #    for j in range(acl,batchlen):
            #        outputs[i,j] = Variable(torch.cuda.FloatTensor((50))).fill_(0)
            #        outputs[i,j,0] = 10.0
            #print(outputs)
            #print(tgtlen)
            #ZX
            loss = criterion(outputs.view(-1,self.args.maxlen),tgtlen.view(-1,))
            loss.backward()
            Qoptimizer.step()
            trainloss.append(loss.data.cpu()[0])
            
            if len(trainloss)%100==99: print(trainloss[-1])
        return sum(trainloss)/len(trainloss)
    
    def qlenfunc_validate(self,QM, M, DS):
        val = 1
        QM.eval()
        M.eval()
        val=0
        data = DS.new_data(self.args.valid)        
        for x,srclen,tgtlen in data:
            (sources, targets) = x
            aclen = [x+1 for x in tgtlen]
            tgtlen = torch.cuda.LongTensor(tgtlen).view(1,-1)+1
            sources = Variable(sources.cuda())
            targets = Variable(targets.cuda()) 

            batchlen = targets.size()[1]
            matrixtgtlen=[]
            matrixtgtlen.append(tgtlen)
            #print(targets.size())
            
            for i in range(batchlen-1):
                tgtlen=tgtlen-1
                for j in range(tgtlen.size(1)):
                    if tgtlen[0][j] < 0:
                        tgtlen[0][j] = 0
                matrixtgtlen.append(tgtlen)
            tgtlen = (torch.stack(matrixtgtlen,dim=-1))
            tgtlen = Variable(tgtlen.squeeze(0))

            enc, (h,c) = M.encode(sources)
            op = Variable(torch.cuda.FloatTensor(enc.size(0),args.hsz).zero_())
            op = op.squeeze(1)
            outputs = []

            if targets is None or val: # test Q func predictions both as val and train
                outp = args.maxlen
            else:
                outp = targets.size(1)

            for i in range(outp): 
                
                if i == 0:
                    prev = Variable(torch.cuda.LongTensor(enc.size(0),1).fill_(3))
                else:
                    if targets is None or val:
                        prev = M.gen(op).max(2)
                        prev = prev[1]
                    else:
                        prev = targets[:,i-1].unsqueeze(1)
                    op = op.squeeze(1)

                op, (h,c) = M.decode_step(prev,op,enc,h,c)
                h = Variable(h.data)
                h1 = torch.cat([h[0],h[1]],dim=-1)
                preds = QM(h1)
                outputs.append(preds)
            outputs = torch.stack(outputs,dim=1)#.squeeze(1)
            print(F.log_softmax(outputs))            
            for i in range(outputs.size(0)):
                tmp = outputs[i]
                print(tmp.squeeze())
                vals,idx = tmp.squeeze().topk(1)
                print(idx)
                print("acc",tgtlen[i])
            #not fully implemented, was testing if the model was correctly working.
            asd
            print(outputs.size())
            aasdasd
            
            for i in range(outputs.size(1)):
                val,idx = torch.topk(outputs[:,i].squeeze())
            print(outputs[:,i].squeeze().size())
            asd
            
            for i,acl in enumerate(aclen):
                for j in range(acl,batchlen):
                    outputs[i,j] = Variable(torch.cuda.FloatTensor((50))).fill_(0)
                    outputs[i,j,0] = 10.0
        return sum(trainloss)/len(trainloss)  
      
    def qA2Bfunc_trainIter(self, QA2BM, M_fwd, M_bwd, DS, Qoptimizer):
        QA2BM.train()
        M_fwd.eval()
        M_bwd.eval()
        val=0
        criterion = nn.MSELoss()
        #weights = torch.cuda.FloatTensor(args.maxlen).fill_(1)
        #weights[0] = 0
        #criterion = nn.CrossEntropyLoss(weights)
        trainloss = []
        while True:
            x= DS.get_batch()
            
            if not x:
                break
            (sources, targets, sources2),srclen,tgtlen, src2len = x
            sources = Variable(sources.cuda())
            targets = Variable(targets.cuda())
            sources2 = Variable(sources2.cuda())
            #QA2BM.zero_grad()
            outputs, h, prevs = M_fwd(sources,targets)
            preds_needed, _, _ = M_bwd(targets,sources2)
            #h1 = torch.cat([a[0],a[1]],dim=-1)
            #calc P(A2|B) over the batch
            targlogprobs = []
            
            for i in range(sources2.size(0)):
                temptarg = sources2[i]
                tempouts = preds_needed[i]
                tempouts = F.log_softmax(tempouts,dim=-1)
                var = Variable(torch.cuda.FloatTensor(1)).fill_(0)
                for j in range(sources2.size(1)):
                    var = var + tempouts[j,temptarg[j].data[0]]
                targlogprobs.append(var)
            targlogprobs = torch.stack(targlogprobs,dim=0) # should be of size batchsize*1
            targlogprobs = Variable(targlogprobs.data)
            
            for i in range(len(prevs)):
                htemp = h[i]
                prev = prevs[i]
                dembedding = M_fwd.decemb(prev)
                inp = torch.cat([htemp,dembedding.squeeze()],dim=-1)
                inp = Variable(inp.data)
                predprobas = QA2BM(inp) #output should be of size batchsize*1
                loss = criterion(predprobas.view(1,-1),targlogprobs.view(1,-1))
                loss.backward()
                #for param in QA2BM.parameters():
                #    print(param.grad.data.sum())
                Qoptimizer.step()
                Qoptimizer.zero_grad()
                trainloss.append(loss.data.cpu()[0])
            
            if len(trainloss)%100==99: print(trainloss[-1])
        return sum(trainloss)/len(trainloss)
    
    def qA2Bfunc_validate(self,):
        raise NotImplementedError

    def qKLdivfunc_trainIter(self,QM,DS,Qoptimizer):
        weights = torch.cuda.FloatTensor(self.args.vsz).fill_(1)
        weights[0] = 0
        criterion = nn.CrossEntropyLoss(weights)
        trainloss = []
        while True:
            x = DS.get_batch()
            if not x:
                break
            (sources,targets),srclen,tgtlen = x
            sources = Variable(sources.cuda())
            targets = Variable(targets.cuda())
            QM.zero_grad()
            logits,_,kld = QM(sources,targets)
            logits = logits.view(-1,logits.size(2))
            targets = targets.view(-1)
            loss = criterion(logits, targets) + kld # still need to optimize
            loss.backward()
            trainloss.append(loss.data.cpu()[0])
            optimizer.step()
            if len(trainloss)%100==99: print(trainloss[-1])
        return sum(trainloss)/len(trainloss)
    
    def qKLdivfunc_validate(self,):
        raise NotImplementedError
    
    def loadseq2seqModel(self, modelpath, DS):
        M,_ = torch.load(modelpath)
        M.enc.flatten_parameters()
        M.dec.flatten_parameters()
        M.endtok = DS.vocab.index("<eos>")
        M.punct = [DS.vocab.index(t) for t in ['.','!','?'] if t in DS.vocab]
        return M
    
    def qModelTrainer(self, QM, Qoptimizer, DS):        
        if self.qfntype == "qlen":
            M = self.loadseq2seqModel(self.args.fwdseq2seqModel,DS)
            for epoch in range(self.args.qepochs):
                trainloss = self.qlenfunc_trainIter(QM, M, DS, Qoptimizer)
                print("train loss epoch",epoch,trainloss)
                #qlenfunc_validate
                if epoch%5==0:
                    torch.save((QM,Qoptimizer),self.args.saveqMstr+str(epoch)+"_QlenM.pt")

        elif self.qfntype == "qA2B":
            M_fwd = self.loadseq2seqModel(self.args.fwdseq2seqModel, DS)
            M_bwd = self.loadseq2seqModel(self.args.bwdseq2seqModel, DS)
            for epoch in range(self.args.qepochs):
                trainloss = self.qA2Bfunc_trainIter(QM, M_fwd, M_bwd, DS, Qoptimizer)
                print("train loss epoch",epoch,trainloss)
                #qA2Bfunc_validate
                if epoch%5==0:
                    torch.save((QM,Qoptimizer),self.args.saveqMstr+str(epoch)+"_QA2BM.pt")
        
        elif self.qfntype == "qRVAE":
            for epoch in range(self.args.qepochs):
                trainloss = self.qKLdivfunc_trainIter(QM,DS,Qoptimizer)
                print("train loss epoch",epoch,trainloss)
                if epoch%5==0:
                    torch.save((QM,Qoptimizer),self.args.saveqMstr+str(epoch)+"_QRVAE.pt")
        
        elif self.qfntype == "qCRVAE":
            raise NotImplementedError
        
        else:
            raise("unknown qtype:")
 
