import torch.nn as nn
import torch
import torch.nn.functional as F
from sklearn import preprocessing

class GlobalDiscriminator(torch.nn.Module):

    def __init__(self, in_feature=700+512,interm_size=512):
        super().__init__()
        
        self.l0 = torch.nn.Linear(in_feature, interm_size)
        self.l1 = torch.nn.Linear(interm_size, interm_size)
        self.l2 = torch.nn.Linear(interm_size, 1)

    def forward(self, language, visual):

        out = torch.cat((language, visual), dim=1)
        #对拼接后的featuremap进行线性l0，relu，l1，relu
        out = F.relu(self.l0(out))
        out = F.relu(self.l1(out))
        #最后再塞进一个线性网络l2，输出为一个标量
        out = self.l2(out)
        return out

class loss(nn.Module):
    def __init__(self,alpha=0.5):
        super(loss,self).__init__()
        self.maxpool = nn.AdaptiveMaxPool1d(1)
        self.global_D = GlobalDiscriminator()
        self.alpha = alpha

    def forward(self,language,visual):
        #language为[N,700],visual为[N,512,20]
        #1、整体dim
        
        total = self.totaldim(language,visual)
        #2、时序dim
        #frame = self.framedim(language,visual)

        #loss = -(total + self.alpha*frame)
        loss = -total
        return loss
    def framedim(self,language,visual):
        num = visual.shape[2]
        frame = [0.0]*num
        score = 0.0
        for i in range(num):
            frame[i] = self.totaldim(language,visual[:,:,0:i+1])
            #frame[i] = self.perframe(language,visual[:,:,0:i+1],visual)
        for i in range(num-1):
            if frame[i+1] < frame[i]:
                score = score + frame[i+1]-frame[i]
        return score

    def totaldim(self,language,visual):
        #language为[N,700],visual为[N,512,20]
        visual = self.maxpool(visual)#[N,512,1]
        visual = visual.squeeze(dim=2)#[N,512]
        visual_fake = torch.cat((visual[1:], visual[0:1]), dim=0)
        Ej = -F.softplus(-self.global_D(language, visual)).mean()
        Em = F.softplus(self.global_D(language, visual_fake)).mean()
        dim = Ej - Em
        return dim

    def compute(self,language,visual,top=1):
        #训练完成后，GlobalDiscriminator便得到了。测试时无需正负样本采样，只需要计算两者的互信息
        #language是(60,700),visual是(1,512,20)
        visual = self.maxpool(visual)#[1,512,1]
        visual = visual.squeeze(dim=2)#[1,512]
        num = language.shape[0]
        visual = visual.expand(num,-1)
        dimlist = -F.softplus(-self.global_D(language, visual)).squeeze()
        _,prd = torch.topk(dimlist,top)
        return prd
