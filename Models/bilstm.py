import torch
import torch.nn as nn
import torch.nn.functional as F
class BiLSTM(nn.Module):
    def __init__(self,voc_size,emb_size,hid_size,out_size):
        super(BiLSTM,self).__init__()
        self.embedding=nn.Embedding(voc_size,emb_size,)
        self.lstm=nn.LSTM(emb_size,hid_size,batch_first=True,bidirectional=True)

        self.output=nn.Linear(hid_size*2,out_size)
        self.softmax=F.softmax

    def forward(self, x):

        embedding=self.embedding(x)
        lstm,_=self.lstm(embedding)
        outputs=self.output(lstm)
        outputs=self.softmax(outputs,dim=2)
        # print(outputs)
        return outputs


