from Models.hmm import HMM
import utils
import os
import metric
from Models.crf import CRFModel
from Models.bilstm import  BiLSTM
from torch.optim import Adamax

import torch
import torch.nn.functional as F
trainWordLists,trainTagLists,word2id,tag2id=utils.create('train.txt',make_vocab=True)
devWordLists,devTagList=utils.create('dev.txt',make_vocab=False)
#隐马尔科夫模型训练
print('HMM************************')
if os.path.exists('ckpts/hmm.pkl'):
    hmm=utils.loadModel('ckpts/hmm.pkl')
    predictTags = hmm.test(devWordLists, word2id, tag2id)
else:
    hmm=HMM(len(tag2id),len(word2id))
    hmm.train(trainWordLists,trainTagLists,tag2id,word2id)
    utils.saveModel('ckpts/hmm.pkl',hmm)
    predictTags=hmm.test(devWordLists,word2id,tag2id)
accuracy=metric.accuracy(predictTags,devTagList)
print('accuracy: ',accuracy)
print('CRF****************************')
# #条件随机序列场模型训练
if os.path.exists('ckpts/crf.pkl'):
    crf=utils.loadModel('ckpts/crf.pkl')
    print(crf)
    predictTags=crf.test(devWordLists)
else:
    crf=CRFModel()
    crf.train(trainWordLists,trainTagLists)
    utils.saveModel('ckpts/crf.pkl',crf)
    predictTags=crf.test(devWordLists)
accuracy=metric.accuracy(predictTags,devTagList)
print('accuracy: ',accuracy)
#BiLSTM模型训练
print('BiLSTM************************')
if os.path.exists('ckpts/bilstm.pkl'):
    model=utils.loadModel('ckpts/bilstm.pkl')
    devWordLists, devTagList = utils.create('dev.txt', make_vocab=False)
    devDatas = utils.batch_data(devWordLists, devTagList, word2id, tag2id)
    id2tag = dict((id, tag) for tag, id in tag2id.items())
    predictTags = []
    while 1:
        try:
            x, y = devDatas.__next__()
            predictScores = model(torch.LongTensor(x))
            scores = torch.argmax(predictScores, dim=2, )
            for i in range(len(scores)):
                predictTag = []
                for j in range(len(y[i])):
                    predictTag.append(id2tag[int(scores[i][j])])
                predictTags.append(predictTag)
        except:
            break
    acc = metric.accuracy(predictTags, devTagList)
    print('accuracy: ', acc)
else:
    id2tag = dict((id, tag) for tag, id in tag2id.items())
    # print(id2tag)
    biLstm=BiLSTM(len(word2id)+1,100,128,len(tag2id))
    optimer=Adamax(biLstm.parameters(),lr=0.001)
    bestAccuracy=0.0
    for epoch in range(30):
        print('epoch: ',epoch)
        trainDatas=utils.batch_data(trainWordLists,trainTagLists,word2id,tag2id)
        while 1:
            try:
                optimer.zero_grad()
                sentence,tag=trainDatas.__next__()
                predictScores=biLstm(torch.LongTensor(sentence))
                loss=0
                # print(len(sentence),len(tag))
                for i in range(len(sentence)):
                    # print(len(sentence[i]),len(tag[i]))
                    for j in range(len(tag[i])):
                        # print('tag',tag[i][j],'score:',predictScores[i][j][tag[i][j]])
                        loss+=torch.log(predictScores[i][j][tag[i][j]])
                        # print('loss: ', loss)
                loss=-loss/len(sentence)
                loss.backward()
                optimer.step()

                print('loss:',loss)
            except :
                break
        print('evaulation**************')
        devDatas=utils.batch_data(devWordLists,devTagList,word2id,tag2id)
        predictTags=[]
        while 1:
            try:
                sentence,tag = devDatas.__next__()
                predictScores = biLstm(torch.LongTensor(sentence))
                scores=torch.argmax(predictScores,dim=2,)
                predictTag=[]
                for i in range(len(scores)):
                    predictTag=[]
                    for j in range(len(tag[i])):
                        predictTag.append(id2tag[int(scores[i][j])])
                    predictTags.append(predictTag)
            except Exception as e:
                break

        acc=metric.accuracy(predictTags, devTagList)
        print('accuracy:',acc)
        if bestAccuracy<acc:
            bestAccuracy=acc
            utils.saveModel('ckpts/bilstm.pkl',biLstm)
