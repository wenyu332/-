import torch

class HMM(object):
    def __init__(self,N,M):
        # N:状态数  M：观测数
        self.N=N
        self.M=M

        self.start=torch.zeros(N)
        self.state_trans_pro=torch.zeros(N,N)
        self.emit_pro=torch.zeros(N,M)
    def train(self,wordLists,tagLists,tagid,wordid):
        #要计算的参数有三个（1）起始概率  （2）状态转移概率  （3）发射概率

        #状态转移概率计算
        for tagList in tagLists:
            length=len(tagList)
            for i in range(length-1):
                current_id=tagid[tagList[i]]
                next_id=tagid[tagList[i+1]]
                self.state_trans_pro[current_id][next_id]+=1
        self.state_trans_pro=self.state_trans_pro/self.state_trans_pro.sum(dim=1,keepdim=True)

        #发射概率计算
        for tagList,wordList in zip(tagLists,wordLists):
            if len(tagList)==len(wordList):
                for i in range(len(tagList)):
                    tag_id=tagid[tagList[i]]
                    word_id=wordid[wordList[i]]
                    self.emit_pro[tag_id][word_id]+=1
        self.emit_pro[self.emit_pro==0]=1e-10
        self.emit_pro=self.emit_pro/self.emit_pro.sum(dim=1,keepdim=True)
        #起始概率计算
        for tagList in tagLists:
            tag_id=tagid[tagList[0]]
            self.start[tag_id]+=1
        self.start[self.start==0]=1e-10
        self.start=self.start/self.start.sum()

    def test(self,word_lists,wordid,tagid):
        pred_tag_lists=[]
        for wordList in word_lists:
            pred_tag_list=self.decoding(wordList,wordid,tagid)
            pred_tag_lists.append(pred_tag_list)
        return pred_tag_lists
    def decoding(self,words,wordid,tagid):
        start_pro=torch.log(self.start)
        state_trans_pro=torch.log(self.state_trans_pro)
        emit_pro=torch.log(self.emit_pro)

        #初始化viterbi矩阵
        length=len(words)
        viterbi=torch.zeros(self.N,length)

        #保存每个状态下的路径
        paths=torch.zeros(self.N,length).long()

        start_wordid=wordid.get(words[0],None)
        emit_pro=emit_pro.t()

        #如果该字不在此表中，则由每个状态发出的概率为平均值
        if start_wordid is None:
            emitPro=torch.log(torch.ones(self.N)/self.N)
        else:
            emitPro=emit_pro[start_wordid]
        viterbi[:,0]=start_pro+emitPro
        paths[:,0]=-1
        # print(viterbi)
        for i in range(1,length):
            word_id=wordid.get(words[i],None)
            if word_id is None:
                emitPro=torch.log(torch.ones(self.N)/self.N)
            else:
                emitPro=emit_pro[word_id]
            for tag_id in range(len(tagid)):
                # print(viterbi[:,i-1]+state_trans_pro[:,tag_id])
                max_pro,max_id=torch.max(viterbi[:,i-1]+state_trans_pro[:,tag_id],dim=0)
                # print(max_pro,emitPro)
                viterbi[tag_id,i]=max_pro+emitPro[tag_id]
                paths[tag_id,i]=max_id
        best_path_prob,best_path=torch.max(viterbi[:,length-1],dim=0)
        best_pointer=best_path.item()
        bestPaths=[best_pointer]


        # print(viterbi)
        # print(paths)
        # print(paths[best_pointer,length-2])
        for back_step in range(length-1,0,-1):
            best_pointer=paths[best_pointer,back_step]
            best_pointer=best_pointer.item()
            bestPaths.append(best_pointer)
        id2tag=dict((id,tag)for tag,id in tagid.items())
        # print(bestPaths)
        # print(id2tag)
        tagList=[id2tag[id] for id in reversed(bestPaths)]
        return tagList




