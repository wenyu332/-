import pickle
import torch
import metric
def create(fileName,make_vocab=True):
    f=open(fileName,encoding='utf8')
    datas=f.readlines()
    f.close()
    word_lists=[]
    tag_lists=[]
    for data in datas:
        words=data.strip().split(' ')
        word_list=[]
        tag_list=[]
        for word in words:
            if len(word)==1:
                tag_list.append('S')
                word_list.append(word)
            elif len(word)==2:
                tag_list.append('B')
                tag_list.append('E')
                word_list.append(word[0])
                word_list.append(word[1])
            elif len(word)>2:
                tag_list.append('B')
                tag_list+=['M']*(len(word)-2)
                tag_list.append('E')
                for i in range(len(word)):
                    word_list.append(word[i])
        word_lists.append(word_list)
        tag_lists.append(tag_list)
    if make_vocab:
        tag2id={}
        word2id = {}
        for tag_list in tag_lists:
            for tag in tag_list:
                if tag not in tag2id:
                    tag2id[tag]=len(tag2id)
        for word_list in word_lists:
            for word in word_list:
                if word not in word2id:
                    word2id[word] = len(word2id)
        word2id['UNK']=len(word2id)
        word2id['PAD']=len(word2id)
        return word_lists,tag_lists,word2id,tag2id
    else:
        return word_lists, tag_lists
def saveModel(fileName,model):
    with open(fileName,'wb') as f:
        pickle.dump(model,f)

def loadModel(fileName):
    print('fileName',fileName)
    with open(fileName,'rb') as f:
        model=pickle.load(f)
    return model
def word2features(sent,i):
    word=sent[i]
    prev_word='<s>' if i==0 else sent[i-1]
    next_word='</s>' if i==len(sent)-1 else sent[i+1]

    features=\
        {
            'w':word,
            'w-1':prev_word,
            'w+1':next_word,
            'w-1:w':prev_word+word,
            'w:w+1':word+next_word,
            'bias':1
        }
    return features
def sent2feature(sent):
    return [word2features(sent,i)for i in range(len(sent))]

def batch_data(x,y,word2id,tag2id):
    if len(x)%32==0:
        for i in range(len(x)//32):
            start=32*i
            end=32*(i+1)
            sentences=[]
            lens=[]
            tags=[]
            for index in range(start,end):
                sentence=[word2id[word]for word in x[index]]
                lens.append(len(sentence))
                sentences.append(sentence)
                tag=[tag2id[tag]for tag in y[index]]
                tags.append(tag)
            max_len=max(lens)
            for num in range(len(sentences)):
                if len(sentence)<max_len:
                    for j in range(max_len-len(sentence)):
                        sentences[num].append(len(word2id))
            yield sentences,tags
    else:
        for i in range(len(x)//32+1):
            start=32*i
            end=32*(i+1)
            if end>len(x):
                end=len(x)
            sentences = []
            lens = []
            labels = []
            for index in range(start, end):
                sentence = [word2id[word] if word in word2id else 'UNK' for word in x[index]]
                lens.append(len(sentence))
                sentences.append(sentence)
                label = [tag2id[tag] for tag in y[index]]
                labels.append(label)
            max_len = max(lens)
            for num in range(len(sentences)):
                if len(sentences[num]) < max_len:
                    for i in range(max_len - len(sentences[num])):
                        sentences[num].append(word2id['PAD'])
                # print(len(sentences[num]))
            yield sentences, labels
def testGenerateData():
    trainWordLists,trainTagLists,word2id,tag2id=create('train.txt',make_vocab=True)
    datas=batch_data(trainWordLists[:50],trainTagLists[:50],word2id,tag2id)
    print(word2id)
    print(tag2id)
    while 1:
        try:
            x,y=datas.__next__()
            for i in range(len(x)):
                print(x[i],y[i])
        except:
            break

from sklearn.metrics import precision_score,precision_recall_fscore_support,classification_report,accuracy_score,auc
print(precision_score([0,1,2,3,4,0,1,2,3,4,0,1,2,3,4,0,1,2,3,4],[1,1,3,2,4,0,1,2,3,4,0,0,3,3,4,4,1,2,2,4],average='macro'))
print(precision_recall_fscore_support([0,1,2,3,4,0,1,2,3,4,0,1,2,3,4,0,1,2,3,4],[1,1,3,2,4,0,1,2,3,4,0,0,3,3,4,4,1,2,2,4],average='macro'))
print(classification_report([0,1,2,3,4,0,1,2,3,4,0,1,2,3,4,0,1,2,3,4],[1,1,3,2,4,0,1,2,3,4,0,0,3,3,4,4,1,2,2,4]))
print(accuracy_score([0,1,2,3,4,0,1,2,3,4,0,1,2,3,4,0,1,2,3,4],[1,1,3,2,4,0,1,2,3,4,0,0,3,3,4,4,1,2,2,4]))
print(auc([0,1,2,3,4,0,1,2,3,4,0,1,2,3,4,0,1,2,3,4],[1,1,3,2,4,0,1,2,3,4,0,0,3,3,4,4,1,2,2,4]))