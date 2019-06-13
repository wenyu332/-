def init():
    dics=set()
    with open('D:\工程项目\分词\dic\chinese.txt','r',encoding='utf8') as f:
        for word in f:
            dics.add(word.strip())
    return dics
def cut_words(raw_sentence,word_dic):
    max_length=max(len(word)for word in word_dic)#最大切分长度
    sentence=raw_sentence.strip()
    word_length=len(sentence)
    cut_words=[]
    while word_length>0:
        cut_length=min(max_length,word_length)
        subSentence=sentence[0:cut_length]
        while cut_length>0:
            if subSentence in word_dic:
                cut_words.append(subSentence)
                break
            elif len(subSentence)==1:
                cut_words.append(subSentence)
                break
            else:
                cut_length=cut_length-1
                subSentence=subSentence[0:cut_length]
        sentence=sentence[cut_length:]
        word_length=word_length-cut_length
    words='/'.join(cut_words)
    return words
def main():
    dics=init()
    while 1:
        print('输入需要切分的语句:')
        sentence=input()
        if not sentence:
            break
        else:
            results=cut_words(sentence,dics)
            print('分词结果： ',results)
if __name__=='__main__':
    main()