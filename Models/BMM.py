def init():
    dics=set()
    with open('D:\工程项目\分词\dic\chinese.txt','r',encoding='utf8') as f:
        for word in f:
            dics.add(word.strip())
    return dics

def cut_words(raw_sentence,dics):
    sentences=raw_sentence.strip()
    max_length=max(len(word) for word in dics)
    cut_word_list=[]
    sentence_length=len(sentences)
    while sentence_length>0:
        cut_length=min(max_length,sentence_length)
        subSentence=sentences[-cut_length:]
        while cut_length>0:
            if subSentence in dics:
                cut_word_list.append(subSentence)
                break
            elif len(subSentence)==1:
                cut_word_list.append(subSentence)
                break
            else:
                cut_length=cut_length-1
                subSentence=subSentence[-cut_length:]
        sentences=sentences[:-cut_length]
        sentence_length=sentence_length-cut_length
    cut_word_list.reverse()
    return '/'.join(cut_word_list)
def main():
    dics=init()
    while 1:
        print('输入需要切分的语句:')
        sentence = input()
        if not sentence:
            break
        else:
            results = cut_words(sentence, dics)
            print('分词结果： ', results)


if __name__ == '__main__':
    main()