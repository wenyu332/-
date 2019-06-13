def accuracy(predictLists,truthLists):
    trueCount=0
    count=0
    for i in range(len(predictLists)):
        for j in range(len(predictLists[i])):
            if predictLists[i][j]==truthLists[i][j]:
                trueCount+=1
            count+=1
    return trueCount/count
