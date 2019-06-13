from sklearn_crfsuite import CRF
import utils
class CRFModel(object):
    def __init__(self,algorithm='lbfgs',
                 c1=0.1,
                 c2=0.1,
                 max_iterations=100,
                 all_possible_transitions=False):
        self.model=CRF(algorithm='lbfgs',
                 c1=0.1,
                 c2=0.1,
                 max_iterations=100,
                 all_possible_transitions=False)

    def train(self,sentences,tagLists):
        features=[utils.sent2feature(sent)for sent in sentences]
        self.model.fit(features,tagLists)
    def test(self,sentences):
        features=[utils.sent2feature(sent)for sent in sentences]
        predictLists=self.model.predict(features)
        return predictLists