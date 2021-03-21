import numpy as np
from collections import Counter

from helper_functions import aggregate_counts, aggregate_counts_for_label
from models.base_model import BaseModel

class NaiveBayes(BaseModel):
    
    def __init__(self) -> None:
        BaseModel.__init__(self, None)
        self.feat2idx = {}
        self.lab2idx = {}
        self.is_trained = False

    def fit(self, D, C):
        parameters = self.train(D, C)
        self.is_trained = True

        return parameters
    

    def train(self, D, C):
        """Trains the model using add-1 smoothing.
        Args:
            D : List of lists of features.
            C : List of labels.
        Returns:
            parameters = np.zeros((vocab_size+1, len(C))). First row contains the logpriors per class.
            The remaining rows are the class loglikelihoods.
        """

        classes= np.unique(C)
        num_classes = len(classes)
        class_counts = Counter(C)

        Nc = [class_counts[c] for c in classes] #number of documents in D for class c
        D, bigdoc = aggregate_counts(D) #Features and Vocab
        Ndoc = len(D) #number of documents in D

        vocab_size = len(bigdoc)
        self.feat2idx = {f: i+1 for i, f in enumerate(bigdoc)} #keep 0 reserved for prior 
        self.lab2idx = {l: i for i,l in enumerate(classes)}
        print("{} classes, {} vocab size".format(num_classes, vocab_size))

        logpriors = np.zeros(num_classes) #prior probability for each class; beta
        likelihoods = np.zeros((vocab_size, num_classes)) #per class feature probability; alpha 

        for ci, c in enumerate(classes):
            logpriors[ci] = np.log(Nc[ci]/Ndoc)
            bigdoc_c = aggregate_counts_for_label(D, C, c)
            for fi, f in enumerate(bigdoc):
                if f in bigdoc_c:
                    likelihoods[fi][ci] = bigdoc_c[f] #count(w,c)

        per_class_sum = np.sum(likelihoods, axis=0) #sum(count(w',c))
        likelihoods = likelihoods + 1 #add-1 smoothing
        likelihoods = (likelihoods) / (per_class_sum + vocab_size) #add-1 smoothing

        parameters = np.zeros((vocab_size+1, num_classes))
        for i in range(len(classes)):
            parameters[0, i] = logpriors[i]
            parameters[1:, i] = np.nan_to_num(np.log(likelihoods[:, i]))

        return parameters

    
    def predict(self, x, w):
        """Predict the most likely class for each x given the trained parameters w.
        Args:
            x : Test set features. 
            w : Array of size vocab_size+1 x num_classes. 
        Returns:
            List of predictions.
        """
        if not self.is_trained:
            raise ("Train model first.")

        idx2lab = {i: l for l, i in self.lab2idx.items()}
        x_matrix = np.zeros((len(x),len(self.feat2idx)+1)) # add prior
        for i, inst in enumerate(x):
            # add prior
            for j, p_c in enumerate(w[0]):
                x_matrix[i][0] = 1
            #likelihood
            for f in inst:
                if f in self.feat2idx:
                    fidx = self.feat2idx[f]
                    x_matrix[i][fidx] = inst[f]
        predicted_label_indices = self.get_label(x_matrix, w)

        return [idx2lab[i] for i in predicted_label_indices]
    
    def get_label(self, x, w):
        """
        Computes the label for each data instance
        """
        scores = np.dot(x, w)
        return np.argmax(scores, axis=1).transpose()
    
    def evaluate(self, real, pred):
        correct = 0
        total = 0
        for real, pred in zip(real, pred):
            if real == pred:
                correct += 1
            total += 1
        
        return correct/total
