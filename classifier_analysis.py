from pylab import *
import tables
import mdp
from mvpa.suite import *

def pca(data, cutoff=0.98):
    """Expects data to be of shape (2, channels, trials).
    Returns a list of conditions length of arrays of shape channels x trials."""
    
    data = hstack((data[0], data[1])).T
    
    pcanode = mdp.nodes.PCANode(output_dim=cutoff, svd=True)
    pcanode.train(data)
    pcanode.stop_training()
    
    print "%d principal components were kept." % pcanode.output_dim
    
    pca_data = pcanode(data)
    return array(vsplit(pca_data, 2))

def classify(data):
    """Expects data of shape (2, channels, trials)."""

    data = pca(data)
    cond1 = Dataset(samples=data[0], labels=0)
    cond2 = Dataset(samples=data[1], labels=1)
    dataset = cond1 + cond2
    
    cv = CrossValidatedTransferError(
                TransferError(SVM()),
                OddEvenSplitter())
    
    error = cv(dataset)
    
    return error

def classifier_analysis(the_file):
    file = tables.openFile(the_file)
    raw_epochs = file.root.lowpass_data_epochs
    conditions, samples, channels, trials = raw_epochs.shape
    
    accuracy = [classify(raw_epochs[:, sample, ...]) for sample in range(samples)]
    
    return accuracy
    
if __name__ == "__main__":
    
    acc = classifier_analysis("R1442.h5")
    
    plot(acc)
    show()