from matplotlib import pyplot as plt

def plot_samples_2d(X_, X=None):
    fig, ax = plt.subplots(2,1, figsize=(5, 10))
    
    ax[0].hist2d(X_.T[0], X_.T[1], bins=200)
    ax[0].set_title("Learned Dist")
    ax[1].hist2d(X.T[0], X.T[1], bins=200)
    ax[1].set_title("GT Dist")
    
    return fig, ax