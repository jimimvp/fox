from matplotlib import pyplot as plt

def plot_samples_2d(X_, X, X_orig):
    fig, ax = plt.subplots(1,3, figsize=(15, 5))
    
    ax[0].hist2d(X_.T[0], X_.T[1], bins=200)
    ax[0].set_title("Learned Dist")
    ax[1].hist2d(X.T[0], X.T[1], bins=200)
    ax[1].set_title("GT Dist")
    ax[2].hist2d(X_orig.T[0], X_orig.T[1], bins=200)
    ax[2].set_title("Before Training")
    return fig, ax