import numpy as np
from pydmd import DMD
from sklearn.utils.extmath import randomized_svd

class HankelDMD_Predictor():
    """
    Class for making predictions with HankelDMD.
    Inputs:
    d: Number of delay embeddings used in forming Hankel matrix (int)
    svd_rank: Rank of SVD performed on Hankel matrix
    (Optional) dmd: DMD class used for fitting the Hankel matrix. Defaults to standard exact DMD (DMD class)
    Attributes:
    dmd: DMD class used for fitting. Contains modes, eigenvalues, amplitudes, etc. (DMD class)
    H: Hankel matrix form of the data (Numpy array)
    A: Low rank matrix that evolves the linear dynamics in the Hankel matrix (Numpy array)
    """
    def __init__(self,d: int,svd_rank: int, dmd = None):
        self._svd_rank = svd_rank
        self._d = d
        self._n = None
        self._m = None
        self._window_length = None
        self._U = None
        self._sig = None
        self._Vh = None
        self.A = None
        self.H = None
        if dmd is None:
            self.dmd = DMD(self._svd_rank,exact=True)
        else:
            self.dmd = dmd
    
    def _hankel_maker(self,X):
        gm = np.zeros([self._d*self._m, self._window_length-1])
        gp = np.zeros([self._d*self._m, self._window_length-1])
        for jj in range(self._m):
            tseries = X[:, jj]
            hankel_idx = np.array([np.arange(ii, ii+self._d) for ii in range(len(tseries) - self._d + 1)])
            hmat = tseries[hankel_idx].T
            gm[jj*self._d:(jj+1)*self._d, :] = hmat[:, :-1]
            gp[jj*self._d:(jj+1)*self._d, :] = hmat[:, 1:]
        return gm, gp
    
    def fit(self,X):
        """
        Creates Hankel matrix from data and fits DMD to that.
        Input:
        X: Data of shape num_timepoints x num_features (numpy array)
        """
        self._n = X.shape[0]
        self._m = X.shape[1]
        self._window_length = self._n - self._d + 1
        self.H,_ = self._hankel_maker(X)
        self.dmd.fit(self.H)
        self._U, self._sig, self._Vh = randomized_svd(self.H,n_components=self._svd_rank)
        self.A = np.real((self._U.T@self.dmd.modes@np.diag(self.dmd.eigs))@(np.linalg.pinv(self.dmd.modes)@self._U))
            
    def predict(self,N,reconstruct=True):
        """
        Creates forecast of N timepoints and optionally reconstructs the data as well.
        Input: 
        N: Number of timepoints to forecast (int)
        reconstruct: Whether or not to reconstruct original data as well (bool, default True)
        """
        if self.A is None:
            print("Error: You need to fit the model first.")
            return
        if reconstruct:
            prediction = np.zeros((self._svd_rank,self._window_length+N))
            prediction[:,0] = self._U.T@self.H[:,0]
        else:
            prediction = np.zeros((self._svd_rank,N))
            prediction[:,0] = self._U.T@self.H[:,-1]
        for j in range(1,prediction.shape[1]):
            prediction[:,j] = self.A@prediction[:,j-1]
        prediction = self._U@prediction
        if reconstruct:
            rec = np.zeros((self._n+N,self._m))
            for i in range(self._m):
                rec[:,i] = np.concatenate([prediction[i*self._d:(i+1)*self._d,0],prediction[(i+1)*self._d-1,1:]])
        else:
            rec = np.zeros((N,self._m))
            for i in range(self._m):
                rec[:,i] = prediction[(i+1)*self._d-1,:]
        return rec
        
        
        