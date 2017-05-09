import numpy as np
from numpy.random import rand
from scipy.stats import multivariate_normal
from time import *
class Forecast:
    def __init__(self,model=None,data=None,window_size=30,num_past=29):
        self.model=model
        self.means=model.means_#(n_components, n_features)
        self.weights=model.weights_# (n_components,)
        self.n_components=len(self.weights)
        self.covariance=model.covariances_#(n_components, n_features, n_features) if 'full'
        self.data=data
        self.window_size=window_size
        self.num_features=self.data.shape[1]/self.window_size
        self.num_past=num_past
        self.num_future=window_size-num_past
        #tmp_data_shape (4290,180)
        #window size: 30
        #num_features: 6
        #past length=29
        #future length=1

    def forecast(self):
        #Calculate the marginal multivariate normal distribution density
        #get 5000 samples
        marginal_density=[]
        sample=[]
        for i in range(self.num_features*(self.window_size-self.num_past)):
            tmp=-5+10*rand(5000)
            sample.append(tmp)
        sample=np.array(sample)
        sample=sample.T
        #print sample.shape
        last_value=[self.data[-1,:self.num_features*(self.window_size-1)]]
        last_value=np.repeat(last_value,5000,axis=0)
        #print last_value.shape
        #print sample.shape
        sample=np.concatenate((last_value,sample),axis=1)
        #print sample.shape
        prob=[]
        for i in range(self.n_components):
            tmp=multivariate_normal.pdf(sample, self.means[i], self.covariance)
            prob.append(sum(tmp))
        prob=np.array(prob)
        weights=self.weights*prob
        weights=weights/np.sum(weights)
        past_means=self.means[:,:self.num_past*self.num_features]
        future_means=self.means[:,self.num_past*self.num_features:]
        past_cov=self.covariance[:self.num_past*self.num_features,:self.num_past*self.num_features]
        fp_cov=self.covariance[self.num_past*self.num_features:,:self.num_past*self.num_features]
        future_cov=np.diag(self.covariance[self.num_past*self.num_features:,self.num_past*self.num_features:])
        future_std=np.sqrt(future_cov)
        tmp1= np.linalg.inv(past_cov)
        #print tmp1.shape
        tmp2= (last_value[0,:]-past_means).transpose()
        #print tmp2.shape
        tmp3=np.dot(tmp1,tmp2)
        #print tmp3.shape
        tmp4=np.dot(fp_cov,tmp3)
        #print tmp4.shape
        prediction=future_means+tmp4.transpose()
        #print prediction.shape
        prediction=np.dot(weights,prediction)
        return prediction,future_std




if __name__=='__main__':
    import matplotlib.pyplot as plt
    t=np.array([0,1,2,3])
    a=np.array([1,2,3,4])
    b=np.array([4,3,2,1])
    # now create a subplot which represents the top plot of a grid
    # with 2 rows and 1 column. Since this subplot will overlap the
    # first, the plot (and its axes) previously created, will be removed
    plt.subplot(221)
    plt.plot(range(12))
    plt.subplot(222)  # creates 2nd subplot with yellow background
    plt.plot(range(10))
    plt.subplot(223)  # creates 2nd subplot with yellow background
    plt.plot(range(11))
    plt.subplot(224)  # creates 2nd subplot with yellow background
    plt.plot(range(5))
    plt.show()


