import numpy as np
import matplotlib.pyplot as plt

class Vec_Corr(object):
    def __init__(self,v1=np.ones((2,10,10)),n1='n1',v2=np.zeros(0),n2=None,rdn_s=1000):
        v1[v1==0.] = np.nan
        v2[v2==0.] = np.nan

        self.v1 = v1 # input vector field: 2 x N x M; each for x and y
        self.n1 = n1
        
        if v2.size > 0:
            self.v2 = v2
            self.n2 = n2
        else:
            self.v2 = self._rdn_v()
            self.n2 = "Random field"
            
        self.s = rdn_s + 10
        
        global _v1,_v2
        _v1 = v1
        _v2 = v2
        
    def _rdn_v(self):
        return np.random.random(self.v1.shape)
    
    def corr_c(self):
        ## Crosby 1993; rho^2 in [0,2]
        v1_x,v1_y = self.v1[0].flatten(),self.v1[1].flatten()
        v2_x,v2_y = self.v2[0].flatten(),self.v2[1].flatten()
        
        n = np.sum(~np.isnan(v1_x) * ~np.isnan(v2_x) * 1.)
        # number of overlapping points
        
        if n <= 1: return 0. # prevent divide by zero
        else:
            v1_x[np.isnan(v1_x)]=0.
            v1_y[np.isnan(v1_y)]=0.
            v2_x[np.isnan(v2_x)]=0.
            v2_y[np.isnan(v2_y)]=0.
        
            v_ = np.vstack([v1_x,v1_y,v2_x,v2_y])
            v_mean = ( v_.sum(1) / (v_ != 0).sum(1) ).reshape(4,1)
            vv = v_ - v_mean # subtraction of mean
            vf = np.matrix(vv)
            
            S = vf*vf.T / (n-1)
            
            return np.trace( S[0:2,0:2]**-1 * S[0:2,2:4] * \
                             S[2:4,2:4]**-1 * S[2:4,0:2] )
    
    def corr_h(self):
        ## Hanson 1992; rho^2 in [0,1]
        v1_x,v1_y = self.v1[0].flatten(),self.v1[1].flatten()
        v2_x,v2_y = self.v2[0].flatten(),self.v2[1].flatten()
        
        n = np.sum(~np.isnan(v1_x) * ~np.isnan(v2_x) * 1.)
        
        if n==0: return 0.
        else:
            v1_x[np.isnan(v1_x)]=0.
            v1_y[np.isnan(v1_y)]=0.
            v2_x[np.isnan(v2_x)]=0.
            v2_y[np.isnan(v2_y)]=0.
            
            v1_c = v1_x + v1_y * 1j
            v2_c = v2_x + v2_y * 1j
            
            v1_n = v1_c - v1_c.sum() / (v1_c != 0).sum() # non-zero mean
            v2_n = v2_c - v2_c.sum() / (v2_c != 0).sum()
            
            sig_1 = np.sqrt(abs(v1_n**2).sum() / n)
            sig_2 = np.sqrt(abs(v2_n**2).sum() / n)
            sig12 = np.sum( v1_n.conj() * v2_n ) / n
            
            rho = sig12 / (sig_1 * sig_2)
        return abs(rho**2)
        
    def rdm_corplt(self):
        cor_c,cor_h=[],[]
        gen = range(10,self.s,10)
        for N in gen: # 10 to rdn_size in steps of 10
            tpc,tph=0,0
            for i in range(10): # take average
                self.v1 = np.random.random((2,N))
                self.v2 = np.random.random((2,N))
                
                tpc += self.corr_c()
                tph += self.corr_h()
            
            cor_c.append(tpc/10)
            cor_h.append(tph/10)

        plt.figure(figsize=(16,9))
        plt.xticks(fontsize=24); plt.yticks(fontsize=24)
        plt.xlabel('Sample Size',fontsize=24); plt.ylabel('rho^2',fontsize=24)
        
        plt.plot(gen,cor_c,c='b',label='Crosby',linewidth=8)
        plt.plot(gen,cor_h,c='r',label='Hanson',linewidth=8)
        plt.legend(loc='upper right',fontsize=24)
        
        self.v1 = _v1 # reset
        self.v2 = _v2
        # plt.show()
        
    def crd_tran(self):
        ## coordinate translation
        v1 = self.v1
        v2 = self.v2
        
        rho_c,rho_h,(xc,yc),(xh,yh)=0.,0.,(0,0),(0,0)
        for (x,y),i in np.ndenumerate(v1[0]):
            if x==0 and y==0: continue
            elif x==0: 
                slc_z,slc_x,slc_y = slice(None),slice(None),slice(None,-y)
            elif y==0:
                slc_z,slc_x,slc_y = slice(None),slice(None,-x),slice(None)
            else:
                slc_z,slc_x,slc_y = slice(None),slice(None,-x),slice(None,-y)
            
            v1_ = np.pad(v1,((0,0),(x,0),(y,0)), mode='constant', 
                         constant_values=np.nan)[slc_z,slc_x,slc_y]
            v2_ = np.pad(v2,((0,0),(x,0),(y,0)), mode='constant', 
                         constant_values=np.nan)[slc_z,slc_x,slc_y]
            
            v1_[np.isnan(v1_)] = 0. # convert to zero for matrix arithmetics
            v2_[np.isnan(v2_)] = 0.
            
            # v1 translated
            self.v1 = v1_
            self.v2 = v2
            
            cor_c = self.corr_c()
            cor_h = self.corr_h()
            
            if cor_c > rho_c: rho_c = cor_c; (xc,yc)=(-x,-y) # negative for v1 tran.
            if cor_h > rho_h: rho_h = cor_h; (xh,yh)=(-x,-y)
        
            # v2 translated
            self.v1 = v1
            self.v2 = v2_
        
            cor_c = self.corr_c()
            cor_h = self.corr_h()
            
            if cor_c > rho_c: rho_c = cor_c; (xc,yc)=(x,y) # positive for v2 tran.
            if cor_h > rho_h: rho_h = cor_h; (xh,yh)=(x,y)
            
        self.v1 = _v1
        self.v2 = _v2
        return rho_c,rho_h,(xc,yc),(xh,yh)
