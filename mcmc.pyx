import numpy as np
cimport numpy as np
cimport cython

cdef noise():

cdef noise():
    ###--add noise
    
    pvm = np.random.normal(loc=0.,scale=0.067,    size=pv_.shape)+pv_
    cdef 
    pcm = np.random.normal(loc=0.,scale=0.067/36.,size=pc_.shape)+pc_

    ###--fit gaussian
    vv  = np.linspace(vr[0],vr[1],pv_.shape[0]) # fit p
    pp  = np.linspace(5.   ,-5.  ,pv_.shape[1]) # fit v
    vvc = np.linspace(vr[0],vr[1],pc_.shape[0])
    ppc = np.linspace(5.   ,-5.  ,pc_.shape[1])

    v0 = [0.5,0.,0.5,0.]
    p0 = [0.5,5.,1.,0.]

    vb,pb = [],[]
    for p in range(len(ppc)): # fit v in p-chan
        fa = {'y':pc_[:,p],'x':vvc,'err':[rms for i in range(len(vvc))]}
        nn = mpfit.mpfit(gs,p0,functkw=fa)
        vb.append(nn.params[1])
    for v in range(len(vv)): # fit p in v-chan
        fa = {'y':pv_[v,:],'x':pp,'err':[rms for i in range(len(pp))]}
        nn = mpfit.mpfit(gs,v0,functkw=fa)
        pb.append(nn.params[1])

    pf,vf=[],[]
    for i in range(len(vb)):
        if np.abs(ppc[i]) > 2. and np.abs(ppc[i]) < 4.:
            pf.append(ppc[i])
            vf.append(vb[i])
    for i in range(len(pb)):
        if (4.65 > vv[i] and vv[i] > vi) or (5.6 > vv[i] and vv[i] > 5.1):
            pf.append(pb[i])
            vf.append(vv[i])
        
    def sort_lists_by(l1,l2, key_list=0, desc=False):
        return izip(*sorted(izip(l1,l2), reverse=desc, key=lambda x: x[key_list]))

    vf,pf = sort_lists_by(vf,pf)
    br = 5
    return vf,pf,br
