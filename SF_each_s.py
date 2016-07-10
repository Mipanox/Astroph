import os
from astropy.utils.data import get_readable_fileobj
from astropy.io import fits
from astropy import wcs
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from symfit.api import Parameter, Variable, Fit, exp, parameters, variables

class SF(object):
    def __init__(self,name,od,bn,ds=None,pol=None,polp=None,choice='directionless',
                 s_i=0.,s_v=0.,cri=None,**kwargs):
        self.nm = name
        self.od = od
        self.bn = bn
        self.ch = choice
        self.cri = cri
        self.__dict__.update(kwargs)
        
        if ds:
            with get_readable_fileobj(ds, cache=True) as f:
                self.fitsfile = fits.open(f)
                self.ds       = self.fitsfile[0].data
                self.dshd     = self.fitsfile[0].header
        else: self.ds = np.zeros([0])
        self.ds[self.ds==0]=np.nan # remove irregular points
        
        if pol:
            self.pol = pol # fits file
            with get_readable_fileobj(pol, cache=True) as e:
                self.fitsfile = fits.open(e)
                self.po       = self.fitsfile[0].data
            hdd = fits.open(pol)
            self.header   = hdd[0].header
        else: self.po = np.zeros([0])
        # default ds and po to zero arrays with size 0 if not given
        self.po[self.po==0]=np.nan
        
        if ds:
            self.s_i = min(np.sqrt(np.nanmean(self.ds[0]**2)),
                           np.sqrt(np.nanmean(self.ds[-1]**2))) # set intensity rms to be the min rms
            self.s_v = self.dshd["cdelt3"] / 1000.              # channel width, in km/s 
            self.ds[self.ds < 2. * self.s_i]=np.nan # blank low SN points
    
    def _m1(self): # compute moment 1 an its associated error
        vel = np.arange(self.ds.shape[0]) * self.s_v + self.dshd["crval3"] / 1000.
        
        v_tot = np.sum(vel)
        v_exp = np.swapaxes(np.tile(vel,(self.ds.shape[2],self.ds.shape[1],1)),0,2)
        # expand vel into the same shape as ds

        mom_1 = np.nansum(self.ds * v_exp,axis=0) / np.nansum(self.ds,axis=0) # mom 1
        I_t   = np.nansum(self.ds        ,axis=0)
        Iv_t  = np.nansum(self.ds * v_exp,axis=0)
        return mom_1
    
    def _m0(self): # compute moment 0 (temporary without error)
        return np.nansum(self.ds, axis=0)

    def __grad_all(self):
        def __get_grad(ar,n): # grad at different scales, see test_symfit_0707.ipynb
            m = n - 1
            (M,N) = ar.shape
    
            # output grad matrix with size (M-m)x(N-m)
            gd180 = np.zeros((M-m,N-m))
            gd360 = np.zeros((M-m,N-m))
    
            # initial values for fitting parameters
            ## Goodman et al. 1993 (doi:10.1086/172465)
            v0 = Parameter(value=5.)
            al = Parameter(value=0.)
            b1 = Parameter(value=0.) 

            v_0, a, b = parameters('v0, al, bl')
            x, y, z   = variables('x, y, z')      

            md = {z: v_0 + a * x + b * y}
    
            for (x,y),i in np.ndenumerate(ar):
                if x >= ar.shape[0]-m or y >= ar.shape[1]-m:
                    # fit grad from (x,y) (to (x+n, y+n)), so right/bottom edges are neglected
                    continue
                else:
                    ap = ar[slice(x,x+n),slice(y,y+n)]
            
                    # array of indices
                    xx,yy=[],[]
                    for (x_,y_),j in np.ndenumerate(ap):
                        xx.append(x_)
                        yy.append(y_)
                    xx = np.asarray(xx)
                    yy = np.asarray(yy)
            
                    ft = Fit(md, x=xx, y=yy, z=ap.flatten())
                    ft_result = ft.execute()
            
                    (a,b) = (ft_result.params.al,ft_result.params.bl)
            
                    gd180[x,y] = np.mod(np.mod(360-np.degrees(np.arctan(b/a)), 360),180)
                    gd360[x,y] = np.mod(360-np.degrees(np.arctan(b/a)), 360)
                
            return gd180,gd360
        
        ar = self._m1()
        n_max = max(ar.shape)/2
    
        gd180_tot,gd360_tot=[__get_grad(ar,2)[0]],[__get_grad(ar,2)[1]]
        # doubling n=2 case since a gradient for 1 cell is not well-defined.
        # replaced by 2x2
    
        for i in range(2,n_max+1):
            gd180_tot.append(__get_grad(ar,i)[0])
            gd360_tot.append(__get_grad(ar,i)[1])
        return gd180_tot,gd360_tot
    
    def _grad(self,dr=None,pol=None):
        if pol: 
            Q = self.po[1] # Q
            U = self.po[2] # U
            # ex,ey = self.du,self.du # estimated from given table (see note)
            s = min(Q.shape); b = max(Q.shape)
            ln = np.unique(np.sort(np.array([(m*m+n*n) for m in range(s) for n in range(m,b)])))
            
            def polpa(q,u):
                ci = 0.5*np.arctan(q/u)
                bla = np.where((q>0)&(u>0),ci, 
                                np.where((q>0)&(u<0),ci % (np.pi/2),
                                          np.where((q<0)&(u<0),ci+(np.pi/2),ci%np.pi)))
                return bla * 180./np.pi
            
            def avg_adj(ar,n): # reshaping even-indexed arrays
                (M,N) = ar.shape
                tt = np.zeros((M-n+1,N-n+1))
                for (x,y),i in np.ndenumerate(ar):
                    if x > ar.shape[0]-n or y > ar.shape[1]-n: continue
                    else:
                        ap = ar[slice(x,x+n),slice(y,y+n)]
                        tt[x,y] = ap.mean()
                return tt
            
            paP = []
            n_max = max(Q.shape)/2
            for i in range(2,n_max+1): # same 'averaging' scales as gd
                sigma = ( (i-1)**2 / (2*np.log(2)) )**0.5
                
                Q_ = ndimage.gaussian_filter(Q, sigma=sigma, mode='wrap')
                U_ = ndimage.gaussian_filter(U, sigma=sigma, mode='wrap')
                
                pa = polpa(Q_,U_)
                
                if i==2: paP.append(avg_adj(pa,i))
                if i%2: paP.append(pa[(i-1)/2:-(i-1)/2,(i-1)/2:-(i-1)/2])
                else:   paP.append(avg_adj(pa,i))
                # reshape to the size of gd
        
        else:
            pa180,pa360=self.__grad_all()
        
        if    dr: return np.array(pa360)
        elif pol: return np.array(paP)
        else    : return np.array(pa180)

    def _sf(self): # default 180 directionless
        def cp(ar1,ar2):
            h = (ar1 - ar2) % 180.
            f = (ar2 - ar1) % 180. # always positive, no need for 'abs'
            return np.minimum(h,f)
        
        def overlap(ar1,ar2): # how many pairs to count
            return (~np.isnan(ar1) * ~np.isnan(ar2))
            
        def sf(ar):
            bn = self.bn
            od = self.od
            s = min(ar[0].shape)
            b = max(ar[0].shape)
            ln = np.unique(np.sort(np.array([(m*m+n*n) for m in range(s) for n in range(m,b)])))
            ct = np.zeros(len(ln)) # count if sf[ix] has been filled with how many pairs
            sf = np.zeros(len(ln))
            # ef = np.zeros(len(ln)) # error
            hs = [[] for _ in range(len(ln))] # for histogram, throw in raw data
            
            for (x,y),i in np.ndenumerate(ar[0]): # run through all index pairs
                ll = x**2+y**2
                ix = np.where(ln==ll)[0][0]
                ic = max(x,y)-1
                
                if max(x,y) > int((max(ar[0].shape)+1)/2.): continue # no more sf > size/2
                elif x==0 and y!=0:
                    aa = np.pad(ar[ic],((0,0),(y,0)),mode='constant', constant_values=(np.nan))[:, :-y]
                    
                    mask = overlap(ar[ic],aa)
                    nb = np.sum(mask * 1.) # convert boolean to int
                    
                    r1 = cp(ar[ic],aa)
                    if nb+ct[ix]==0: continue
                    else:
                        sf[ix]  = (sf[ix]*ct[ix] + np.nansum(r1**od))/(nb + ct[ix]) # new average
                    ct[ix] += nb
                    hs[ix] += r1[~np.isnan(r1)].flatten().tolist() # get all pairs and append to corresponding list

                elif y==0 and x!=0:
                    aa = np.pad(ar[ic],((x,0),(0,0)),mode='constant', constant_values=(np.nan))[:-x, :]
                                            
                    mask = overlap(ar[ic],aa)
                    nb = np.sum(mask * 1.)
            
                    r1 = cp(ar[ic],aa)
                    if nb+ct[ix]==0: continue
                    else:
                        sf[ix] = (sf[ix]*ct[ix] + np.nansum(r1**od))/(nb + ct[ix])
                    ct[ix] += nb
                    hs[ix] += r1[~np.isnan(r1)].flatten().tolist()

                elif x!=0 and y!=0:
                    # positive and negative configuration
                    aa = np.pad(ar[ic],((x,0),(y,0)),mode='constant', constant_values=(np.nan))[:-x, :-y]
                    bb = np.pad(ar[ic],((0,x),(y,0)),mode='constant', constant_values=(np.nan))[x:, :-y]
                    
                    maska,maskb = overlap(ar[ic],aa),overlap(ar[ic],bb)
                    nb = np.sum(maska * 1.) + np.sum(maskb * 1.)# 
                    # number of computed pairs
                    
                    r1,r2 = cp(ar[ic],aa),cp(ar[ic],bb)
                    if nb+ct[ix]==0: continue
                    else:
                        sf[ix] = (sf[ix]*ct[ix] + np.nansum(r1**od) \
                                                + np.nansum(r2**od))/(nb + ct[ix])
                    ct[ix] += nb
                    hs[ix] += r1[~np.isnan(r1)].flatten().tolist() + \
                              r2[~np.isnan(r2)].flatten().tolist()
            
            ## bin
            ls = np.sqrt(ln)
            bs = np.arange(ls[0], round(ls[-1])+1, bn) - 1.e-12 # make sure no ambiguity for integers
            (counts, _) = np.histogram(ls, bs)
            idx = counts.cumsum() # binning indices for sf
    
            sf_,ef_,ln_,hs_ = [],[],[],[]
            j = 1
            for i in idx:
                if i==1: continue
                else:
                    if np.sum(ct[j:i])==0: 
                        sf_.append(0.)
                        ln_.append(0.)  # avoid divided by zero
                        hs_.append([0.])
                    else:
                        sf_.append(np.sum((sf[j:i]*ct[j:i])/np.sum(ct[j:i]))**(1./od)) # weighted average of sf
                        ln_.append(np.sum(ls[j:i]*ct[j:i])/np.sum(ct[j:i])) # weighted average of ls
                        hs_.append(reduce(lambda x, y: x + y, hs[j:i]))
                j = i
            if np.sum(ct[j:])==0:
                sf_.append(0.)
                ln_.append(0.)
                hs_.append([0.])
            else:
                sf_.append(np.sum((sf[j:]*ct[j:])/np.sum(ct[j:]))**(1./od)) # don't forget the last bin
                ln_.append(np.sum(ls[j:]*ct[j:])/np.sum(ct[j:]))
                hs_.append(reduce(lambda x, y: x + y, hs[j:]))
            
            return sf_,ef_,ln_,hs_
        
        a = self.ds
        b = self.po
        
        ## now omit these two to save computing time
        #if   a.size and not b.size : return sf(self._grad()),'directionless gradient'
        #elif b.size and not a.size : return sf(self._grad(pol=1)),'polarization'
        
        gf = self._grad()
        b  = self._grad(pol=1) + 90.
            
        d = []
        for i in range(len(gf)):
            d.append(cp(b[i],gf[i]))
        return (sf(gf),'directionless gradient'), \
               (sf(b), 'B-field'), \
               (sf(d),'difference (directionless)')
    
    def wrfile(self):
        ((sfg,efg,dig,hig),wg),((sfb,efb,dib,hib),wb),((sff,eff,dis,his),wh) = self._sf()
        
        def wrf(dis,sff,wh,his,clr):
            plt.clf()
        
            plt.errorbar(dis,sff,yerr=0.,markersize=4,fmt='o',ecolor=clr,c=clr,elinewidth=1.5,capsize=4)
            plt.xlabel('distance (pixel)')
            plt.ylabel('angle difference')
            plt.xlim(0,max(dis)+1)
            plt.ylim(0,90.)
            plt.title('%s: Structure Function for %s (Order %s)' %(self.nm,wh,int(self.od)))
            pp.savefig()
        
            bn = 5.
            for i in range(len(dis)):
                if dis[i] > 0:
                    hist,bins = np.histogram(np.array(his[i]),range=(0.,90.),
                                             bins=90/bn,density=True) # 5 deg per bin, prob. density func.
                    center = (bins[:-1] + bins[1:])/2
            
                    plt.clf()
                    plt.bar(center, hist, align='center',width=bn)
                    plt.xlim(0,90)
                    plt.xlabel('Angle difference (deg)')
                    plt.ylabel('Normalized number of pairs in each bin (%s deg)' %bn)
                    plt.title("Histogram at scale %s (pixel)" %round(dis[i],2))
                    pp.savefig()
        
        pp = PdfPages('SF of %s.pdf' %(self.nm))
        plt.figure()
        
        wrf(dig,sfg,wg,hig,'b')
        wrf(dib,sfb,wb,hib,'r')
        wrf(dis,sff,wh,his,'g')
        
        pp.close()
        
