'''
see the ipython notebook in ~/Desktop/coding_temp for usage and development
Update: 2016/05/10:
 - handling of nan values
 - added pol and difference (default 180)
'''

import os
from astropy.utils.data import get_readable_fileobj
from astropy.io import fits
from astropy import wcs
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

class SF(object):
    def __init__(self,name,od,bn,gd=None,pol=None,choice='directionless'):
        self.nm = name
        self.od = od
        self.bn = bn
        self.ch = choice
        
        if gd:
            with get_readable_fileobj(gd, cache=True) as f:
                self.fitsfile = fits.open(f)
                self.grd       = self.fitsfile[0].data[0]
                self.header   = self.fitsfile[0].header
        else: self.grd = np.zeros([0])
        
        if pol:
            with get_readable_fileobj(pol, cache=True) as e:
                self.fitsfile = fits.open(e)
                self.po       = self.fitsfile[0].data
                self.header   = self.fitsfile[0].header
        else: self.po = np.zeros([0])
        # default grd and po to zero arrays with size 0 if not given
    
    def gd(self):
        gx,gy = np.gradient(self.grd,1,1)
        return gx[1:-1,1:-1],gy[1:-1,1:-1]
    
    def grad(self):
        gx,gy = self.gd()
        
        pa180 = np.mod(np.mod(360-np.degrees(np.arctan2(gy,gx)), 360),180)
        pa360 = np.mod(360-np.degrees(np.arctan2(gy,gx)), 360)
        
        if self.ch   == 'directionless': return pa180
        elif self.ch == 'full':          return pa360
        
    def draw(self):
        if self.grd.size: # if grd != None
            gf = np.pad(self.grad(),((1,1),(1,1)),mode='constant')
            gx = np.pad(self.gd()[0],((1,1),(1,1)),mode='constant')
            gy = np.pad(self.gd()[1],((1,1),(1,1)),mode='constant')
            # pad back to dimension equal to the original map
        
            f = open("gd.vg", "w")
            f.write('\n'+'COLOR'+' '+str(1))
            for xx in range(gf.shape[1]):
                for yy in range(gf.shape[0]):
                    f.write('\n'+'v'+' '+'abspix'+' '+'abspix'+' '+'sdvg'+' '+'no'+' '+ \
                            str(xx+1)+' '+str(yy+1)+' '+ \
                            str(10*np.abs(np.sqrt(gx[yy,xx]**2+gy[yy,xx]**2)))+' '+str(gf[yy,xx]))
            f.close()
            os.system('rm -rf test_gf.fits test_gf')

            hdu = fits.PrimaryHDU(gf)
            hdu.writeto('test_gf.fits')

            os.system('fits in=test_gf.fits out=test_gf op=xyin')
            os.system('cgdisp in=test_mom0,test_gf \
                       device=test_gf.ps/vcps slev=a,3.8 type=c,p \
                       levs1=3,6,9,12,15,18,21,24,30,33,36,39,42,45,48,51,54,60 labtyp=relpix \
                       nxy=1,1 \
                       range=0,180,lin,6 options=relax,wedge,blacklab cols1=0 olay=gd.vg')
    
    def _sf(self): # default 180 directionless
        def cp(ar1,ar2):
            h = (ar1 - ar2) % 180.
            f = (ar2 - ar1) % 180. # always positive, no need for 'abs'
            return np.minimum(h,f)
        
        def overlap(ar1,ar2): # how many pairs to count
            return (~np.isnan(ar1) * ~np.isnan(ar2)) *1. # convert boolean to int
            
        def _sf(ar):
            bn = self.bn
            od = self.od
        
            s = min(ar.shape)
            b = max(ar.shape)
            ln = np.unique(np.sort(np.array([(m*m+n*n) for m in range(s) for n in range(m,b)])))
            ct = np.zeros(len(ln)) # count if sf[ix] has been filled with how many pairs
            sf = np.zeros(len(ln))
        
            xv = ar.shape[0]
            yv = ar.shape[1]
            
            for (x,y),i in np.ndenumerate(ar): # run through all index pairs
                ll = x**2+y**2
                ix = np.where(ln==ll)
        
                if x==0 and y!=0:
                    aa = np.pad(ar,((0,0),(y,0)),
                                mode='constant', constant_values=(np.nan))[:, :-y]
                    nb = np.sum(overlap(ar,aa))
                       
                    if nb+ct[ix]==0: continue # avoid dividing zero
                    else:
                        sf[ix] = (sf[ix]*ct[ix] + np.nansum(cp(ar,aa)**od))/(nb + ct[ix]) # new average
                    ct[ix] += nb
            
                elif y==0 and x!=0:
                    aa = np.pad(ar,((x,0),(0,0)),
                                mode='constant', constant_values=(np.nan))[:-x, :]
                    nb = np.sum(overlap(ar,aa))

                    if nb+ct[ix]==0: continue
                    else:
                        sf[ix] = (sf[ix]*ct[ix] + np.nansum(cp(ar,aa)**od))/(nb + ct[ix]) 
                    ct[ix] += nb
            
                elif x!=0 and y!=0:
                    aa = np.pad(ar,((x,0),(y,0)),
                                mode='constant', constant_values=(np.nan))[:-x, :-y]
                    bb = np.pad(ar,((0,x),(y,0)),
                                mode='constant', constant_values=(np.nan))[x:, :-y] # negative configuration
                    nb = 2.*np.sum(overlap(ar,aa)) # number of computed pairs, 2 for aa & bb

                    if nb+ct[ix]==0: continue
                    else:
                        sf[ix] = (sf[ix]*ct[ix] + np.nansum(cp(ar,aa)**od) \
                                                + np.nansum(cp(ar,bb)**od))/(nb + ct[ix])
                    ct[ix] += nb
                
            ## bin
            ls = np.sqrt(ln)
            bs = np.arange(ls[0], round(ls[-1])+1, bn) - 1.e-12 # make sure no ambiguity for integers
            (counts, _) = np.histogram(ls, bs)
            idx = counts.cumsum() # binning indices for sf
        
            sf_,ln_ = [],[]
            j = 1
            for i in idx:
                if i==1: continue
                elif np.sum(ct[j:i])==0: 
                    sf_.append(0.)
                    ln_.append(0.) # avoid divided by zero
                else:
                    sf_.append(np.sum((sf[j:i]*ct[j:i])/np.sum(ct[j:i]))**(1./od)) # weighted average of sf
                    ln_.append(np.sum(ls[j:i]*ct[j:i])/np.sum(ct[j:i])) # weighted average of ls
                j = i
            if np.sum(ct[j:])==0:
                sf_.append(0.)
                ln_.append(0.)
            else:
                sf_.append(np.sum((sf[j:]*ct[j:])/np.sum(ct[j:]))**(1./od)) # don't forget the last bin
                ln_.append(np.sum(ls[j:]*ct[j:])/np.sum(ct[j:]))
        
            return sf_,ln_
            
        a = self.grd
        b = self.po     # remember to make sure they have the same shape
        
        if   a.size and not b.size  : return _sf(self.grad()),'directionless gradient'
        elif b.size and not a.size  : return _sf(b),'polarization'
        elif a.size and b.size      : 
            gf = np.pad(self.grad(),((1,1),(1,1)),mode='constant') # padding to match dimensions
            return _sf(cp(gf,b)),'difference (directionless)'
    
    def wrfile(self):
        ((sff,dis),wh) = self._sf()
        
        pp = PdfPages('SF_%s.pdf' %self.nm)
        plt.figure()
        plt.clf()
        
        plt.scatter(dis,sff,s=10,c='b',edgecolors='none')
        plt.xlabel('distance (pixel)')
        plt.ylabel('angle difference')
        plt.xlim(0,max(dis)+1)
        plt.ylim(0,90.)
        
        plt.title('%s: Structure Function for %s (Order %s)' %(self.nm,wh,int(self.od)))
        pp.savefig()
            
        pp.close()
        
    def show(self):
        print self.data[0]
