import os
from astropy.utils.data import get_readable_fileobj
from astropy.io import fits
from astropy import wcs
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

class SF(object):
    def __init__(self,dataset,name,od,bn,choice='directionless'):
        self.ds = dataset
        self.nm = name
        self.od = od
        self.bn = bn
        self.ch = choice
        
        with get_readable_fileobj(dataset, cache=True) as f:
            self.fitsfile = fits.open(f)
            self.data     = self.fitsfile[0].data
            self.header   = self.fitsfile[0].header
        w = wcs.WCS(self.header)
    
    def gd(self):
        gx,gy = np.gradient(self.data[0],1,1)
        return gx,gy
    def grad(self):
        gx,gy = self.gd()
        
        pa180 = np.zeros(gx.shape)
        pa360 = np.zeros(gx.shape)
        
        for yy in range(1,gx.shape[0]-1):
            for xx in range(1,gx.shape[1]-1):
                pa180[yy,xx] = np.mod(np.mod(360-np.degrees(np.arctan2(gy[yy,xx],gx[yy,xx])), 360),180)
                pa360[yy,xx] = np.mod(360-np.degrees(np.arctan2(gy[yy,xx],gx[yy,xx])), 360)
        if self.ch   == 'directionless': return pa180
        elif self.ch == 'full':          return pa360
        
    def draw(self):
        gf = self.grad()
        gx,gy = self.gd()
        
        f = open("gd.vg", "w")
        f.write('\n'+'COLOR'+' '+str(1))
        for xx in range(1,gf.shape[1]-1): # do not plot the ones at the boarder
            for yy in range(1,gf.shape[0]-1):
                f.write('\n'+'v'+' '+'abspix'+' '+'abspix'+' '+'sdvg'+' '+'no'+' '+ \
                        str(xx+1)+' '+str(yy+1)+' '+ \
                        str(10*np.abs(np.sqrt(gx[yy,xx]**2+gy[yy,xx]**2)))+' '+str(gf[yy,xx]))
        f.close()
        os.system('rm -rf test_gf.fits test_gf')

        hdu = fits.PrimaryHDU(gf)
        hdu.writeto('test_gf.fits')

        ## test drawing
        os.system('fits in=test_gf.fits out=test_gf op=xyin')
        os.system('cgdisp in=test_mom0,test_gf \
                   device=test_gf.ps/vcps slev=a,3.8 type=c,p \
                   levs1=3,6,9,12,15,18,21,24,30,33,36,39,42,45,48,51,54,60 labtyp=relpix \
                   nxy=1,1 \
                   range=0,360,lin,6 options=relax,wedge,blacklab cols1=0 olay=gd.vg')
    
    def sf(self): # default 180 directionless
        a  = self.grad()
        bn = self.bn
        od = self.od
        
        s = min(a.shape)
        b = max(a.shape)
        ln = np.unique(np.sort(np.array([(m*m+n*n) for m in range(s) for n in range(m,b)])))
        ct = np.zeros(len(ln)) # count if sf[ix] has been filled with how many pairs
        sf = np.zeros(len(ln))
        
        xv = a.shape[0]
        yv = a.shape[1]
        
        def cp(ar1,ar2):
            h = (ar1 - ar2) % 180.
            f = (ar2 - ar1) % 180. # always positive, no need for 'abs'
            return np.minimum(h,f)
            
        for (x,y),i in np.ndenumerate(a): # run through all index pairs
            ll = x**2+y**2
            ix = np.where(ln==ll)
        
            if x==0 and y!=0:
                aa = np.pad(a,((0,0),(y,0)),
                            mode='constant', constant_values=(np.nan))[:, :-y]
                nb = np.sum(np.pad(np.ones((xv,yv)),((0,0),(y,0)), mode='constant')[:, :-y])
            
                sf[ix] = (sf[ix]*ct[ix] + np.nansum(cp(a,aa)**od))/(nb + ct[ix]) # new average
                ct[ix] += nb
            
            elif y==0 and x!=0:
                aa = np.pad(a,((x,0),(0,0)),
                            mode='constant', constant_values=(np.nan))[:-x, :]
                nb = np.sum(np.pad(np.ones((xv,yv)),((0,0),(y,0)), mode='constant')[:-x, :])
            
                sf[ix] = (sf[ix]*ct[ix] + np.nansum(cp(a,aa)**od))/(nb + ct[ix]) 
                ct[ix] += nb
            
            elif x!=0 and y!=0:
                aa = np.pad(a,((x,0),(y,0)),
                            mode='constant', constant_values=(np.nan))[:-x, :-y]
                bb = np.pad(a,((0,x),(y,0)),
                            mode='constant', constant_values=(np.nan))[x:, :-y] # negative configuration
                nb = 2.*np.sum(np.pad(np.ones((xv,yv)),((x,0),(y,0)), mode='constant')[:-x, :-y])
                # number of computed pairs
        
                sf[ix] = (sf[ix]*ct[ix] + np.nansum(cp(a,aa)**od) \
                                        + np.nansum(cp(a,bb)**od))/(nb + ct[ix])
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
            sf_.append(np.sum((sf[j:i]**(1./od)*ct[j:i])/np.sum(ct[j:i]))) # weighted average of sf
            ln_.append(np.sum(ls[j:i]*ct[j:i])/np.sum(ct[j:i])) # weighted average of ls
            j = i
        
        return sf_,ln_
    
    def wrfile(self):
        sff,dis = self.sf()
        
        pp = PdfPages('SF_%s.pdf' %self.nm)
        plt.figure()
        plt.clf()
        
        plt.scatter(dis,sff,s=10,c='b',edgecolors='none')
        plt.xlabel('distance (pixel)')
        plt.ylabel('angle difference')
        plt.xlim(0,max(dis)+1)
        plt.ylim(0,90.)
        plt.title('%s: Structure Function for directionless gradients (Order %s)' %(self.nm,int(self.od)))
        pp.savefig()
    
        pp.close()
        
    def show(self):
        print self.data[0]
