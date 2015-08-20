import numpy as np
from astropy.utils.data import get_readable_fileobj
from astropy.io import fits
from astropy import wcs
from pvextractor import Path, extract_pv_slice
import matplotlib.pyplot as plt
import matplotlib.cm as cm

class DS(object):
    def __init__(self,dataset,center=[0.,0.],pa=0.,length=10.,
                 vrange=None,rms=None,gray=True,level=[1.,10.]):
        self.ds  = dataset
        self.cr  = center
        self.pa  = pa
        self.ln  = length
        self.vr  = vrange
        self.rms = rms
        self.gs  = gray
        self.lv  = level
        
        with get_readable_fileobj(dataset, cache=True) as f:
            self.fitsfile = fits.open(f)
            self.data     = self.fitsfile[0].data
            self.header   = self.fitsfile[0].header
        w = wcs.WCS(self.header)

    def slc(self):
    ###--generate slices--###
        #how many " per pixel
        cv = round(np.abs(self.header["cdelt1"]*3600),5)
        #absolute pixel value
        ca = [self.cr[0]/cv+self.header["crpix1"]-1, \
              self.cr[1]/cv+self.header["crpix2"]-1]
        bm = ( int(ca[0]-self.ln/cv*np.sin(np.radians(self.pa))), \
               int(ca[1]+self.ln/cv*np.cos(np.radians(self.pa))) )
        tp = ( int(ca[0]+self.ln/cv*np.sin(np.radians(self.pa))), \
               int(ca[1]-self.ln/cv*np.cos(np.radians(self.pa))) )
        return [bm,tp]

    def pvdraw(self):
    ###--generate PV diagram---###
        if self.header["naxis"] == 3: da = self.data
        else                        : da = self.data[0]
        
        if self.vr == None:
            pv = extract_pv_slice(da, Path(self.slc()))
        else:
            vii = int(self.header["crpix3"]+
                     (self.vr[0] - self.header["crval3"]/1000.) \
                      /(self.header["cdelt3"]/1000.))-1
            vff = int(self.header["crpix3"]+
                     (self.vr[1] - self.header["crval3"]/1000.) \
                      /(self.header["cdelt3"]/1000.))-1
            pv = extract_pv_slice(da[vii:vff], Path(self.slc()))
        return pv.data

    def pvshow(self):
    ###--show PV diagram---##
        cp = plt.cm.get_cmap("Greys")

        y = np.linspace(self.ln   , -self.ln  , self.pvdraw().shape[1])
        
        bm,bmm = self.lv[0],self.lv[0]*self.rms
        tp,tpp = self.lv[1],self.lv[1]*self.rms
        if self.vr == None:
            x = np.linspace(self.header["crval3"]/1000., \
                            self.header["crval3"]/1000.* \
                            self.header["naxis%s" %(self.header["naxis"])], \
                            self.pvdraw().shape[0])
            if self.gs == False:
                if self.rms == None:
                    plt.contour(y,x,self.pvdraw(), colors='k')
                else:
                    lv = np.linspace(bmm,tpp,tp-bm+1)
                    plt.contour(y,x,self.pvdraw(), colors='k', levels=lv)
            else:
                if self.rms == None:
                    plt.contourf(y,x,self.pvdraw(), cmap=cp)
                else:
                    lv = np.linspace(bmm,tpp,tp-bm+1)
                    plt.contourf(y,x,self.pvdraw(), levels=lv, cmap=cp)
        else:
            x = np.linspace(self.vr[0], self.vr[1], self.pvdraw().shape[0])
            if self.gs == False:
                if self.rms == None:
                    plt.contour(y,x,self.pvdraw(), colors='k')
                else:
                    lv = np.linspace(bmm,tpp,tp-bm+1)
                    plt.contour(y,x,self.pvdraw(), colors='k', levels=lv)
            else:
                if self.rms == None:
                    plt.contourf(y,x,self.pvdraw(), cmap=cp)
                else:
                    lv = np.linspace(bmm,tpp,tp-bm+1)
                    plt.contourf(y,x,self.pvdraw(), levels=lv, cmap=cp)
        plt.ylabel("velocity (km/s)")
        plt.xlabel("position offset (arcsec)")
        plt.show()
