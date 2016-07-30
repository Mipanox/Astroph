'''
see the ipython notebook in ~/Desktop/coding_temp for usage and development
Updated: 2016/07/30:
- intrinsic m0,m1 calculations
'''

import os
from astropy.utils.data import get_readable_fileobj
from astropy.io import fits
from astropy import wcs
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

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
                self.header   = self.fitsfile[0].header
        else: self.po = np.zeros([0])
        # default ds and po to zero arrays with size 0 if not given
        self.po[self.po==0]=np.nan
        
        '''
        if polp:
            with get_readable_fileobj(polp, cache=True) as e:
                self.fitsfile = fits.open(e)
                self.pop      = self.fitsfile[0].data
                self.header   = self.fitsfile[0].header
        else: self.pop = np.ones(self.po.shape)*0.07 # set to 7% percentage
        '''
        if ds:
            self.s_i = min(np.sqrt(np.nanmean(self.ds[0]**2)),
                           np.sqrt(np.nanmean(self.ds[-1]**2))) # set intensity rms to be the min rms
            self.s_v = self.dshd["cdelt3"] / 1000.              # channel width, in km/s 
            self.ds[self.ds < self.s_i]=np.nan # blank low SN points

        self.m0 = self._m0()
        self.m1 = self._m1()[0]
            
    def _m0(self): return np.nansum(self.ds,axis=0)
    def _m1(self): # compute moment 1 an its associated error
        vel = np.arange(self.ds.shape[0]) * self.s_v + self.dshd["crval3"] / 1000.
        
        v_tot = np.sum(vel)
        v_exp = np.swapaxes(np.tile(vel,(self.ds.shape[2],self.ds.shape[1],1)),0,2)
        # expand vel into the same shape as ds

        mom_1 = np.nansum(self.ds * v_exp,axis=0) / np.nansum(self.ds,axis=0) # mom 1
        I_t   = np.nansum(self.ds        ,axis=0)
        Iv_t  = np.nansum(self.ds * v_exp,axis=0)
        m1__e = np.sqrt(np.nansum((self.s_i*(v_exp*I_t - Iv_t))**2,axis=0) + \
                        np.nansum((self.s_v*(self.ds*I_t))**2     ,axis=0))/ I_t**2
        
        return mom_1,m1__e
    
    def _gd(self):
        m1,e1 = self._m1()
        ## see numpy documentation : https://goo.gl/qbDii6
        ## testing notebook at npgradient
        ## - add error propagation
        
        outvals,errvals = [],[]
        N = len(m1.shape)
        
        # create slice objects --- initially all are [:, :, ..., :]
        sl1,sl2,sl3,sl4 = [slice(None)]*N,[slice(None)]*N,[slice(None)]*N,[slice(None)]*N

        for axis in range(N):
            # Use first order differences for time data
            out = np.empty_like(m1)
            err = np.empty_like(m1)
            sl1[axis],sl2[axis],sl3[axis] = slice(1,-1),slice(2,None),slice(None,-2)
            # 1D equivalent -- out[1:-1] = (y[2:] - y[:-2])/2.0
            out[sl1] = (m1[sl2] - m1[sl3])/2.
            err[sl1] = np.sqrt(e1[sl2]**2+e1[sl3]**2)/2.
        
            sl1[axis],sl2[axis],sl3[axis] = 0,1,0
            # 1D equivalent -- out[0] = (y[1] - y[0])
            out[sl1] = (m1[sl2] - m1[sl3])
            err[sl1] = np.sqrt(e1[sl2]**2+e1[sl3]**2)/2.

            sl1[axis],sl2[axis],sl3[axis] = -1,-1,-2
            # 1D equivalent -- out[-1] = (y[-1] - y[-2])
            out[sl1] = (m1[sl2] - m1[sl3])
            err[sl1] = np.sqrt(e1[sl2]**2+e1[sl3]**2)/2.
            
            outvals.append(out)
            errvals.append(err)

            # reset the slice object in this dimension to ":"
            sl1,sl2,sl3,sl4 = [slice(None)]*N,[slice(None)]*N,[slice(None)]*N,[slice(None)]*N
           
        gx,gy = outvals
        ex,ey = errvals
        return gx[1:-1,1:-1],gy[1:-1,1:-1],ex[1:-1,1:-1],ey[1:-1,1:-1]
    
    def _grad(self,dr=None,pol=None):
        if pol: 
            gy = self.po[1] # Q
            gx = self.po[2] # U
            ex,ey = self.du,self.du # estimated from given table (see note)
            
            os.system('rm -rf %s %s_U %s_Q %s_pa' %(self.pol[:-5],self.nm,self.nm,self.nm))
            os.system('fits in=%s out=%s op=xyin' %(self.pol,self.pol[:-5]))
            os.system('fits in=%s out=%s_Q.fits op=xyout region=image"(2,2)"' %(self.pol[:-5],self.nm))
            os.system('fits in=%s out=%s_U.fits op=xyout region=image"(3,3)"' %(self.pol[:-5],self.nm))
            os.system('fits in=%s_Q.fits out=%s_Q op=xyin' %(self.nm,self.nm))
            os.system('fits in=%s_U.fits out=%s_U op=xyin' %(self.nm,self.nm))
            os.system('impol in=%s_Q,%s_U pa=%s_pa sigma=0.0000001' %(self.nm,self.nm,self.nm))
            # do not blank output, calculate error below
            os.system('fits in=%s_pa out=%s_pa.fits op=xyout' %(self.nm,self.nm))
            
            with get_readable_fileobj('%s_pa.fits' %(self.nm), cache=True) as f:
                fitsfile = fits.open(f)
                pa       = fitsfile[0].data
        
        else:
            gx,gy,ex,ey = self._gd()
            pa180 = np.mod(np.mod(360-np.degrees(np.arctan2(gy,gx)), 360),180)
            pa360 = np.mod(360-np.degrees(np.arctan2(gy,gx)), 360)
            
        er180 = 180/np.pi * abs(gy/gx/(1+(gy/gx)**2)) * np.sqrt((ex/gx)**2 + (ey/gy)**2)
        
        if self.cri: # error threshold, above which SF will not count
            if pol:
                pa[er180/2.    > self.cri] = np.nan
                er180[er180/2. > self.cri] = np.nan
            else:
                pa360[er180 > self.cri] = np.nan
                pa180[er180 > self.cri] = np.nan
                er180[er180 > self.cri] = np.nan
        
        if    dr: return pa360,er180
        elif pol: return pa,   er180/2. # due to the 1/2 for Stokes
        else    : return pa180,er180
        
    def draw(self):
        ## deprecated: self.m0 is no longer provided at input ##
        # g  = self._m1()[0]
        # pp = self.pop # default using constant polarization percentage
        
        def cp(ar1,ar2): return np.minimum((ar1-ar2)%180.,(ar2-ar1)%180.)
        def cgdisp(self,m0,ds,mode,ang,clr):
            os.system('rm -rf %s %s' %(self.nm+mode+'.fits',self.nm+mode))

            hdu = fits.PrimaryHDU(ds)
            hdu.writeto('%s' %(self.nm+mode+'.fits'))
            
            os.system('fits in=%s out=%s op=xyin' %(self.nm+mode+'.fits',self.nm+mode))
            
            if m0 == 'y':
                os.system('cgdisp in=%s,%s \
                           device=%s.ps/vcps slev=a,%s type=c,p \
                           levs1=3,6,9,12,15,18,21,24,30,33,36,39,42,45,48,51,54,60 labtyp=relpix \
                           nxy=1,1 range=0,%s,lin,%s options=relax,wedge,blacklab cols1=0 olay=%s.vg'
                           %(self.m0,self.nm+mode,self.nm+mode,self.rms,ang  ,clr      ,mode))
                #            mom0   ,data        ,plot_name   ,rms     ,range,clr_scale,olay_name
            else: # draw pa of field only
                os.system('cgdisp in=%s \
                           device=%s_pa.ps/vcps slev=a,1000 type=c \
                           levs1=1 labtyp=relpix nxy=1,1 options=relax,blacklab cols1=0 olay=%s_e.vg'
                           %(self.m0,self.nm+mode,mode))
            
        if self.ds.size: # if gd != None
            gf = np.pad(self._grad(dr=1)[0],((1,1),(1,1)),mode='constant') # draw 360
            ef = np.pad(self._grad(dr=1)[1],((1,1),(1,1)),mode='constant')
            gx = np.pad(self._gd()[0]  ,((1,1),(1,1)),mode='constant')
            gy = np.pad(self._gd()[1]  ,((1,1),(1,1)),mode='constant')
            # pad back to dimension equal to the original map
            mode = 'gd'
            
            f = open("%s.vg"   %mode, "w")
            e = open("%s_e.vg" %mode, "w")
            f.write('\n'+'COLOR'+' '+str(1))
            f.write('\n'+'LWID'+' '+str(2))
            e.write('\n'+'LWID'+' '+str(2))
            
            M_mag = np.nanmax(np.abs(np.sqrt(gx**2+gy**2)))/2. # normalizes arrow length (max=2)
            for xx in range(gf.shape[1]):
                for yy in range(gf.shape[0]):
                    f.write('\n'+'v'+' '+'abspix'+' '+'abspix'+' '+'sdvg'+' '+'no'+' '+ \
                            str(xx+1)+' '+str(yy+1)+' '+ \
                            str(np.abs(np.sqrt(gx[yy,xx]**2+gy[yy,xx]**2))/M_mag)+' '+str(gf[yy,xx]))
                              
                    e.write('\n'+'COLOR'+' '+str(15))
                    e.write('\n'+'v'+' '+'abspix'+' '+'abspix'+' '+'sdvg'+' '+'no'+' '+ \
                            str(xx+1)+' '+str(yy+1)+' '+ \
                            str(np.abs(np.sqrt(gx[yy,xx]**2+gy[yy,xx]**2))/M_mag)+' '+str(gf[yy,xx]+ef[yy,xx]))
                    e.write('\n'+'v'+' '+'abspix'+' '+'abspix'+' '+'sdvg'+' '+'no'+' '+ \
                            str(xx+1)+' '+str(yy+1)+' '+ \
                            str(np.abs(np.sqrt(gx[yy,xx]**2+gy[yy,xx]**2))/M_mag)+' '+str(gf[yy,xx]-ef[yy,xx]))
                    
                    e.write('\n'+'COLOR'+' '+str(1))
                    e.write('\n'+'v'+' '+'abspix'+' '+'abspix'+' '+'sdvg'+' '+'no'+' '+ \
                            str(xx+1)+' '+str(yy+1)+' '+ \
                            str(np.abs(np.sqrt(gx[yy,xx]**2+gy[yy,xx]**2))/M_mag)+' '+str(gf[yy,xx]))
            f.close()
            e.close()
            cgdisp(self,m0='y',ds=gf,mode=mode,ang=360.,clr=6)
            cgdisp(self,m0='n',ds=gf,mode=mode,ang=360.,clr=6)
            
        if self.po.size:
            mode = 'pol'
            
            b  = self._grad(pol=1)[0] + 90.
            b_e = self._grad(pol=1)[1]
            
            f = open("%s.vg"   %mode, "w")
            e = open("%s_e.vg" %mode, "w")
            f.write('\n'+'COLOR'+' '+str(8))
            f.write('\n'+'LWID'+' '+str(3))
            e.write('\n'+'LWID'+' '+str(3))
            
            for xx in range(b.shape[1]):
                for yy in range(b.shape[0]):
                    f.write('\n'+'v'+' '+'abspix'+' '+'abspix'+' '+'sdvg'+' '+'no'+' '+ \
                            str(xx+1)+' '+str(yy+1)+' '+ \
                            str(0.5)+' '+str(b[yy,xx]    )+' '+str(0))
                    f.write('\n'+'v'+' '+'abspix'+' '+'abspix'+' '+'sdvg'+' '+'no'+' '+ \
                            str(xx+1)+' '+str(yy+1)+' '+ \
                            str(0.5)+' '+str(b[yy,xx]+180)+' '+str(0))
                    
                    e.write('\n'+'COLOR'+' '+str(15)) # error
                    e.write('\n'+'v'+' '+'abspix'+' '+'abspix'+' '+'sdvg'+' '+'no'+' '+ \
                            str(xx+1)+' '+str(yy+1)+' '+ \
                            str(0.5)+' '+str(b[yy,xx]+b_e[yy,xx]    )+' '+str(0))
                    e.write('\n'+'v'+' '+'abspix'+' '+'abspix'+' '+'sdvg'+' '+'no'+' '+ \
                            str(xx+1)+' '+str(yy+1)+' '+ \
                            str(0.5)+' '+str(b[yy,xx]+b_e[yy,xx]+180)+' '+str(0))
                    e.write('\n'+'v'+' '+'abspix'+' '+'abspix'+' '+'sdvg'+' '+'no'+' '+ \
                            str(xx+1)+' '+str(yy+1)+' '+ \
                            str(0.5)+' '+str(b[yy,xx]-b_e[yy,xx]    )+' '+str(0))
                    e.write('\n'+'v'+' '+'abspix'+' '+'abspix'+' '+'sdvg'+' '+'no'+' '+ \
                            str(xx+1)+' '+str(yy+1)+' '+ \
                            str(0.5)+' '+str(b[yy,xx]-b_e[yy,xx]+180)+' '+str(0))
                    
                    e.write('\n'+'COLOR'+' '+str(8))
                    e.write('\n'+'v'+' '+'abspix'+' '+'abspix'+' '+'sdvg'+' '+'no'+' '+ \
                            str(xx+1)+' '+str(yy+1)+' '+ \
                            str(0.5)+' '+str(b[yy,xx]    )+' '+str(0))
                    e.write('\n'+'v'+' '+'abspix'+' '+'abspix'+' '+'sdvg'+' '+'no'+' '+ \
                            str(xx+1)+' '+str(yy+1)+' '+ \
                            str(0.5)+' '+str(b[yy,xx]+180)+' '+str(0))
                
            f.close()
            e.close()
            cgdisp(self,m0='y',ds=b,mode=mode,ang=180.,clr=6)
            cgdisp(self,m0='n',ds=b,mode=mode,ang=180.,clr=6)
            
        if self.po.size and self.ds.size:
            mode = 'diff'
            
            gf = np.pad(self._grad(dr=1)[0],((1,1),(1,1)),mode='constant') # draw 360
            ef = np.pad(self._grad(dr=1)[1],((1,1),(1,1)),mode='constant')
            b   = self._grad(pol=1)[0] + 90.
            b_e = self._grad(pol=1)[1]
            
            d   = cp(b,gf)
            d_e = np.sqrt(b_e**2+ef**2)
            
            f = open("%s.vg"   %mode, "w")
            e = open("%s_e.vg" %mode, "w")
            f.write('\n'+'COLOR'+' '+str(4))
            f.write('\n'+'LWID'+' '+str(3))
            e.write('\n'+'LWID'+' '+str(3))
            
            for xx in range(d.shape[1]):
                for yy in range(d.shape[0]):
                    f.write('\n'+'v'+' '+'abspix'+' '+'abspix'+' '+'sdvg'+' '+'no'+' '+ \
                            str(xx+1)+' '+str(yy+1)+' '+ \
                            str(0.5)+' '+str(d[yy,xx]    )+' '+str(0))
                    f.write('\n'+'v'+' '+'abspix'+' '+'abspix'+' '+'sdvg'+' '+'no'+' '+ \
                            str(xx+1)+' '+str(yy+1)+' '+ \
                            str(0.5)+' '+str(d[yy,xx]+180)+' '+str(0))
                    
                    e.write('\n'+'COLOR'+' '+str(15)) # error
                    e.write('\n'+'v'+' '+'abspix'+' '+'abspix'+' '+'sdvg'+' '+'no'+' '+ \
                            str(xx+1)+' '+str(yy+1)+' '+ \
                            str(0.5)+' '+str(d[yy,xx]+d_e[yy,xx]    )+' '+str(0))
                    e.write('\n'+'v'+' '+'abspix'+' '+'abspix'+' '+'sdvg'+' '+'no'+' '+ \
                            str(xx+1)+' '+str(yy+1)+' '+ \
                            str(0.5)+' '+str(d[yy,xx]+d_e[yy,xx]+180)+' '+str(0))
                    e.write('\n'+'v'+' '+'abspix'+' '+'abspix'+' '+'sdvg'+' '+'no'+' '+ \
                            str(xx+1)+' '+str(yy+1)+' '+ \
                            str(0.5)+' '+str(d[yy,xx]-d_e[yy,xx]    )+' '+str(0))
                    e.write('\n'+'v'+' '+'abspix'+' '+'abspix'+' '+'sdvg'+' '+'no'+' '+ \
                            str(xx+1)+' '+str(yy+1)+' '+ \
                            str(0.5)+' '+str(d[yy,xx]-d_e[yy,xx]+180)+' '+str(0))
                    
                    e.write('\n'+'COLOR'+' '+str(8))
                    e.write('\n'+'v'+' '+'abspix'+' '+'abspix'+' '+'sdvg'+' '+'no'+' '+ \
                            str(xx+1)+' '+str(yy+1)+' '+ \
                            str(0.5)+' '+str(d[yy,xx]    )+' '+str(0))
                    e.write('\n'+'v'+' '+'abspix'+' '+'abspix'+' '+'sdvg'+' '+'no'+' '+ \
                            str(xx+1)+' '+str(yy+1)+' '+ \
                            str(0.5)+' '+str(d[yy,xx]+180)+' '+str(0))
                
            f.close()
            e.close()
            cgdisp(self,m0='y',ds=d,mode=mode,ang=90.,clr=8)
            cgdisp(self,m0='n',ds=d,mode=mode,ang=90.,clr=8)
    
    def _sf(self): # default 180 directionless
    ## NOTE: od has been defaulted to 2 (for error propagation)
        def cp(ar1,ar2):
            h = (ar1 - ar2) % 180.
            f = (ar2 - ar1) % 180. # always positive, no need for 'abs'
            return np.minimum(h,f)
        
        def overlap(ar1,ar2): # how many pairs to count
            return (~np.isnan(ar1) * ~np.isnan(ar2))
            
        def sf((ar,er)):
            bn = self.bn
            od = self.od
            s = min(ar.shape)
            b = max(ar.shape)
            ln = np.unique(np.sort(np.array([(m*m+n*n) for m in range(s) for n in range(m,b)])))
            ct = np.zeros(len(ln)) # count if sf[ix] has been filled with how many pairs
            sf = np.zeros(len(ln))
            ef = np.zeros(len(ln)) # error
            hs = [[] for _ in range(len(ln))] # for histogram, throw in raw data
            
            xv = ar.shape[0]
            yv = ar.shape[1]
            
            for (x,y),i in np.ndenumerate(ar): # run through all index pairs
                ll = x**2+y**2
                ix = np.where(ln==ll)[0][0]
        
                if x==0 and y!=0:
                    aa = np.pad(ar,((0,0),(y,0)),mode='constant', constant_values=(np.nan))[:, :-y]
                    ea = np.pad(er,((0,0),(y,0)),mode='constant', constant_values=(np.nan))[:, :-y]
                    
                    mask = overlap(ar,aa)
                    nb = np.sum(mask * 1.) # convert boolean to int
                    
                    r1 = cp(ar,aa)
                    if nb+ct[ix]==0: continue
                    else:
                        sf[ix]  = (sf[ix]*ct[ix] + np.nansum(r1**od))/(nb + ct[ix]) # new average
                        ef[ix] += np.nansum(r1**od * ((er*mask)**2 + (ea*mask)**2))
                    ct[ix] += nb
                    hs[ix] += r1[~np.isnan(r1)].flatten().tolist() # get all pairs and append to corresponding list

                elif y==0 and x!=0:
                    aa = np.pad(ar,((x,0),(0,0)),mode='constant', constant_values=(np.nan))[:-x, :]
                    ea = np.pad(er,((x,0),(0,0)),mode='constant', constant_values=(np.nan))[:-x, :]
                                            
                    mask = overlap(ar,aa)
                    nb = np.sum(mask * 1.)
            
                    r1 = cp(ar,aa)
                    if nb+ct[ix]==0: continue
                    else:
                        sf[ix] = (sf[ix]*ct[ix] + np.nansum(r1**od))/(nb + ct[ix]) 
                        ef[ix] += np.nansum(r1**od * ((er*mask)**2 + (ea*mask)**2))
                    ct[ix] += nb
                    hs[ix] += r1[~np.isnan(r1)].flatten().tolist()

                elif x!=0 and y!=0:
                    # positive and negative configuration
                    aa = np.pad(ar,((x,0),(y,0)),mode='constant', constant_values=(np.nan))[:-x, :-y]
                    ea = np.pad(er,((x,0),(y,0)),mode='constant', constant_values=(np.nan))[:-x, :-y]
                    bb = np.pad(ar,((0,x),(y,0)),mode='constant', constant_values=(np.nan))[x:, :-y]
                    eb = np.pad(er,((0,x),(y,0)),mode='constant', constant_values=(np.nan))[x:, :-y]
                    
                    maska,maskb = overlap(ar,aa),overlap(ar,bb)
                    nb = np.sum(maska*1.) + np.sum(maskb*1.)
                    # number of computed pairs
                    
                    r1,r2 = cp(ar,aa),cp(ar,bb)
                    if nb+ct[ix]==0: continue
                    else:
                        sf[ix] = (sf[ix]*ct[ix] + np.nansum(r1**od) \
                                                + np.nansum(r2**od))/(nb + ct[ix])
                        ef[ix] += np.nansum(r1**od * ((er*maska)**2 + (ea*maska)**2)) + \
                                  np.nansum(r2**od * ((er*maskb)**2 + (eb*maskb)**2))
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
                        ef_.append(0.)
                        ln_.append(0.)  # avoid divided by zero
                        hs_.append([0.])
                    else:
                        sf_.append(np.sum((sf[j:i]*ct[j:i])/np.sum(ct[j:i]))**(1./od)) # weighted average of sf
                        ef_.append(np.sqrt(np.sum(ef[j:i])) / sf_[-1] / np.sum(ct[j:i])) # final error
                        ln_.append(np.sum(ls[j:i]*ct[j:i])/np.sum(ct[j:i])) # weighted average of ls
                        hs_.append(reduce(lambda x, y: x + y, hs[j:i]))
                j = i
            if np.sum(ct[j:])==0:
                sf_.append(0.)
                ef_.append(0.)
                ln_.append(0.)
                hs_.append([0.])
            else:
                sf_.append(np.sum((sf[j:]*ct[j:])/np.sum(ct[j:]))**(1./od)) # don't forget the last bin
                ef_.append(np.sqrt(np.sum(ef[j:])) / sf_[-1] / np.sum(ct[j:]))
                ln_.append(np.sum(ls[j:]*ct[j:])/np.sum(ct[j:]))
                hs_.append(reduce(lambda x, y: x + y, hs[j:]))
            
            return sf_,ef_,ln_,hs_
        
        a = self.ds
        b = self.po
        
        if   a.size and not b.size : return sf(self._grad()),'directionless gradient'
        elif b.size and not a.size : return sf(self._grad(pol=1)),'polarization'
        elif a.size and b.size : 
            gf = np.pad(self._grad()[0],((1,1),(1,1)),mode='constant') 
            ef = np.pad(self._grad()[1],((1,1),(1,1)),mode='constant')
            b   = self._grad(pol=1)[0] + 90.
            b_e = self._grad(pol=1)[1]
            
            d   = (cp(b,gf),np.sqrt(b_e**2+ef**2))
            return sf(d),'difference (directionless)'
    
    def wrfile(self):
        ((sff,eff,dis,his),wh) = self._sf()
        
        pp = PdfPages('SF of %s_%s.pdf' %(wh,self.nm))
        plt.figure()
        
        plt.clf()
        
        plt.errorbar(dis,sff,yerr=eff,markersize=4,fmt='o',ecolor='b',c='b',elinewidth=1.5,capsize=4)
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
            
        pp.close()
