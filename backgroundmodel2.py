# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 20:22:37 2019

@author: Ayodeji Babalola
"""
import gc
gc.collect()
import os.path
import matlab_funcs as mfun
import numpy as np
import Filter as ft
import wellmanager as wm
import ctrend as ct
from progressbar import Percentage,Bar,ProgressBar
import cell
import variogram
from scipy import linalg
import normal_score_transform as nscore
from joblib import Parallel, delayed

cell = getattr(cell,"cell")
variogram = getattr(variogram,"variogram") 
#from multiprocessing import Process,Pool
# To do:
# 1. Change LOGS and logs... confusing 
# parralell processing for run_model
def unwrap_self(arg, **kwarg):
    return backgroundmodel.krigging_parralell(*arg, **kwarg)

wellmanager = getattr(wm,"wellmanager") # extracting well class
class backgroundmodel():
    def __init__(self):
        self.wellfiles = None
        self.wellcoord = None
        self.dlines = None
        self.dtraces = None
        self.wellxy = None
        self.newlogs = None
        self.wellmanger = wellmanager()
        self.nwells = None
        self.wlog = None
        self.LOGS = None # cell
        self.logs = None # cell
        self.nlogs = None
        self.logzone = None # cell
        self.nz = None
        self.horfiles = None
        self.hor = None
        self.horz = None
        self.HorIndx = None # cell
        self.corr_indx = None
        self.coordName = None
        self.old_dt = None
        self.dt = None
        self.bkmodfreq = None
        self.fdom = None
        self.tseis = None
        self.tmin = None
        self.tmax = None
        self.modinterp = None
        self.model = None
        self.modvar = None
        self.Algorithm = None
        self.Type = None
        self.nCDP = None
        self.Bin = None
        self.nLines = None
        self.nTraces = None
        self.GEOM = None
        self.Grid = None
        self.nGrid = None
        self.bin = None
        self.vario_vp = variogram()   # import the algorithms
        self.vario_vs = variogram()
        self.vario_rho = variogram()  
        self.nzones = None
        self.layered = None
        self.layered_nscore = None
        self.krig_data = {}
        self.saving_dir = None

#---------------------------- 
    def __repr__(self):
        return repr("background-Modeling Class")
#---------------------------- 
    def load_hor(self,options = None):
        print('-----loading horizons-----------')
        if (options is None):
            options = 'pristine'
# MAKING SURE THAT TOTAL THICKNESS IS MORE THAN FILTER LENGTH
        period = round(1000/self.bkmodfreq)
        FL = round(period/self.dt)
        
        nhor =self.horfiles.size
        horz = mfun.cell(nhor)
        horm = mfun.cell(nhor)
        
        for i in range(nhor):
            m = mfun.load_obj(self.horfiles[i])
            tmp = np.stack((m.Inline,m.Crossline,m.Z), axis = 1)
            horz[i] = tmp
        ns = horz[0][:,0].size
        
        horm = horz
        
        for ii in range(ns):
            for i in range(nhor-1):
                temp = horm[i+1][ii,2] - horm[i][ii,2]
                if temp < (FL*self.dt):
                    horm[i+1][ii,2] = horm[i+1][ii,2] + (FL*self.dt) + 1
         
        if (options == 'pristine'):
            hor = horz
        elif(options == 'modify'):
            hor = horm
            
 #   recoordinating with seismic time sampling
            
        for i in range(nhor):
            temp = hor[i][:,2]
            tmp = self.recoordzone(temp)
            hor[i][:,2] = tmp
                
        self.hor = hor
     
#----------------------------         
    def init(self):
        if (os.path.isdir('Model Building2') is None):  # BUG
            os.mkdir('Model Building2')
            print('creating ...Model Building2.... in the current directory')
        
        if (self.Type == "Elastic"):
            self.nlogs  = 4 
        elif(self.Type == "Acoustic"):
            self.nlogs =3 
        self.nlogs = 2 
        self.wellmanger.wellnames = self.wellfiles
        self.wellmanger.read()
        
        self.nhor = self.horfiles.size
        self.nwells = self.wellfiles.size        
        self.wellcoord = np.stack((self.wellmanger.Inline,self.wellmanger.Crossline),axis = 1)
        self.wellxy = np.stack((self.wellmanger.Xloc,self.wellmanger.Yloc),axis = 1)
        
        self.logs = mfun.cell(self.nwells,self.nlogs)
        self.newlogs = mfun.cell(self.nwells,self.nlogs)
        
        self.load_hor()
        self.loadwells() 
        
        self.nCDP = self.hor[0][:,1].size
        self.ntrc = self.nLines * self.nTraces
        self.tseis = np.arange(self.tmin,self.tmax+self.dt,self.dt)
        
        if(self.nCDP !=self.ntrc):
            print("No of traces in the seismic grid is unequal to horizon")
        
        self.model_preallocate()  # stopped here
        
        if(self.tmin == 0):
            self.corr_indx = 0
        else:
            self.corr_indx = np.fix(1/self.dt*self.tmin)
#----------------------------  
    def loadwells(self):
        print('-----loading wells-----------')
        self.LOGS = mfun.cell(self.nwells,self.nlogs)
        for i in range(self.nwells):
            tmp_in = getattr(self.wellmanger.wells[i],self.wlog)
            time_in = self.wellmanger.wells[i].time
            
            if (time_in is None):
                msg = ' z-axis does not exist in' + self.wellmanger.wells[i]
                raise Exception (msg)            
            if (tmp_in is None):
                msg = self.wlog + 'does not exist in' + self.wellmanger.wells[i]
                raise Exception (msg)                
                
            if (self.old_dt == self.dt):
                tmp_out = tmp_in
                tcord = time_in
            else:
                tmp_out,tcord = ft.resample(tmp_in,time_in,self.old_dt,self.dt)
                
            self.LOGS[i,0] = tcord
            self.LOGS[i,1] = tmp_out
            
        self.cleanLOGS()
#----------------------------
    def cleanLOGS(self):
        print('inside cleanLOGS-> background modeling')
        print('---A better version is to use HRS idea of replacing nan values with averages----')        
        print('additionally removing nan value by interpolation: please find time to debug ')
        print('-------------------------------------------------------------------------------')
        LOGS = mfun.cell(self.nwells,self.nlogs)
        for i in range(self.nwells):
            LOGS[i,0],LOGS[i,1] = self.remove_nan(self.LOGS[i,0],self.LOGS[i,1])
            LOGS[i,1] = mfun.remove_nan_interp(LOGS[i,1])
        self.LOGS = LOGS
    
#----------------------------
    def remove_nan(self,time_in, dat_in):
       #   indnan = np.argwhere(np.isnan(dat_in))      
        ns = dat_in.size
        indnan = np.isnan(dat_in)   
        indx1 = np.nonzero(indnan==0)
        indx2 = np.nonzero(indnan==1)
#        indx1 = np.argwhere(np.isnan(dat_in))  

        
        if (indx1 is None):
            time_out = time_in
            dat_out = dat_in
        else:
            min_zone = np.min(indx2) -1
            max_zone = np.max(indx2) 
            if(max_zone +1 < ns):
                min_zone = max_zone
                max_zone = ns-1
            elif (max_zone+1 ==ns and min_zone==0):
                min_zone = np.min(indx1)
                max_zone = np.max(indx1)
            else:
                min_zone = np.min(indx1)
                max_zone = np.max(indx1)
        time_out = time_in[min_zone:max_zone]
        dat_out = dat_in[min_zone:max_zone]
        
        return time_out,dat_out
#----------------------------           
    def ctrendlogs(self):
        z = self.LOGS.extr_allrows(0)  # cell2mat(obj.LOGS(:,1)) 
        data = self.LOGS.extr_allrows(1)  # cell2mat(obj.LOGS(:,2)) 
        params = ct.ctrendlogs(data,z,self.wlog)
        self.logs = ct.predict(params,self.wlog,self.LOGS,self.tseis)
        
#----------------------------
        
    def distanc(self,loc):
        inline = loc[0]
        xline =  loc[1]    
        distance  = np.zeros(self.nwells)
        
        for i in range(self.nwells):
            distance[i] = np.sqrt(((inline - self.wellcoord[i,0])*self.Bin[0])**2 \
                    + ((xline -self.wellcoord[i,1])*self.Bin[1])**2)
        return distance

#----------------------------     
    def invdist2(self,dat,loc):
        weight = np.zeros(self.nwells)
        distance = self.distanc(loc)
        
        # calculating weight
        for i in range(self.nwells):
            weight[i] = distance[i]**-2/ np.sum(distance**-2)
            if (distance[i] == 0):
                print('--------------------------------------------------')
                print ('location where distance is zero is = ' + str(loc))
                print ('Dont panic : maybe at well-location')
                print ('well-locations are =')
                print(self.wellcoord)
                print('--------------------------------------------------')
        
        ns,nw = dat.shape
        data = np.zeros(ns)
        
        for ii in range(self.nwells):    
                data = data + (weight[ii]*dat[:,ii])
        
        return data
#----------------------------    
    def inverse_distance(self):
        
        k=0
        self.HorIndx = np.zeros((self.nCDP,self.nhor),dtype = int)
        tindex = np.zeros(2,dtype = int)
        ind = np.arange(2)
        for ii in range(self.nhor-1):
            for i in range(self.nwells):
                self.newlogs[i,1],self.newlogs[i,0]= \
                self.segment_logs(self.logzone[i,ind+k],self.logs[i,0],self.logs[i,1])
                
            insetlog = self.newlogs
            self.newlogs = self.normalize_logs()
            #nr = self.newlogs[0,0].size
            
            """          
            Pool = Pool(2)
            for mm in range(self.nCDP):
                P = Process(target = self.invdist,args = (mm,nr))
                P.start()
                P.join()
               # self.invdist(mm,nr)
            """
            #self.nCDP = 100
            pbar = ProgressBar(maxval= self.nCDP).start()          
            for mm in range(self.nCDP):
            #for mm in range(49150):
                for n in range(2):
                    tindex[n] = self.taxis(self.tseis,self.hor[n][mm,2])
                if (self.tseis[0] == 0) :   
                    self.HorIndx[mm,:] = tindex - self.corr_indx
                else:
                    self.HorIndx[mm,:] = tindex
                
    # I need this Index for Inversion
    # correction index added in case sart time is not zero            
    # inverse distance weighting                   
                tempdata = self.generate_data_by_resamp(self.newlogs,tindex)
                data = self.invdist2(tempdata,self.hor[0][mm,0:2])
                # when using : last index is not used
                self.model[mm,tindex[0]:tindex[1]] = data
                pbar.update(mm)

            pbar.finish()                
        
         # re-insert the well data            
            for i in range(self.nwells):
                indx = self.coordextr(i)                
                # edit this for multiple zones later
                tindex = self.HorIndx[indx,:] # tmp_ind = np.arange(2)
                #tindex  = self.HorIndx[indx,tmp_ind+k]
                self.model[indx,tindex[0]:tindex[1]] = insetlog[i,1]
               
#----------------------------   
    def invdist(self,mm,nr):
        k = 0
        tindex = np.zeros(2,dtype = int)        
        for n in range(2):
                    tindex[n] = self.taxis(self.tseis,self.hor[n+k][mm,2])
                    
        if (self.tseis[0] == 0) :
            self.HorIndx[mm,:] = tindex - self.corr_indx 
        else:
             self.HorIndx[mm,:] = tindex
             
        tempdata = self.generate_data_by_resamp(self.newlogs,tindex,nr)
        data = self.invdist2(tempdata,self.hor[0][mm,0:2])
        self.model[mm,tindex[0]:tindex[1]] = data
        
#----------------------------   
    def taxis(self,tseis,hor):
        if (hor.size > 100):
            nhor = 1
        else:
            nhor = hor.size
            
        hor = np.fix(hor)        
        if (nhor == 1):
            ind = np.nonzero(hor == tseis)
            tindex = int(ind[0])
        else:
            tindex = np.zeros(nhor)            
            for i in range(nhor):
                ind = np.nonzero(hor[i] == tseis)
                tindex[i] = int(ind[0])
                
        return tindex
            
        
#----------------------------                  
    def normalize_logs(self):
        
        nw,nl = self.newlogs.size()
        logs = mfun.cell(nw,nl)
        sz = self.newlogs.size_data()
        ndat = np.max(sz[:,1])
        tindex = [0, ndat] 
        ns_out = tindex[1] - tindex[0]
        
        if (self.nzones is None):
            self.nzones = np.int(ns_out)
            msg = 'number of zones is forced to be' + str(ns_out)
            print(msg) 
        else:
            ns_out = self.nzones
            
        for i in range(nw):
            nsamp = self.newlogs[i,1].size
            logs[i,1],logs[i,0] = \
            ft.resample(self.newlogs[i,1],self.newlogs[i,0],self.dt,(nsamp/ns_out))          
            
        return logs
    
    def tie_log(self,dat_in,dt,tindex):
        print("BUGGY: just resample directly for now")        
        ns_in = dat_in.size
        ns_out = abs(tindex[0] - tindex[1]) + 1
        
        if (ns_in > ns_out):
            ns_out = abs(tindex[0] - tindex[1]) 
        
        t1 = np.arange(ns_in) * dt
        t2 = np.arange(ns_out) * dt
        
        zone_in  = [0 ,np.max(t1)]
        zone_out = [0 ,np.max(t2)]
        
        dat_out,tout = self.stretchlog(dat_in,t1,zone_in,zone_out)
        return dat_out,tout
        
#----------------------------
    def stretchlog(self,trin,t,zonein,zoneout):
        print("BUGGY: just resample directly for now")
       
        dt = t[1] - t[0]
        tmin1 = zonein[0]
        tmin2 = zoneout[0]
        tmax1 = zonein[1]
        tmax2 = zoneout[1]
        
        nsampout = round((tmax2-tmin2)/dt) + 1
        dtout = (tmax1-tmin1)/(nsampout-1);
        tout = np.arange(tmin2,tmax2,dt)
        trout,tmp = ft.resample(trin,t,dt,dtout)
        
        return trout,tout
  
#----------------------------
    def generate_data(self,newlogs,tindex,nr):
        nwells,tm = newlogs.size()
        itr = abs(tindex[0] - tindex[1]) 
        
        indx = np.zeros((nwells,itr), dtype=int)
        for nn in range(nwells):
            indx[nn,:] = np.fix(np.linspace(0,nr-1,itr))
            
        tempdata = np.zeros((itr,nwells))
        
        for ii in range(itr):
            for i in range(nwells):
                tmp = newlogs[i,1] # extract the well
                tempdata[ii,i] = tmp[indx[i,ii]]
        return tempdata
#----------------------------
    def generate_data_by_resamp(self,newlogs,tindex):
        nwells,tm = newlogs.size()
        itr = abs(tindex[0] - tindex[1])    
               
        tempdata = np.zeros((itr,nwells))        
        for i in range(nwells):
            tmp = newlogs[i,1] # extract the well
            tempdata[:,i] = ft.resamp_by_newlength(tmp,itr)
        return tempdata
#----------------------------
    def segment_data(self):
        self.horzextr()
        self.logzone = self.recoordzone(self.logzone)
 
#----------------------------
    def  horzextr(self):
        data = np.zeros((self.nwells,self.nhor))
        
        for ii in range(self.nwells):
            indx = self.coordextr(ii)
            for i in range(self.nhor):
                temp = self.hor[i][:,2]
                data[ii,i] = temp[indx]
        
        self.logzone = data

#----------------------------
    def  coordextr(self,ind):
        coord = self.hor[0][:,0:2]
        IL = self.wellcoord[ind,0]
        XLR = self.wellcoord[ind,1]
        
        minIL = self.Grid[0,0]
        maxIL = self.Grid[0,1]
        minXL = self.Grid[1,0]
        maxXL = self.Grid[1,1]
        
        lines = np.min(np.nonzero(coord[:,0] == IL))
        cdp = np.fix((XLR - minXL)/self.dtraces)
        indx = lines  + cdp
        indx = np.int(indx)
        return indx        
             
#----------------------------
        """
    def  recoordzone(self,logzone):
        
        nwells,ncol = mfun.m_size(logzone)
        tcord = np.zeros((self.nwells,self.nlogs))
        
        for ii in range(nwells):
            for i in  range(ncol):
                minT = np.round(logzone[ii,i])
                num1 = minT  - np.remainder(minT,self.dt)
                tcord[ii,i] = num1
       """        
#----------------------------
    def segment_logs(self,zones,z_in,dat_in):
        zones = np.round(zones)
        aa = np.nonzero(z_in == zones[0])  # np.nonzero(z_in == zones[0])
        bb = np.nonzero(z_in == zones[1])
        if (aa is None):
            raise Exception ('lower zone exceeded')
        elif(bb is None):
            raise Exception ('upper zone exceeded')
        
        ind1  =int(aa[0])
        ind2  =int(bb[0])
        dat_out = dat_in[ind1:ind2]
        z_out = z_in[ind1:ind2]
        
        return dat_out,z_out
      
#----------------------------
    def applyfiter(self):
        for i in range(self.nCDP):
            tindex = (self.HorIndx[i,0], self.HorIndx[i,self.nhor-1])
            temp = self.model[i,tindex[0]:tindex[1]]
            #temp = self.remove_nan(temp)
            data = ft.Filt(temp,self.bkmodfreq,self.dt)  
            temp = []
            self.model[i,tindex[0]:tindex[1]] = data 
     # progress bar
  
#---------------------------        
    def gendata(self,newlogs,tindex,nr):
        tmp,nwells = newlogs.size
        itr = abs(tindex[0] - tindex[1])+1 
        indx = np.zeros(nwells,nr)
        for nn in range(nwells):
            indx[nn,:] = np.linspace(0,nr,itr,dtype = int)
               
        tempdata = np.zeros(itr,3,nwells)   
           
        for iii in range(itr):
            for ii in range(3):
                for i in range(nwells):
                    tmp = newlogs[i,ii+1]
                    tempdata[iii,ii,i] = tmp[indx[i,iii]]  # extract data from vp,vs,rho at indx         
           
        return tempdata
#---------------------------
    def generate_coordinate(self):
       """ 
       if (self.nGrid is not None):
            self.nGrid = self.Grid
       """
       print('-------------------------------------------------')
       print ('Inside background Modeling-> generate_coordinate')
       print('There is a BUG with estimating the coordinates')
       print('In V2: This should be a Class')
       IL = np.arange(self.Grid[0,0],self.Grid[0,1]+self.dlines,self.dlines)
       XL = np.arange(self.Grid[1,0],self.Grid[1,1]+self.dtraces,self.dtraces)
       
       if (self.GEOM[0,1]- self.GEOM[0,0])>1 :
           X = np.linspace(self.GEOM[0,0],self.GEOM[0,1],self.Bin[0],dtype = int)
       else:
           X = np.linspace(self.GEOM[0,1],self.GEOM[0,0],self.Bin[0],dtype = int)  
        
       if (self.GEOM[1,1]- self.GEOM[1,0])>1 :
           Y = np.linspace(self.GEOM[1,0],self.GEOM[1,1],self.Bin[1],dtype = int)
       else:
           Y = np.linspace(self.GEOM[1,1],self.GEOM[1,0],self.Bin[1],dtype = int)        
       
       nLines = IL.size
       nTraces = XL.size
       ns = nLines * nTraces
        
       coord_bin = np.zeros((ns,2))
       coord_XY = np.zeros((ns,2))
        
       k=0        
       for i in range(nLines):
           inL = np.tile(IL[i],nTraces)
           inX = np.tile(X[i],nTraces)
           
           ind = np.arange(nTraces)
           tmp = np.vstack((inL,XL))
           coord_bin[k*nTraces+ind,:] = tmp.T
           tmp = np.vstack((inX,Y))
           coord_XY[k*nTraces+ind,:] = tmp.T
           k = k+1
           return coord_bin,coord_XY       
        
       return coord_bin,coord_XY
#---------------------------- 
    def hor_grid(self):
        nr,nc = self.hor.size
        self.horz = mfun.cell(nr,nc)
        ntrc = 2
        print("ntrc is not parametezed , please check")
        ind = np.arange(ntrc)
        for i in range(nc):
            tmp = self.hor[i]
            self.horz[i] = tmp[ind]
#----------------------------     
    def hor_extr(self):
# function moves to well coordinates
# extract horizon top and base time samples
# this is required later to extract data from well-logs
# [data] = horzextr(wellcoord,hor,Grid)     
        data = np.zeros(self.nwells,self.nhor)
        
        for ii in range(self.nwells):
            indx = self.coordextr(ii)
            for i in range (self.nhor):
                tmp = self.hor[i]
                temp = tmp[:,2]
                data[ii,i] = temp[indx]
            
            self.logzone = data
                
#----------------------------  
    def model_preallocate(self):
        print("model pre-allocation started")
        self.nsamp = self.tseis.size
        ndata = self.nsamp*self.nCDP
       
        if (ndata>1e3):
            self.tmin = np.min(self.hor[0][:,2])
            self.tmax =np.max(self.hor[self.nhor-1][:,2])
            self.tseis = np.arange(self.tmin,self.tmax+self.dt,self.dt)
            self.nsamp = self.tseis.size
           
            self.model = np.zeros((self.nCDP,self.nsamp))
            if (self.Algorithm == 'krigging'):
                self.modvar = np.zeros((self.nCDP,self.nsamp))  
            
        else:
            self.model = np.zeros((self.nCDP,self.nsamp)) 
            if (self.Algorithm == 'krigging'):
                self.modvar = np.zeros((self.nCDP,self.nsamp)) 
           
        #print('model pre-allocation skipped')   
        print("model pre-allocation done")
#----------------------------          
    def loadhor(self,options = None):
        if (options is None):
            options  = 'pristine'
        
        Period = round(1000/self.bkmodfreq)
        FL = round(Period/self.dt)
        
        nhor = self.horfiles.size
        horz = mfun.cell(nhor)
        horm = mfun.cell(nhor)
        
        for i in range(nhor):
            m = mfun.load_obj(self.horfiles[i])
            temp  = np.concatenate(m.Inlne,m.Crossline,m.Z)
            horz[i] = temp
        
        nrm,ncm = horz.size_data
        ns = nrm[0]
        
        horm = horz
        
        for ii in range(ns):
            for i in range(nhor-1):
                temp = horm[i+1][ii,2] - horm[i][ii,2]
                if temp < (FL*self.dt):
                    horm[ii][ii,2] = horm[ii+1][ii,3] + (FL*self.dt) + 1
                    
        if (options == 'pristine'):
             hor =horz
        elif (options,'modify'):
             hor  = horm 
         
        for i in range(nhor):
            temp = hor[i][:,2]
            temp = self.recoordzone(temp)
            hor[i][:,2] = temp
            
        self.hor = hor
#---------------------------
        
    def recoordzone(self,logzone):
# function to recoordinate logzone with seismic coordinates        
        nwells,ncol = mfun.m_size(logzone)
        if (ncol == 1):
            tcord = np.zeros(nwells)
        else:
            tcord = np.zeros((nwells,ncol))

        for ii in range(nwells):
            if(ncol == 1):
                minT = np.round(logzone[ii])
                tcord[ii] = minT - np.remainder(minT,self.dt)
            else: 
                for i in range(ncol):
                    minT = np.round(logzone[ii,i])
                    num1 = minT - np.remainder(minT,self.dt)                
                    tcord[ii,i] = num1
        return tcord
              
#----------------------------    
    def lodwells(self):
        self.LOGS = mfun.cell(self.nwells,self.nlogs)
        
        for i in range(self.nwells):
            dat_in = self.wellmanger.wells[i].extract_logs(self.wlog)
            z_in  = self.wellmanger.wells[i].time
            self.LOGS[i,1],self.LOGS[i,0] = ft.resample(dat_in,z_in,self.old_dt,self.dt)
   
        self.clean_logs()            
        
#---------------------------- 
    def save_mod(self):
        print ('inside save_mod->backgrundmodeling class')
        print('To do : check if model directory exist and if not create it')               
     
        
        if (self.wlog == 'Vp'):
            mfun.numpy_save(self.model,'models\\VPmod')
        elif (self.wlog == 'Vs'):
            mfun.numpy_save(self.model,'models\\VSmod')
        elif (self.wlog == 'Rhob'):
            mfun.numpy_save(self.model,'models\\RHOmod') 
          
        #mfun.numpy_save(self.HorIndx,'models\HorIndx')
        
        print('Model Building done')
#----------------------------    
    def save_interp_mod(self):
        print ('inside save_interp_mod->backgrundmodeling class')
        print('To do : check if model directory exist and if not create it')
        
        BKModelcoord ={}
        
        if (self.coordName is None):
            coord_bin,coord_XY = self.generate_coordinate()
            BKModelcoord['Lines'] = coord_bin[:,0]
            BKModelcoord['Traces'] = coord_bin[:,1]
        else:
            tmp = mfun.load_obj(self.coordName)
            BKModelcoord['Lines'] = tmp['Lines']
            BKModelcoord['Traces'] = tmp['Traces']         
            
            

        BKModelcoord['HorIndx'] = self.HorIndx
        BKModelcoord['hor'] = self.hor
        BKModelcoord['corr_indx'] = self.corr_indx
        BKModelcoord['nLines'] = self.nLines 
        BKModelcoord['nTraces'] = self.nTraces
        BKModelcoord['GEOM'] = self.GEOM
        BKModelcoord['Grid'] = self.Grid 
        BKModelcoord['bin'] = self.bin 
        BKModelcoord['dt'] = self.dt
        BKModelcoord['tmin'] = self.tmin
        BKModelcoord['tmax'] = self.tmax
        BKModelcoord['wellfiles'] = self.wellfiles
        BKModelcoord['wellcoord'] = self.wellcoord
        BKModelcoord['wellxy'] = self.wellxy
        BKModelcoord['dlines'] = self.dlines
        BKModelcoord['dtraces'] = self.dtraces        
        mfun.save_obj('models\BKModelcoord',BKModelcoord)
        
                
        if (self.wlog == 'Vp'):
            mfun.numpy_save(self.model,'models\VPinterp')  
            mfun.save_obj('models\\vpLOGS',self.LOGS)   # original logs... no time axis
            mfun.save_obj('models\\vplogss',self.logs)   # resampled log on the seisimic grid 
            if (self.Algorithm):
               mfun.save_obj('models\\VPmodcov',self.modvar) 
        elif (self.wlog == 'Vs'):
            mfun.numpy_save(self.model,'models\VSinterp') 
            mfun.save_obj('models\\vsLOGS',self.LOGS) 
            mfun.save_obj('models\\vslogss',self.logs)  
            if (self.Algorithm):
               mfun.save_obj('models\\VSmodcov',self.modvar)             
        elif (self.wlog == 'Rhob'):
            mfun.numpy_save(self.model,'models\RHOinterp') 
            mfun.save_obj('models\\rhoLOGS',self.LOGS) 
            mfun.save_obj('models\\rhologss',self.logs)   
            if (self.Algorithm):
               mfun.save_obj('models\\RHOmodcov',self.modvar) 
               
    print('Interpolated models saved')
#----------------------------    
    def krigging_parralell(self,itr):    
        z = self.layered_nscore[itr,:]
        XY = self.krig_data['XY']
            
        x = self.wellxy[:,0]
        y = self.wellxy[:,1]
        xi = XY[:,0]
        yi = XY[:,1]
        numest = xi.size
        chunksize = 100
        """    
        tmp_x = mfun.bxfun_minus(x,x) 
        tmp_y = mfun.bxfun_minus(y,y) 
        Dx = np.hypot(tmp_x,tmp_y)  # check
        """
        if (self.wlog == 'Vp'):
            Dx = self.vario_vp.bxfun_dist(x,y)
        elif (self.wlog == 'Vs'):
            Dx = self.vario_vs.bxfun_dist(x,y)    
        elif (self.wlog == 'Rhob'):
            Dx = self.vario_rho.bxfun_dist(x,y)     
            
        # now calculate the matrix with variogram values 
        if (self.wlog == 'Vp'):
            At = self.vario_vp.covmat_2D(Dx)
        elif (self.wlog == 'Vs'):
            At = self.vario_vs.covmat_2D(Dx)
        elif (self.wlog == 'Rhob'):
            At = self.vario_rho.covmat_2D(Dx)
# the matrix must be expanded by one line and one row to account for
# condition, that all weights must sum to one (lagrange multiplier) 
        nr,nc = At.shape
        A = np.ones((nr+1,nc+1))
        A[0:nr,0:nc] = At  # note 0:3 actually is 0:2, python ugh angry face :)
        A[nr,nr] = 0

#    A is often very badly conditioned. Hence we use the Pseudo-Inverse for
#    solving the equations            
        #A = np.linalg.pinv(A)  
        A = linalg.pinv(A)
        
        z = np.append(z,0) # we also need to expand z
            
# allocate the output zi
        zi = np.empty(numest)
        zi[:]  = np.nan
        s2zi = np.empty(numest)
        s2zi[:] = np.nan
# parametrize engine
        nrloops   = np.int(numest/chunksize)
        
        for r in range(nrloops+1):
            if (r<nrloops):
                IX = np.arange(r*chunksize, (r+1)*chunksize) 
            else:
                IX = np.arange(r*chunksize + 1, numest)
         # build b
            if (self.wlog == 'Vp'):
                bx = self.vario_vp.bxfun_dist2(x,xi[IX],y,yi[IX])
            elif (self.wlog == 'Vs'):
                bx = self.vario_vs.bxfun_dist2(x,xi[IX],y,yi[IX]) 
            elif (self.wlog == 'Rhob'):
                bx = self.vario_rho.bxfun_dist2(x,xi[IX],y,yi[IX])
                
            # now calculate the matrix with variogram values 
            if (self.wlog == 'Vp'):
                bt = self.vario_vp.covmat_2D(bx)
            elif (self.wlog == 'Vs'):
                bt = self.vario_vs.covmat_2D(bx)
            elif (self.wlog == 'Rhob'):
                bt = self.vario_rho.covmat_2D(bx)
             
            nr,nc = bt.shape
            bb = np.ones((nr+1,nc))        
            bb[0:nr,0:nc] = bt  # note 0:3 actually is 0:2, python ugh angry face :)
  
            Lambda = np.matmul(A,bb) 
            zi[IX] = np.matmul(Lambda.T,z)            
            s2zi[IX] = np.sum(bb*Lambda,0) 
            #Lambda = np.matmul(A,bb)
            #zi[IX] = np.matmul(Lambda.T,z)            
            #s2zi[IX] = np.sum(bb*Lambda,0)  
            
        return zi,s2zi    
#----------------------------    
    def krigging(self,XY = None,z=None,chunksize =None):     
        
        if (z  is None):
            z = self.layered_nscore
            XY = self.krig_data['XY']
            
        x = self.wellxy[:,0]
        y = self.wellxy[:,1]
        xi = XY[:,0]
        yi = XY[:,1]
        numest = xi.size
        if (chunksize is None):
            chunksize = 100
        """    
        tmp_x = mfun.bxfun_minus(x,x) 
        tmp_y = mfun.bxfun_minus(y,y) 
        Dx = np.hypot(tmp_x,tmp_y)  # check
        """
        if (self.wlog == 'Vp'):
            Dx = self.vario_vp.bxfun_dist(x,y)
        elif (self.wlog == 'Vs'):
            Dx = self.vario_vs.bxfun_dist(x,y)    
        elif (self.wlog == 'Rhob'):
            Dx = self.vario_rho.bxfun_dist(x,y)     
            
        # now calculate the matrix with variogram values 
        if (self.wlog == 'Vp'):
            At = self.vario_vp.covmat_2D(Dx)
        elif (self.wlog == 'Vs'):
            At = self.vario_vs.covmat_2D(Dx)
        elif (self.wlog == 'Rhob'):
            At = self.vario_rho.covmat_2D(Dx)
# the matrix must be expanded by one line and one row to account for
# condition, that all weights must sum to one (lagrange multiplier) 
        nr,nc = At.shape
        A = np.ones((nr+1,nc+1))
        A[0:nr,0:nc] = At  # note 0:3 actually is 0:2, python ugh angry face :)
        A[nr,nr] = 0

#    A is often very badly conditioned. Hence we use the Pseudo-Inverse for
#    solving the equations            
        #A = np.linalg.pinv(A)  
        A = linalg.pinv(A)
        
        z = np.append(z,0) # we also need to expand z
            
# allocate the output zi
        zi = np.empty(numest)
        zi[:]  = np.nan
        s2zi = np.empty(numest)
        s2zi[:] = np.nan
# parametrize engine
        nrloops   = np.int(numest/chunksize)
        
        for r in range(nrloops+1):
            if (r<nrloops):
                IX = np.arange(r*chunksize, (r+1)*chunksize) 
            else:
                IX = np.arange(r*chunksize + 1, numest)
         # build b
            if (self.wlog == 'Vp'):
                bx = self.vario_vp.bxfun_dist2(x,xi[IX],y,yi[IX])
            elif (self.wlog == 'Vs'):
                bx = self.vario_vs.bxfun_dist2(x,xi[IX],y,yi[IX]) 
            elif (self.wlog == 'Rhob'):
                bx = self.vario_rho.bxfun_dist2(x,xi[IX],y,yi[IX])
                
            # now calculate the matrix with variogram values 
            if (self.wlog == 'Vp'):
                bt = self.vario_vp.covmat_2D(bx)
            elif (self.wlog == 'Vs'):
                bt = self.vario_vs.covmat_2D(bx)
            elif (self.wlog == 'Rhob'):
                bt = self.vario_rho.covmat_2D(bx)
             
            nr,nc = bt.shape
            bb = np.ones((nr+1,nc))        
            bb[0:nr,0:nc] = bt  # note 0:3 actually is 0:2, python ugh angry face :)
  
            Lambda = np.matmul(A,bb) 
            zi[IX] = np.matmul(Lambda.T,z)            
            s2zi[IX] = np.sum(bb*Lambda,0) 
            #Lambda = np.matmul(A,bb)
            #zi[IX] = np.matmul(Lambda.T,z)            
            #s2zi[IX] = np.sum(bb*Lambda,0)  
            
        return zi,s2zi
#----------------------------         
    def nscore_trans(self):  
        zmin = -1.0e21        
        zmax = 1.0e21 
        wt = np.ones(100)
        lt = 1
        ltpar = 0
        ut = 1
        utpar = 0
        nr,nc = self.layered.shape
        dat_flatten = self.layered.flatten()
        ns = dat_flatten.shape
        wt = np.ones(ns)
        ns = nscore.NormalScoreTransform(dat_flatten,wt,zmin,zmax,lt,ltpar,ut,utpar) 
        ns.create_transform_func()
        self.layered_nscore = np.reshape(ns.transform(self.layered),(nr,nc))
        return ns

#----------------------------         
    def krig(self):         
        self.HorIndx = np.zeros((self.nCDP,self.nhor),dtype = int)        
        tindex = np.zeros(2,dtype = int)
        ind = np.arange(2)
        k=0
        for ii in range(self.nhor-1):
            for i in range(self.nwells):
                self.newlogs[i,1],self.newlogs[i,0]= \
                self.segment_logs(self.logzone[i,ind+k],self.logs[i,0],self.logs[i,1])
                
        #insetlog = self.newlogs
        self.newlogs = self.normalize_logs()  # nzones is set here if none
        self.HorIndxZ = np.zeros((self.nCDP,self.nzones + 1),dtype = int) # had to here cos it uses nzones
        XY = self.creatinglogsInGrid()  # it had to be here cos it uses newlogs
        nscore_fun = self.nscore_trans()  # creates normal_scores layered logs
        pbar = ProgressBar(maxval= self.nCDP).start() 
        for mm in range(self.nCDP):           
            for n in range(2):
                tindex[n] = self.taxis(self.tseis,self.hor[n][mm,2])
            if (self.tseis[0] == 0) :   
                self.HorIndx[mm,:] = tindex - self.corr_indx
            else:
                self.HorIndx[mm,:] = tindex 
                
            self.HorIndxZ[mm,:] = np.round(np.linspace
                         (self.HorIndx[mm,0],self.HorIndx[mm,1],self.nzones+1,dtype = int))  
            pbar.update(mm)
        
        """
        #self.HorIndxZ = mfun.numpy_load('HorIndxz')
        model = np.zeros((self.nCDP,self.nzones))
        modelvar = np.zeros((self.nCDP,self.nzones))
        pbar = ProgressBar(maxval= self.nzones).start() 
        for ii in range(self.nzones):
            pbar.update(ii)
            z = self.layered_nscore[i,:]
            temp,modelvar[:,ii] = self.krigging(XY,z) # use createloginGrid for this.
            model[:,ii] = nscore_fun.back_transform(temp)
        print('....krigging done...')
        
        pbar = ProgressBar(maxval= self.nCDP).start() 
        for mm in range(self.nCDP):
            pbar.update(mm)            
            tindex = self.HorIndx[mm,:]
            ns = tindex[1] - tindex[0]            
            self.model[mm,tindex[0]:tindex[1]] = ft.resamp_by_newlength(model[:,i],ns)
            self.modvar[mm,tindex[0]:tindex[1]] = ft.resamp_by_newlength(modelvar[:,i],ns)
       
        """    
        pbar = ProgressBar(maxval= self.nzones).start()         
        for ii in range(self.nzones):
            pbar.update(ii)
            z = self.layered_nscore[ii,:]
            mean_tmp,covar_ = self.krigging(XY,z) # use createloginGrid for this.
            mean_ = nscore_fun.back_transform(mean_tmp)
            Tindex = np.array([self.HorIndxZ[:,ii],self.HorIndxZ[:,ii+1]],dtype =int)
            for i in range(self.nCDP):
                indx = Tindex[:,i]
                if (indx[0] == indx[1]):
                    self.model[i,indx[0]] = mean_[i]
                    self.modvar[i,indx[0]] = covar_[i]                    
                else:
                    self.model[i,indx[0]:indx[1]] = mean_[i]
                    self.modvar[i,indx[0]:indx[1]] = covar_[i]

#------------------------------------------------------------------------------ 
    def krig_parrallel(self): 
        self.HorIndx = np.zeros((self.nCDP,self.nhor),dtype = int)        
        tindex = np.zeros(2,dtype = int)
        ind = np.arange(2)
        k=0
        for ii in range(self.nhor-1):
            for i in range(self.nwells):
                self.newlogs[i,1],self.newlogs[i,0]= \
                self.segment_logs(self.logzone[i,ind+k],self.logs[i,0],self.logs[i,1])
                
        #insetlog = self.newlogs
        self.newlogs = self.normalize_logs()  # nzones is set here if none
        self.HorIndxZ = np.zeros((self.nCDP,self.nzones + 1),dtype = int) # had to here cos it uses nzones
        XY = self.creatinglogsInGrid()  # it had to be here cos it uses newlogs
        nscore_fun = self.nscore_trans()  # creates normal_scores layered logs
        pbar = ProgressBar(maxval= self.nCDP).start() 
        for mm in range(self.nCDP):           
            for n in range(2):
                tindex[n] = self.taxis(self.tseis,self.hor[n][mm,2])
            if (self.tseis[0] == 0) :   
                self.HorIndx[mm,:] = tindex - self.corr_indx
            else:
                self.HorIndx[mm,:] = tindex 
                
            self.HorIndxZ[mm,:] = np.round(np.linspace
                         (self.HorIndx[mm,0],self.HorIndx[mm,1],self.nzones+1,dtype = int))  
            pbar.update(mm)  
            
        itr = range(self.nzones)
        #itr = range(10)
        self.krig_data['XY'] = XY
        model = []
        """
        model = Parallel(n_jobs= -1, backend="threading")\
            (delayed(unwrap_self)(i) for i in zip([self]*len(itr), itr))
        """
        model = Parallel(n_jobs = 3,prefer="threads",verbose=50)\
            (delayed(unwrap_self)(i) for i in zip([self]*len(itr),itr)) 
            
        for ii in range(self.nzones):
             mean_ = nscore_fun.back_transform(model[ii][0]) 
             covar_ = model[ii][1]
             Tindex = np.array([self.HorIndxZ[:,ii],self.HorIndxZ[:,ii+1]],dtype =int)
             for i in range(self.nCDP):
                indx = Tindex[:,i]
                if (indx[0] == indx[1]):
                    self.model[i,indx[0]] = mean_[i]
                    self.modvar[i,indx[0]] = covar_[i]                    
                else:
                    self.model[i,indx[0]:indx[1]] = mean_[i]
                    self.modvar[i,indx[0]:indx[1]] = covar_[i]                 
         
#------------------------------------------------------------------------------ 
    def creatinglogsInGrid(self):         
        tmp = mfun.load_obj(self.coordName)
        """
        BKModelcoord ={}
        BKModelcoord['Lines'] = tmp['Lines']
        BKModelcoord['Traces'] = tmp['Traces']  
        BKModelcoord['X'] = tmp['X']
        BKModelcoord['Y'] = tmp['Y'] 
        """
        XY = np.vstack((tmp['X'],tmp['Y']))
        self.layered = np.zeros((self.nzones,self.nwells))
        for i in range(self.nwells):                       
            self.layered[:,i] = self.newlogs[i,1]
        return XY.T
        
#---------------------------- 
    def save_geodata(self,path):
        geodata = {}
        print('Run after VP, VS and Rho Models are saved on disk')
        fp_vplogs = path + 'vplogss'
        fp_vslogs = path + 'vslogss'
        fp_rhologs = path + 'rhologss' 
        
        fp_vpLOGS = path + 'vpLOGS' 
        fp_vsLOGS = path + 'vsLOGS'
        fp_rhoLOGS = path + 'rhoLOGS' 
        
        # This is the resampled log on the seisimic grid 
        # loaded into cells
        geodata['Vplogs'] =   mfun.load_obj(fp_vplogs) 
        geodata['Vslogs'] =   mfun.load_obj(fp_vslogs) 
        geodata['rhologs'] =   mfun.load_obj(fp_rhologs)    
        
        # This is the original  log loaded into cells
        geodata['VpLOGS'] =   mfun.load_obj(fp_vpLOGS) 
        geodata['VsLOGS'] =   mfun.load_obj(fp_vsLOGS) 
        geodata['rhoLOGS'] =   mfun.load_obj(fp_rhoLOGS)  
        
        mfun.save_obj('models\geodata',geodata)        
#---------------------------- 
    def process(self):
        self.init()
        self.ctrendlogs()
        self.segment_data()
        
        if (self.Algorithm is None):
            self.Algorithm = 'inverse-distance'
        
        if (self.Algorithm == 'inverse-distance'):
            self.inverse_distance()
        elif(self.Algorithm == 'krigging'):
            self.vario_vp.init()
            self.vario_vs.init()
            self.vario_rho.init()      
            
            self.vario_vp.setrot2D(0)
            self.vario_vs.setrot2D(0)
            self.vario_rho.setrot2D(0)            
            #self.krig()
            self.krig_parrallel()
        
        self.save_interp_mod()
        self.applyfiter()
        self.save_mod()
        
#----------------------------         
    def run_model(self,wlog):
        
        for i in range(len(wlog)):
            msg = '...............working on' + '  ' + wlog[i] + 'model...............'
            print(msg)
            self.wlog = wlog[i]
            self.process()
        self.save_geodata('models\\')
#*****************************************************************************                 
 
#----------------------------------------------- 
if __name__ == '__main__':             
    Poseidon3D = backgroundmodel()
    Poseidon3D.old_dt = 1
    Poseidon3D.dt = 1
    Poseidon3D.bkmodfreq = 12
    Poseidon3D.fdom = 125
    Poseidon3D.tmax = 3600
    Poseidon3D.tmin = 0
    Poseidon3D.GEOM = np.array(([422811.47, 431621.93],[ 8487174.21 ,8497673.86 ]))
    Poseidon3D.Bin = np.array([18.75, 12.5])
    Poseidon3D.rotation = 'EW'
    Poseidon3D.Grid = np.array(([2829, 3560], [2130, 2430]))
    Poseidon3D.coordName = 'seismic\\poseidoncoord'
    Poseidon3D.nLines = 732 
    Poseidon3D.nTraces = 301 
    Poseidon3D.dlines = 1  
    Poseidon3D.dtraces = 1   
    Poseidon3D.horfiles =  np.array(['hor_files\\Top_JameisonFM','hor_files\\Base_Horizon'])  
    Poseidon3D.wellfiles = np.array(['wells_files\\Boreas_1','wells_files\\Poseidon_2'\
                                     ,'wells_files\\Poseidon_North'])
    Poseidon3D.type = 'Elastic'    
    
        # variogram
    Poseidon3D.vario_vp.a_hmax = 7000
    Poseidon3D.vario_vp.a_hmin = 4000
    Poseidon3D.vario_vp.a_vert = 5000 # not taken into consideration
    Poseidon3D.vario_vp.c0 = 0
    Poseidon3D.vario_vp.ang = np.array([175 ,265, 0])
    Poseidon3D.vario_vp.nst = 1
    Poseidon3D.vario_vp.iss = 1
    Poseidon3D.vario_vp.it = 2
    Poseidon3D.vario_vp.cc = 0.16
    Poseidon3D.vario_vp.Type = 'bounded'
    
    Poseidon3D.vario_vs.a_hmax = 517
    Poseidon3D.vario_vs.a_hmin = 267
    Poseidon3D.vario_vs.a_vert = 267 # not taken into consideration
    Poseidon3D.vario_vs.c0 = 0
    Poseidon3D.vario_vs.ang = np.array([0 ,90, 0])
    Poseidon3D.vario_vs.nst = 1
    Poseidon3D.vario_vs.iss = 1
    Poseidon3D.vario_vs.it = 2
    Poseidon3D.vario_vs.cc = 0.26
    
    Poseidon3D.vario_rho.a_hmax = 517
    Poseidon3D.vario_rho.a_hmin = 267
    Poseidon3D.vario_rho.a_vert = 267  # not taken into consideration
    Poseidon3D.vario_rho.c0 = 0
    Poseidon3D.vario_rho.ang = np.array([0 ,90, 0])
    Poseidon3D.vario_rho.nst = 1
    Poseidon3D.vario_rho.iss = 1
    Poseidon3D.vario_rho.it = 2
    Poseidon3D.vario_rho.cc = 0.04
    Poseidon3D.Algorithm = 'krigging'
    #Poseidon3D.nzones = 100
    
    #-----------------------------------------------  
    """
    Poseidon3D.wlog  = 'Rhob' 
    Poseidon3D.process()
    
    Poseidon3D.wlog  = 'Vp' 
    Poseidon3D.process()
    
    Poseidon3D.wlog  = 'Vs' 
    Poseidon3D.process()
    """
    wlog = np.array(['Vs','Vp','Rhob'])
    #wlog = np.array(['Vp'])
    Poseidon3D.run_model(wlog)
    
    
    
           
    import matplotlib.pyplot as plt
    mat = Poseidon3D.model[0:Poseidon3D.nTraces-1,:]
    plt.imshow(mat.T)
     
        
        
                
            
            
            
    
        
        
            
        
    
        
        