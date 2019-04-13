import numpy as np
from scipy.optimize import curve_fit, leastsq
from scipy.signal import argrelmax, argrelmin
import inspect
import logging

import pympedance.Synthesiser as psyn


###
# Addition to psyn: parallel compliance

class StraightDuctWithParallel(psyn.StraightDuct):
    def __init__(self, *args, **kwargs):
        self.compliance = kwargs.pop('compliance',None)
        self.inertance = kwargs.pop('inertance',None)
        self.resistance = kwargs.pop('resistance',None)
        super(StraightDuctWithParallel, self).__init__( *args, **kwargs)
         
    def two_point_transfer_mx_at_freq(self, *args, **kwargs):
        """
        return the travelling wave matrix T of the
        duct section between from_pos to end_pos.

        [p_out, p_in]_from_pos = T [p_out, p_in]_to_pos
        """

        tmx = super().two_point_transfer_mx_at_freq( *args, **kwargs)
        print('twopt')

        omega = 2*np.pi*kwargs['freq']
        zpar = 0
            
        if self.compliance is not None:
            ycomp = 1j*omega*self.compliance
            zpar += 1/ycomp

        if self.inertance is not None:
            yinert = 1/(1j*omega*self.inertance)
            zpar += 1/yinert

        if self.resistance is not None:
            yresist = 1/self.resistance
            zpar += 1/yresist

        zpar /= self.char_impedance
        tmx[1,0] += tmx[0,0]*zpar
        tmx[1,1] += tmx[0,1]*zpar
            
        return tmx

    def normalized_two_point_transfer_mx_at_freq(self, *args, **kwargs):
        """
        return the travelling wave matrix T of the
        duct section between from_pos to end_pos.

        [p_out, p_in]_from_pos = T [p_out, p_in]_to_pos
        """

        tmx = super().normalized_two_point_transfer_mx_at_freq( *args, **kwargs)
        try:
            freq = args[0]
        except IndexError:
            freq = kwargs['freq']
       
        omega = 2*np.pi*freq
        zpar = 0
            
        if self.compliance is not None:
            ycomp = 1j*omega*self.compliance
            zpar += 1/ycomp

        if self.inertance is not None:
            yinert = 1/(1j*omega*self.inertance)
            zpar += 1/yinert

        if self.resistance is not None:
            yresist = 1/self.resistance
            zpar += 1/yresist

        zpar /= self.char_impedance
 
        tmx[1,0] += tmx[0,0]/zpar
        tmx[1,1] += tmx[0,1]/zpar
            
        return tmx

###
# Functions that calculate impedance distances

def logzdiff(f,z1,z2):
    '''
    Calculates the distance in dB between the 
    absolute value of two curves
    '''
    z1db = 20*np.log10(np.abs(z1))
    z2db = 20*np.log10(np.abs(z2))
    dbdiff = z1db-z2db
    return np.sqrt(np.sum(dbdiff**2))
    
def pk_param(x,y,rad=1):
    allmax = argrelmax(y,order=rad)[0]
    idmax = np.ones(len(allmax))
    allmin = argrelmin(y,order=rad)[0]
    idmin = -np.ones(len(allmin))


    extu = np.concatenate((allmax,allmin))
    idxext = np.argsort(extu)
    allext = extu[idxext]
    allid = np.concatenate((idmax,idmin))[idxext]
    extdict = []

    dx = np.median(np.diff(x))
    for ii, (tp, xx) in enumerate(zip(allid,allext)):
        val = y[xx]
        
        # quadratic interpolation 
        pos = x[xx]
        if xx>0 and xx<len(x)-1:
            c = y[xx]
            b = (y[xx+1] - y[xx-1])/2
            a = (y[xx+1] + y[xx-1])/2 - c
            dxx = -b/2/a
            pos -= dxx*dx
            val = a*dxx**2 + b*dxx + c
        
        thisdict={'pos':pos,
                  'val':val,
                  'type':tp}

        sal_l = sal_r = np.max(allext)-np.min(allext)
        if ii>0:
            sal_l = (val-y[allext[ii-1]])*tp
        if ii<len(allext)-1:
            sal_r = (val-y[allext[ii+1]])*tp
        thisdict['sal']=min(sal_l,sal_r)
        extdict.append(thisdict)
    return extdict


def pkdiff(f,z1,z2,rad=40):
    '''
    Calculates frequency and magnitude differences 
    between peaks in the two impedances
    '''
    pks1 = pk_param(f,20*np.log10(np.abs(z1)),rad=rad)
    pks2 = pk_param(f,20*np.log10(np.abs(z2)),rad=rad)
    posdiff=np.zeros(len(pks1))
    valdiff=np.zeros(len(pks1))

    #pks1 = [pk for pk in pks1 if np.sign(pk['type'])==np.sign(typ)]
    #pks2 = [pk for pk in pks2 if np.sign(pk['type'])==np.sign(typ)]
    # find nearest
    for ii, pk1 in enumerate(pks1): 
        sel = np.array([ii for ii,pk2 in enumerate(pks2) if pk2['type'] == pk1['type']])
        pks2pos = np.array([pks2[ii]['pos'] for ii in sel])
        pk1pos = pk1['pos']
        idx = np.argmin(np.abs(pk1pos-pks2pos))
        pk2pos = pks2pos[idx]
        posdiff[ii] = (2*(pk1pos-pk2pos)/(pk1pos+pk2pos))
        valdiff[ii] = (pk1['val']-pks2[sel[idx]]['val'])*pk1['type']
    
    # return np.sqrt(np.sum(np.array(posdiff)**2)), np.sqrt(np.sum(np.array(valdiff)**2))
    return posdiff, valdiff


class FunctionFitter(object):
    """
    Function fitter is an interface to scipy.optimize.curve_fit

    Parameter list, initial values and bounds are stored 
    in the same object

    A parameter_mask can be defined in order to turn on or off 
    fitting for particular parameters. A value of False
    in param_mask means that the parameter will not be
    optimized and the starting value will be used


    Start with:
        ff = FunctionFitter(model_function, 
                            input_values, 
                            target_values, 
                            parameter_dict,
                            bounds)

    parameter_dict is a dict of form:
    {parameter_name: starting_parameter_value}

    Access initial settings:
    * ff.params: list with complete set of parameters (to be fitted and fixed)
    * ff.free_params: list with parameters to be fitted
    * ff.vals: list with initial parameter values
    * ff.cost_start: cost function
    * ff.y_start: evaluation of initial model
    * ff.residual_start: vector with residuals for each value of x
    * ff.x: abscissa
    * ff.bounds: get or set boundaries

    Access fitted model:
    * ff.popt: fitted parameters (ff.params for parameter names)
    * ff.pcov: fitted covariance
    * ff.y_fit: evaluation of fitted model
    * ff.residual: residual of fitted model
    * ff.cost: cost function  of fitted model
    """
    def __init__(self, func, x, yt, pardict=None, bounds=None):
        aspec = inspect.getfullargspec(func)
        defaults = aspec.defaults
        if pardict:
            self.params = list(pardict.keys())
            self._vals = list(pardict.values())
        else:
            self.params = aspec.args[-len(defaults):]
            self._vals = defaults

        self.param_mask = [True for pp in self.params]
        #self.vals = aspec.defaults
        self._bounds = None
        self.bounds = bounds
        #self.bounds = bounds
        self.func = func
        self.x = x
        self.yt = yt

    def cost_func(self, x,yt,yf):
        return 20*np.log10(np.abs(yt/yf))

    def arg_list_to_dict(self,args): 
        allargs = {p:v for p,v  in zip(self.params, self._vals)}
        inno = 0
        for p, v, m in zip(self.params, self._vals, self.param_mask):
            if m:
                allargs[p] = args[inno]
                inno+=1
        return allargs

    def __call__(self, x, *args):
        allargs = self.arg_list_to_dict(args)
        return self.func(x, **allargs)

    def get_parameter(self, param):
        ret = dict()
        pidx = self.params.index(param)
        ret['name'] = self.params[pidx]
        ret['initial_val'] = self._vals[pidx]
        ret['optimize'] = self.param_mask[pidx]
        try:
            ret['bounds'] = self.bounds[pidx]
        except TypeError:
            ret['bounds'] = None
        
        try:
            oidx = self.free_params.index(param)
            ret['fit_val'] = self.popt[oidx]
            ret['fit_stdev'] = np.sqrt(self.pcov[oidx,oidx])
        except (TypeError, NameError, AttributeError, ValueError):
            pass

        return ret
        
    def fit(self):
        """
        Run the optimization
        """
        if self.bounds:
            popt, pcov = curve_fit(self, self.x, self.yt, p0=self.vals,
                                   bounds=self.format_bounds())
        else:
            popt, pcov = curve_fit(self, self.x, self.yt, p0=self.vals)#,bounds=list(zip(*bounds)))
        self.popt = popt
        self.pcov = pcov


    def fit_custom(self):
        if self.bounds:
            logging.warn('Unimplemented')
            return
        else:
            def cost_func(params):
                allargs = self.arg_list_to_dict(params)
                yit = self.func(self.x, **allargs)
                return self.cost_func(self.x,self.yt,yit)
            res = leastsq(cost_func, self.vals, full_output=True)
            return res

    @property
    def bounds(self):
        if self._bounds is not None:
            return [v for v, m in zip(self._bounds,self.param_mask) if m]
        else:
            return None

    @bounds.setter
    def bounds(self, new_bounds):
        self._bounds=[]
        for pp in self.params:
            try:
                newb = [new_bounds[pp][0], new_bounds[pp][1]]
            except (KeyError, TypeError):
                newb = [-np.inf,np.inf]
            self._bounds.append(newb)

    def format_bounds(self):
        return list(zip(*self.bounds))
    # @bounds.setter
    # def bounds(self, intervals):
    #     try:
    #         for bb, param, default in zip(intervals, 
    #                                     self.params, 
    #                                     self._vals):
    #             try:
    #                 assert default>bb[0]
    #                 assert default<bb[1]
    #             except AssertionError:
    #                 logging.warn('Default value of {} ({}) exceeds boundaries ({} - {})'.format(param,default,bb[0],bb[1]))
    #     except TypeError:
    #         logging.warn('Boundaries not set')
    #         self._bounds = None

    @property
    def y_fit(self):
        return self(self.x, *self.popt)

    @property
    def y_start(self):
        allargs = {p:v for p,v  in zip(self.params, self._vals)}
        return self.func(self.x, **allargs)
    
    @property
    def residual(self):
        return self.y_fit-self.yt
    
    @property
    def residual_start(self):
        return self.y_start-self.yt
    
    @property
    def cost(self):
        return np.sqrt(np.sum(self.residual**2))

    @property
    def cost_start(self):
        return np.sqrt(np.sum(self.residual_start**2))
        
    @property
    def vals_fit(self):
        allargs = {p:v for p,v  in zip(self.params, self._vals)}
        inno = 0
        for p, v, m in zip(self.params, self._vals, self.param_mask):
            if m:
                allargs[p] = self.popt[inno]
                inno+=1
        return allargs

        
    @property
    def free_params(self):
        return [p for p, m in zip(self.params,self.param_mask) if m]
        
    @property
    def vals(self):
        return [v for v, m in zip(self._vals,self.param_mask) if m]
    
    @vals.setter
    def vals(self, vals):
        inno = 0
        for ii, m in enumerate(self.param_mask):
            if m:
                self._vals[ii] = vals[inno]
                inno+=1

    def apply_mask_dict(self,mask_dict):
        for k,v in mask_dict.items():
            self.switch_fitting(k,v)

    def switch_fitting(self,param,val):
        try:
            idx = self.params.index(param)
        except ValueError:
            logging.warn('apply mask: %s not in parameter list. Skipping'%param)
            return
        self.param_mask[idx]=bool(val)

    def fitting_on(self, param):
        self.switch_fitting(param,True)

    def fitting_off(self, param):
        self.switch_fitting(param,False)
        
def func_cf(x, **kwargs):
    loss_mult = kwargs['loss_mult']
    
    duct = psyn.Duct()
    
    ii=0
    while True:
        try:
            length = kwargs['length_%d'%ii]
            radius = kwargs['radius_%d'%ii]
        except KeyError:
            break
        try:
            lc = kwargs['lc_%d'%ii]
            duct.append_element(StraightDuctWithParallel(length=length,radius=radius,compliance=np.pi*(radius)**2*lc/340**2/1.2,loss_multiplier=loss_mult))
        except KeyError:
            duct.append_element(psyn.StraightDuct(length=length,radius=radius,loss_multiplier=loss_mult))
        ii+=1
            

    term = kwargs['term']
    if term<0.5:
        duct.set_termination(psyn.FlangedPiston(radius=radius))
    elif term>0.5:
        duct.set_termination(psyn.PerfectClosedEnd())
    
    return duct.get_input_impedance_at_freq(x)

def logf_cf(x,**kwargs):
    z = func_cf(x,**kwargs)
    return 20*np.log10(np.abs(z))


def optimise_impedance(f,z,argdict,mask=None):
    ff=FunctionFitter(logf_cf, f, 20*np.log10(np.abs(z)), pardict=argdict)
    if mask:
        ff.apply_mask_dict(mask)
    ff.fit()
    z_fit = func_cf(f,**ff.vals_fit) 
    fdiff,mdiff = pkdiff(f,z,z_fit)
    metrics = {"cost": logzdiff(f,z,z_fit),
               "peak_d_f": np.mean(fdiff),
               "peak_d_m": np.mean(mdiff)}
    return ff
