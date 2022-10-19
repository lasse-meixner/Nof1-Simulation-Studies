import numpy as np
from scipy.stats import ttest_1samp, t
from tqdm import tqdm
from statsmodels.api import MixedLM

try:
    import rpy2
    from rpy2.robjects.packages import STAP
    import rpy2.robjects.numpy2ri
    rpy2.robjects.numpy2ri.activate()

    # R Objects
    ar1_model_r = """
    library(nlme)

    ar1_model = function(
    y0,y1,sub,T,t,c0
    ){
    y0 <- as.vector(y0)
    y1 <- as.vector(y1)
    sub <- as.vector(sub)
    t <- as.vector(t)
    c0 <- as.vector(c0)
    m0 <- nlme::lme(y0 ~ T + t + c0, random = ~ 1 | sub, correlation = corAR1( form = ~ 1 | sub))
    m1 <- nlme::lme(y1 ~ T + t + c0, random = ~ 1 | sub, correlation = corAR1( form = ~ 1 | sub))
    p0_value <- summary(m0)$tTable[2,5]
    t0_value <- summary(m0)$tTable[2,4]
    t1_value <- summary(m1)$tTable[2,4]
    fe1_value <- summary(m1)$tTable[2,1]
    return(c(p0_value,t0_value,t1_value,fe1_value))
    }"""

    ar1_model = STAP(ar1_model_r, "ar1_model") # installs R script as package, use: ar1_model.ar1_model(y0,y1,sub,T,t,c0)

except:
    print("rpy2 not installed in Environment. Proceed with limited capabilities.")

import matplotlib.pyplot as plt
import seaborn as sns

class CrossOverExperiment():
    #
    def __init__(self,subjects,periods=2, base_effect=2, mu=0, tau=0.2, carryover=0, alpha_sd=2, epsilon_sd=1, error_type="ar1",**kwargs):
        """Create an instance of a CrossOverExperiment

        Parameters
        ----------
        subjects : int
            -
        periods : int, optional
            -, by default 2
        base_effect : int, optional
            The base effect (absence of treatment), by default 2
        mu : float, optional
            The treatment effect, by default 0
        tau : float, optional
            Linear time trend effect, by default 0.2
        carryover : float, optional
            The carryover effect, by default 0
        alpha_sd : int, optional
            Standard deviation of individual random effect, by default 2
        epsilon_sd : int, optional
            Standard deviation of idiosyncratic error, by default 1
        error_type : str, optional
            Can be either "ar1", "compound" or else. Currently only supports "ar1". By default "ar1"
        """
        self.subjects = subjects
        self.periods = periods
        self.base_effect = base_effect
        self.mu = mu
        self.tau = tau
        self.carryover = carryover
        self.alpha_sd = alpha_sd
        self.epsilon_sd = epsilon_sd
        self.error_type = error_type
        self.params = {
            "Subjects":self.subjects,
            "Periods":self.periods,
            "Base Effect":self.base_effect,
            "Mu":self.mu,
            "Tau":self.tau,
            "Carryover Effect":self.carryover,
            "Alpha Standard Deviation":self.alpha_sd,
            "Epsilon Standard Deviation":self.epsilon_sd,
            "Error Structure":self.error_type,
            **kwargs
        }
        self._isfit = False
    
    @property
    def subjects_(self):
        return self.subjects

    @property
    def periods_(self):
        return self.periods

    @property
    def params_(self):
        return self.params

    @staticmethod
    # TODO speeding this up???
    def gen_ar1_error(subjects,periods,rho=0.5):
        matrix = np.array([np.random.normal(0,1,subjects)])
        for p in range(periods-1):
            next = matrix[-1]*rho + np.random.normal(0,np.sqrt(1-(rho**2)),subjects)
            matrix = np.vstack([matrix,next])
        return matrix

    @staticmethod
    def gen_compound_error(subjects,periods,rho=0.5):
        first = np.array([np.random.normal(0,1,subjects)])
        rest = np.repeat(first,periods-1,axis=0)*rho + np.random.normal(0,np.sqrt(1-(rho**2)),(periods-1,subjects))
        matrix = np.vstack([first,rest])
        return matrix

    def get_critical_value(self,conf):
        """Get critical value for t distribution

        Args:
            conf (float): Type I error

        Returns:
            float: critical value for H0
        """
        critical_t = t.ppf(conf,self.subjects*(self.periods-1)) if self.periods>2 else t.ppf(conf,self.subjects-1) # df = n(k-1) (see Senn 2017)
        return critical_t
    
    def generate_data(self, return_for_t=True):
        """Generate data for one instance of the design. Can be called in loop to estimate experiment statistics.

        Args:
            return_for_t (bool, optional): If True, returns diffs for the ttest. Defaults to True.

        Returns:
            tuple: If return_for_t is False, returns outcome and feature vectors required to estimate MLM: (y,T,t,cO)
        """
        subj = np.array([np.arange(1,self.subjects+1)]).repeat(self.periods,axis=1).reshape(-1,2)
        t = np.array([np.arange(1,self.periods+1)]).repeat(self.subjects,axis=0).reshape(-1,2)
        c = np.ones(self.subjects*self.periods).reshape(-1,2)
        # leave features as 2d
        p1 = np.random.choice([1,0],size=int(self.subjects*self.periods/2))
        p2 = -p1 + 1
        T = np.c_[p1,p2]
        # carryover
        cO = np.hstack([[0],[1 if T.ravel()[x-1] and t.ravel()[x]!=1 else 0 for x in range(1,len(T.ravel()))]]).reshape(-1,2)
        # random subject intercept
        alpha = np.random.normal(loc=0,scale=self.alpha_sd,size=self.subjects).repeat(self.periods).reshape(-1,2)
        # select correct idiosyncratic error structure
        if self.error_type=="ar1":
            epsilon = self.gen_ar1_error(self.subjects,self.periods,self.params.get("rho",0.5)).reshape(-1,2)
        elif self.error_type=="compound":
            epsilon = self.gen_compound_error(self.subjects,self.periods,self.params.get("rho",0.5)).reshape(-1,2)
        else:
            epsilon = np.random.normal(loc=0,scale=self.epsilon_sd,size=self.subjects*self.periods).reshape(-1,2)
        # produce outcome for both h0 and h1
        y0 = c * self.base_effect + t*self.tau + alpha + epsilon # without Treatment effect (and carryover)
        y1 = c * self.base_effect + T*self.mu + t*self.tau + cO*self.carryover + alpha + epsilon
        # for t test: check for which block treatment has been allocated first or second period, then subtract accordingly
        if return_for_t:
            B_T = T[:,1] - T[:,0] # gives 1 if B_T, -1 if T_B
            diffs0 = (y0[:,1]-y0[:,0]) * B_T
            diffs1 = (y1[:,1]-y1[:,0]) * B_T
            return diffs0, diffs1
        else:
            return (y0.ravel(),y1.ravel()),subj.ravel(),T.ravel(),t.ravel(),cO.ravel() #(y0,y1)pTto

    def run_t_test(self,iterations):
        """Loops over experimental design data and computes the ttest pvalue over the paired differences.

        Args:
            iterations (int): Number of iterations in the loop
        """
        self.p_values = []
        self.null_statistics = []
        self.statistics = []
        self.estimates = []
        for i in tqdm(range(iterations)):
            d0,d1 = self.generate_data(return_for_t=True)
            t_stat0, p0 = ttest_1samp(d0,0,alternative="two-sided")
            t_stat1, p1 = ttest_1samp(d1,0,alternative="two-sided")
            self.p_values.append(p0)
            self.null_statistics.append(t_stat0)
            self.statistics.append(t_stat1)
            self.estimates.append(d1.mean())
        self._lastfit = "t_test"
        self._isfit = True

    def run_mixed_linear_model(self,iterations):
        """Loops over experimental design data and computes MLM estimates.

        Args:
            iterations (int): Number of iterations in the loop
        """
    
        self.p_values = []
        self.null_statistics = []
        self.statistics = []
        self.estimates = []
        if self.error_type=="ar1":
            (y0,y1),sub,T,t,cO = self.generate_data(return_for_t=False)
            for i in tqdm(range(iterations)):
                p0,t0,t1,fe1 = ar1_model.ar1_model(y0,y1,sub,T,t,cO)
                self.p_values.append(p0)
                self.null_statistics.append(t0)
                self.statistics.append(t1)
                self.estimates.append(fe1)
            self._lastfit = "MLM_ar1"

        elif self.error_type=="compound":
            raise ValueError("To be implemented.")
            # (y0,y1),sub,T,t,cO = self.generate_data(return_for_t=False)
            # for i in tqdm(range(iterations)):

        else:
            print("WARNING:\n Still running with statsmodels.")
            for i in tqdm(range(iterations)):
                (y0,y1),sub,T,t,cO = self.generate_data(return_for_t=False)
                mlm0 = MixedLM(y0,np.array([np.ones(len(T)),T,t,cO]).T,groups=sub).fit()
                mlm1 = MixedLM(y1,np.array([np.ones(len(T)),T,t,cO]).T,groups=sub).fit()
                self.p_values.append(mlm0.pvalues[1])
                self.null_statistics.append(mlm0.tvalues[1])
                self.statistics.append(mlm1.tvalues[1])
                self.estimates.append(mlm1.fe_params[1])
                self._lastfit = "MLM"
        self._isfit=True


    def get_results(self,conf=0.95):
        """Requires prior fitting.

        Args:
            conf (float): Critical level

        Returns:
            dict: Results dictionary
        """
        assert self._isfit==True
        assert conf>0 and conf<1

        #get df
        critical_t = self.get_critical_value(conf=conf)

        p_value = (np.array(self.p_values)>conf).sum()/len(self.p_values)
        bias = np.array(self.estimates).mean() - self.mu
        mse = ((np.array(self.estimates) - self.mu)**2).mean()
        power = (np.array(self.statistics)>critical_t).sum()/len(self.statistics)
        return {"p_value":p_value,"bias":bias,"mse":mse,"power":power}
    
    def plot_results(self,conf=0.95,**kwargs):
        f = sns.displot(self.null_statistics,**kwargs)
        plt.axvline(x=self.get_critical_value(conf),color="red",ls="--")
        plt.axvline(x=-self.get_critical_value(conf),color="red",ls="--")
        return f

        



class RCT():
    # generate docstring and give documentation of how the params in the kwargs have to be passed
    def __init__(self,subjects,base_effect=2, mu=0, alpha_sd=2, epsilon_sd=1,**kwargs):
        """Initialize RCT design with arg: subjects.

        Args:
            subjects (int): _description_

        Raises:
            ValueError: Requires arg: subjects to be even for both arms to be of equal size.
        """
        if subjects%2!=0:
            raise ValueError("Pass even number of subjects")
        self.subjects = subjects
        self.base_effect = base_effect
        self.mu = mu
        self.alpha_sd = alpha_sd
        self.epsilon_sd = epsilon_sd
        self.residual_sd = np.sqrt(alpha_sd**2 + epsilon_sd**2) # alpha and epsilon independent
        self.params = {
            "Subjects":self.subjects,
            "Base Effect":self.base_effect,
            "Mu":self.mu,
            "Alpha Standard Deviation":self.alpha_sd,
            "Epsilon Standard Deviation":self.epsilon_sd,
            **kwargs
        }
        self._isfit = False


    @property
    def subjects_(self):
        return self.subjects
    
    @property
    def params_(self):
        return self.params


    def generate_data(self):
        """Generate data for one instance of the design. Can be called in loop to estimate experiment statistics.

        Returns:
            tuple: outcome vector of treated, outcome vector of untreated
        """
        c = np.ones(self.subjects) * self.base_effect
        T = np.hstack([np.ones(int(self.subjects/2)),np.zeros(int(self.subjects/2))])
        alpha = np.random.normal(loc=0,scale=self.alpha_sd,size=self.subjects)
        epsilon = np.random.normal(loc=0,scale=self.epsilon_sd,size=self.subjects)
        y0 = c + alpha + epsilon
        y1 = c + self.mu*T + alpha + epsilon
        diffs0 = y0[T==1] - y0[T==0]
        diffs1 = y1[T==1] - y1[T==0]
        return diffs0,diffs1

    def run_t_test(self,iterations):
        """Loops over experimental design data and computes the ttest pvalue over the independent samples.

        Args:
            iterations (int): Number of iterations in the loop
        """
        self.p_values = []
        self.statistics = []
        self.estimates = []
        for i in tqdm(range(iterations)):
            d0,d1 = self.generate_data()
            t_stat0, p0 = ttest_1samp(d0,0,alternative="two-sided")
            t_stat1, p1 = ttest_1samp(d1,0,alternative="two-sided")
            self.p_values.append(p0)
            self.statistics.append(t_stat1)
            self.estimates.append(d1.mean())
        self._isfit = True
    
    def get_results(self,conf=0.95):
        """Requires self.run_t_test() to be called prior in order for self.pvalues to be set.

        Args:
            conf (float): Critical level

        Returns:
            dict: Results dictionary
        """
        assert self._isfit==True
        assert conf>0 and conf<1

        critical_t = t.ppf(conf,self.subjects-2) # df 

        p_value = (np.array(self.p_values)>conf).sum()/len(self.p_values)
        bias = np.array(self.estimates).mean() - self.mu
        mse = ((np.array(self.estimates) - self.mu)**2).mean()
        power = (np.array(self.statistics)>critical_t).sum()/len(self.statistics)
        return {"p_value":p_value,"bias":bias,"mse":mse,"power":power}


        

