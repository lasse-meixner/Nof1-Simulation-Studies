from email.mime import base
import numpy as np
from scipy.stats import ttest_ind, ttest_1samp
from tqdm import tqdm
from statsmodels.api import MixedLM

# Objects

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
    
    def generate_data(self):
        pass

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
    
    def generate_data(self, return_for_t=True, **kwargs):
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
        if kwargs.get("type")=="ar1":
            epsilon = self.gen_ar1_error(self.subjects,self.periods,kwargs.get("rho")).reshape(-1,2)
        elif kwargs.get("type")=="compound":
            epsilon = self.gen_compound_error(self.subjects,self.periods,kwargs.get("rho")).reshape(-1,2)
        else:
            epsilon = np.random.normal(loc=0,scale=self.epsilon_sd,size=self.subjects*self.periods).reshape(-1,2)
        # produce outcome
        y = c * self.base_effect + T*self.mu + t*self.tau + cO*self.carryover + alpha + epsilon
        # for t test: check for which block treatment has been allocated first or second period, then subtract accordingly
        if return_for_t:
            B_T = T[:,1] - T[:,0] # gives 1 if B_T, -1 if T_B
            diffs = (y[:,1]-y[:,0]) * B_T
            return diffs
        else:
            return y.ravel(),subj.ravel(),T.ravel(),t.ravel(),cO.ravel() #ypTto

    def run_t_test(self,iterations):
        """Loops over experimental design data and computes the ttest pvalue over the paired differences.

        Args:
            iterations (int): Number of iterations in the loop
        """
        self.p_values = []
        for i in tqdm(range(iterations)):
            diffs = self.generate_data(return_for_t=True,**self.params)
            p = ttest_1samp(diffs,0,alternative="two-sided").pvalue
            self.p_values.append(p)
        self._lastfit = "t_test"
        self._isfit = True

    def run_mixed_linear_model(self,iterations):
        self.p_values = []
        if self.params.get("type")=="ar1":
            for i in tqdm(range(iterations)):
                y,sub,T,t,cO = self.generate_data(False,**self.params)
                mlm = MixedLM(y,np.array([np.ones(len(T)),T,t,cO]).T,groups=sub).fit(cov_type="hac-panel",cov_kwds={"groups":sub,"maxlags":1}).pvalues[1]
                self.p_values.append(p)

        elif self.params.get("type")=="compound":
            for i in tqdm(range(iterations)):
                y,sub,T,t,cO = self.generate_data(False,**self.params)
                p = MixedLM(y,np.array([np.ones(len(T)),T,t,cO]).T,groups=sub).fit(cov_type="clustered", cov_kewds={"groups":sub}).pvalues[1]
                self.p_values.append(p)

        else:
            for i in tqdm(range(iterations)):
                y,sub,T,t,cO = self.generate_data(False,**self.params)
                p = MixedLM(y,np.array([np.ones(len(T)),T,t,cO]).T,groups=sub).fit().pvalues[1]
                self.p_values.append(p)
        self._lastfit = "MLM"
        self._isfit=True


    def get_p_ratio(self,alpha):
        """Requires self.run_t_test() to be called prior in order for self.pvalues to be set.

        Args:
            alpha (_type_): Critical level

        Returns:
            _type_: Ratio of pvalues below critical level
        """
        assert self._isfit==True
        return (np.array(self.p_values)<alpha).sum()/len(self.p_values)
        



class RCT():
    # generate docstring and give documentation of how the params in the kwargs have to be passed
    def __init__(self,subjects,**kwargs):
        """Initialize RCT design with arg: subjects.

        Args:
            subjects (int): _description_

        Raises:
            ValueError: Requires arg: subjects to be even for both arms to be of equal size.
        """
        if subjects%2!=0:
            raise ValueError("Pass even number of subjects")
        self.subjects = subjects
        self.params = dict(kwargs)


    @property
    def subjects_(self):
        return self.subjects
    
    @property
    def params_(self):
        return self.params


    def generate_data(self,**kwargs):
        """Generate data for one instance of the design. Can be called in loop to estimate experiment statistics.

        Returns:
            tuple: outcome vector of treated, outcome vector of untreated
        """
        c = np.ones(self.subjects) * kwargs.get("base_effect")
        T = np.hstack([np.ones(int(self.subjects/2)),np.zeros(int(self.subjects/2))])
        alpha = np.random.normal(loc=0,scale=kwargs.get("alpha_sd"),size=self.subjects)
        epsilon = np.random.normal(loc=0,scale=kwargs.get("epsilon_sd"),size=self.subjects)
        y = c + kwargs.get("mu")*T + alpha + epsilon
        treated = y[T==1]
        untreated = y[T==0]
        return treated, untreated

    def run_t_test(self,iterations):
        """Loops over experimental design data and computes the ttest pvalue over the independent samples.

        Args:
            iterations (int): Number of iterations in the loop
        """
        self.pvalues = []
        for i in tqdm(range(iterations)):
            treated, untreated = self.generate_data(**self.params)
            p = ttest_ind(treated,untreated,equal_var=False,alternative="two-sided").pvalue
            self.pvalues.append(p)
    
    def get_p_ratio(self,alpha):
        """Requires self.run_t_test() to be called prior in order for self.pvalues to be set.

        Args:
            alpha (_type_): Critical level

        Returns:
            _type_: Ratio of pvalues below critical level
        """
        return (np.array(self.pvalues)<alpha).sum()/len(self.pvalues)


        

