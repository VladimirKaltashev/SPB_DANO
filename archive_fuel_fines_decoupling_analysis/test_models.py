"""Compare absorbed FE to explicit dummies, and conditional Poisson to profiled GLM."""
import unittest
import numpy as np
import pandas as pd
import statsmodels.api as sm

from models import double_demean,linear_fit,conditional_poisson


class FixedEffectsTests(unittest.TestCase):
    def test_absorbed_coefficients_and_cluster_SE_equal_full_dummies(self):
        rng=np.random.default_rng(123);n,T=35,5
        ids=np.repeat(np.arange(n),T);months=np.tile(np.arange(T),n);regions=ids%5
        x=rng.normal(size=(n*T,2));a=rng.normal(size=n)
        y=x@np.array([.3,-.7])+a[ids]+months*.1+rng.normal(size=n*T)
        full=pd.DataFrame(x,columns=['x1','x2']).join(pd.get_dummies(pd.DataFrame({'client':ids.astype(str),'month':months.astype(str)}),drop_first=True,dtype=float))
        full=sm.add_constant(full);ols=sm.OLS(y,full).fit()
        robust=ols.get_robustcov_results(cov_type='cluster',groups=regions,use_correction=True,df_correction=True,use_t=True)
        rows,_=linear_fit(double_demean(y.reshape(n,T)).ravel(),double_demean(x.reshape(n,T,2)).reshape(-1,2),regions,n,len(full.columns),['x1','x2'],'test','test')
        np.testing.assert_allclose(rows.beta,ols.params[['x1','x2']],atol=1e-12)
        np.testing.assert_allclose(rows.SE,robust.bse[1:3],rtol=1e-10)
        np.testing.assert_allclose(rows.p_value,robust.pvalues[1:3],rtol=1e-10)

    def test_conditional_poisson_matches_explicit_poisson_FE(self):
        rng=np.random.default_rng(42);n,T=40,5
        ids=np.repeat(np.arange(n),T);months=np.tile(np.arange(T),n)
        x=rng.uniform(.1,3,n*T);mu=np.exp(-.4+.15*x+rng.normal(0,.4,n)[ids]+.08*months)
        y=rng.poisson(mu);y[ids==0]=0
        d=pd.DataFrame({'client_id':ids,'month':months,'region':ids%5,'fuel_liters_100':x,'fines_count':y})
        actual,audit=conditional_poisson(d)
        keep=d.groupby('client_id').fines_count.transform('sum').gt(0);small=d.loc[keep].reset_index(drop=True)
        X=small[['fuel_liters_100']].join(pd.get_dummies(small[['client_id','month']].astype(str),drop_first=True,dtype=float))
        X=sm.add_constant(X);expected=sm.GLM(small.fines_count,X,family=sm.families.Poisson()).fit()
        self.assertAlmostEqual(actual.beta.iloc[0],expected.params.fuel_liters_100,places=6)
        self.assertGreaterEqual(audit['removed_zero_total_clients'],1)


if __name__=='__main__':unittest.main()
