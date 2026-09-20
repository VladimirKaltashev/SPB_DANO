"""Balanced two-way fixed effects and one conditional Poisson robustness."""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp
from scipy.stats import t


def double_demean(array):
    """Input shape clients × months [× covariates], balanced calendar panel."""
    return array-array.mean(axis=1,keepdims=True)-array.mean(axis=0,keepdims=True)+array.mean(axis=(0,1),keepdims=True)


def linear_fit(y, x, clusters, n_clients, full_rank, names, model, relationship):
    n,k=x.shape
    assert np.linalg.matrix_rank(x)==k and np.isfinite(x).all() and np.isfinite(y).all()
    beta=np.linalg.lstsq(x,y,rcond=None)[0]
    residual=y-x@beta
    bread=np.linalg.inv(x.T@x)
    codes,unique=pd.factorize(clusters)
    scores=np.zeros((len(unique),k));np.add.at(scores,codes,x*residual[:,None])
    g=len(unique)
    assert n>full_rank and g>1
    correction=g/(g-1)*(n-1)/(n-full_rank)
    cov=bread@(scores.T@scores)@bread*correction
    se=np.sqrt(np.diag(cov));critical=t.ppf(.975,g-1)
    rows=[]
    for i,name in enumerate(names):
        rows.append(dict(model=model,relationship=relationship,term=name,beta=beta[i],SE=se[i],
            p_value=2*t.sf(abs(beta[i]/se[i]),g-1),ci95_low=beta[i]-critical*se[i],ci95_high=beta[i]+critical*se[i],
            n_clients=n_clients,N=n,n_clusters=g,cluster='region' if g==16 else 'client',
            full_parameter_rank=full_rank,CR1_correction=correction,
            R_squared=1-np.sum(residual**2)/np.sum((y-y.mean())**2)))
    return pd.DataFrame(rows),residual


def linear_models(data):
    clients=data.client_id.nunique();months=data.month.nunique()
    assert data.groupby('client_id').size().eq(months).all()
    y=data.fines_count.to_numpy().reshape(clients,months)
    x=data[['fuel_liters_100','price_shock','has_fuel_purchase']].to_numpy().reshape(clients,months,3)
    yt=double_demean(y).ravel();xt=double_demean(x).reshape(-1,3)
    between=data.groupby('client_id',sort=True).agg(fuel_liters=('fuel_liters','mean'),fines_count=('fines_count','mean'),region=('region','first'))
    b,_=linear_fit(between.fines_count.to_numpy(),np.column_stack([np.ones(clients),between.fuel_liters.to_numpy()/100]),
        between.region,clients,2,['const','fuel_liters_100'],'between','between client averages')
    primary=[b];alternative=[]
    for model,cols,names in [('within',[0],['fuel_liters_100']),
                              ('within_price',[0,1],['fuel_liters_100','price_shock']),
                              ('extensive',[2],['has_fuel_purchase'])]:
        rank=clients+months-1+len(cols)
        for cluster,target in [('region',primary),('client_id',alternative)]:
            rows,residual=linear_fit(yt,xt[:,cols],data[cluster],clients,rank,names,model,'client FE + month FE')
            rows['cluster']='region' if cluster=='region' else 'client'
            rows['R_squared_definition']='incremental R2 after client and month FE'
            target.append(rows)
        assert np.allclose(residual.reshape(clients,months).mean(axis=1),0,atol=1e-10)
        assert np.allclose(residual.reshape(clients,months).mean(axis=0),0,atol=1e-10)
    audit=dict(price_shock_twfe_sd=float(xt[:,1].std()),fuel_price_twfe_correlation=float(np.corrcoef(xt[:,0],xt[:,1])[0,1]),
        raw_condition_number=float(np.linalg.cond(xt[:,:2])),scaled_condition_number=float(np.linalg.cond(xt[:,:2]/xt[:,:2].std(axis=0))),
        fuel_twfe_sd_liters=float(xt[:,0].std()*100),fines_twfe_sd=float(yt.std()),
        price_identified=bool(np.linalg.matrix_rank(xt[:,:2])==2))
    centered=pd.DataFrame(dict(client_id=data.client_id,month=data.month,
        fuel_client_demeaned=(x[:,:,0]-x[:,:,0].mean(axis=1,keepdims=True)).ravel()*100,
        fines_client_demeaned=(y-y.mean(axis=1,keepdims=True)).ravel(),
        fuel_twfe_residual=xt[:,0]*100,fines_twfe_residual=yt))
    return pd.concat(primary,ignore_index=True),pd.concat(alternative,ignore_index=True),between.reset_index(),centered,audit


def conditional_poisson(data):
    """Conditional likelihood given each client's total count; vectorized exact score/Hessian."""
    nclient=data.client_id.nunique();nt=data.month.nunique()
    y=data.fines_count.to_numpy().reshape(nclient,nt)
    fuel=data.fuel_liters_100.to_numpy().reshape(nclient,nt)
    total=y.sum(axis=1);keep=total>0
    y=y[keep];fuel=fuel[keep];total=total[keep]
    # Client-constant shifts cancel out of the conditional likelihood; center for stability.
    fuel=fuel-fuel.mean(axis=1,keepdims=True)
    month=np.broadcast_to(np.eye(nt)[:,1:],(len(y),nt,nt-1))
    x=np.concatenate([fuel[:,:,None],month],axis=2)
    n=len(y)*nt;k=x.shape[2]
    def components(beta):
        eta=np.einsum('itk,k->it',x,beta);lse=logsumexp(eta,axis=1)
        p=np.exp(eta-lse[:,None]);mu=total[:,None]*p
        residual=y-mu
        scores=np.einsum('itk,it->ik',x,residual)
        means=np.einsum('it,itk->ik',p,x)
        info=np.einsum('it,itk,itl->kl',mu,x,x)-np.einsum('i,ik,il->kl',total,means,means)
        nll=-np.sum(y*eta)+total@lse
        return nll,-scores.sum(axis=0),info,scores
    result=minimize(lambda b:components(b)[0],np.zeros(k),jac=lambda b:components(b)[1],
        hess=lambda b:components(b)[2],method='trust-exact',options={'gtol':1e-6,'maxiter':100})
    _,gradient,info,scores=components(result.x)
    if np.linalg.norm(gradient,np.inf)>1e-5:
        raise ValueError(f'Conditional Poisson did not converge: {result.message}, gradient={gradient}')
    regions=data.groupby('client_id',sort=True).region.first().to_numpy()[keep]
    codes,groups=pd.factorize(regions);cluster_scores=np.zeros((len(groups),k));np.add.at(cluster_scores,codes,scores)
    bread=np.linalg.inv(info);g=len(groups)
    cov=bread@(cluster_scores.T@cluster_scores)@bread*g/(g-1)
    se=np.sqrt(np.diag(cov));critical=t.ppf(.975,g-1)
    beta=result.x[0]
    row=dict(model='conditional_poisson',relationship='client conditional Poisson + month FE',term='fuel_liters_100',
        beta=beta,SE=se[0],p_value=2*t.sf(abs(beta/se[0]),g-1),ci95_low=beta-critical*se[0],ci95_high=beta+critical*se[0],
        n_clients=len(y),N=n,n_clusters=g,cluster='region',IRR_per_100_liters=np.exp(beta),
        IRR_ci95_low=np.exp(beta-critical*se[0]),IRR_ci95_high=np.exp(beta+critical*se[0]))
    check=dict(converged_gradient_inf=float(np.linalg.norm(gradient,np.inf)),
        removed_zero_total_clients=int((~keep).sum()),retained_clients=len(y),
        conditional_loglike=float(-components(result.x)[0]),min_info_eigenvalue=float(np.linalg.eigvalsh(info).min()),
        covariance_correction='G/(G-1), conditional score by region, t(G-1)',
        coefficients=result.x.tolist())
    return pd.DataFrame([row]),check
