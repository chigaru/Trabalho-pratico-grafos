import numpy as np
import scipy.stats as stats
import scipy.optimize as opt

npz_path = "data/processed/gplus_undirected_gcc.npz"
data = np.load(npz_path)
degrees = data["indptr"][1:] - data["indptr"][:-1]

def evaluate_models_at_kmin(k_min):
    x_tail = degrees[degrees >= k_min].astype(float)
    n_tail = len(x_tail)

    gamma = 1.0 + n_tail / np.sum(np.log(x_tail / (k_min - 0.5)))
    ll_pl = n_tail * np.log((gamma - 1.0) / (k_min - 0.5)) - gamma * np.sum(np.log(x_tail / (k_min - 0.5)))
    aic_pl = 2 * 1 - 2 * ll_pl

    sorted_x = np.sort(x_tail)
    emp_cdf = np.arange(1, n_tail + 1) / n_tail
    model_cdf_pl = 1.0 - (sorted_x / (k_min - 0.5)) ** (-(gamma - 1.0))
    ks_pl = np.max(np.abs(emp_cdf - model_cdf_pl))

    loc_exp = float(k_min)
    scale_exp = np.mean(x_tail - k_min)
    ll_exp = np.sum(stats.expon.logpdf(x_tail, loc=loc_exp, scale=scale_exp))
    aic_exp = 2 * 1 - 2 * ll_exp
    model_cdf_exp = stats.expon.cdf(sorted_x, loc=loc_exp, scale=scale_exp)
    ks_exp = np.max(np.abs(emp_cdf - model_cdf_exp))

    def neg_ll_ln(params):
        mu, s = params
        if s <= 0.01:
            return 1e12
        surv = stats.norm.sf((np.log(k_min) - mu) / s)
        if surv <= 1e-12:
            return 1e12
        log_pdf = stats.lognorm.logpdf(x_tail, s=s, scale=np.exp(mu))
        return -(np.sum(log_pdf) - n_tail * np.log(surv))

    init_mu = np.mean(np.log(x_tail))
    init_s = np.std(np.log(x_tail))
    res = opt.minimize(neg_ll_ln, [init_mu, init_s], method="Nelder-Mead")
    mu_ln, s_ln = res.x
    surv_opt = stats.norm.sf((np.log(k_min) - mu_ln) / s_ln)
    ll_ln = np.sum(stats.lognorm.logpdf(x_tail, s=s_ln, scale=np.exp(mu_ln))) - n_tail * np.log(surv_opt)
    aic_ln = 2 * 2 - 2 * ll_ln
    model_cdf_ln = (stats.lognorm.cdf(sorted_x, s=s_ln, scale=np.exp(mu_ln)) - (1.0 - surv_opt)) / surv_opt
    ks_ln = np.max(np.abs(emp_cdf - model_cdf_ln))

    print(f"=== Análise para k_min = {k_min} (n_tail = {n_tail:,}) ===")
    print(f"Power Law:   gamma = {gamma:.4f}, KS = {ks_pl:.4f}, LL = {ll_pl:.2f}, AIC = {aic_pl:.2f}")
    print(f"Exponencial: scale = {scale_exp:.2f}, KS = {ks_exp:.4f}, LL = {ll_exp:.2f}, AIC = {aic_exp:.2f}, R(PL-Exp) = {ll_pl - ll_exp:.2f}")
    print(f"Log-Normal:  mu = {mu_ln:.2f}, s = {s_ln:.2f}, KS = {ks_ln:.4f}, LL = {ll_ln:.2f}, AIC = {aic_ln:.2f}, R(PL-LN) = {ll_pl - ll_ln:.2f}")

evaluate_models_at_kmin(500)
evaluate_models_at_kmin(1500)
evaluate_models_at_kmin(4496)
