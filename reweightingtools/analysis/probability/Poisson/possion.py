'''
ToDo: Documentation, imports
'''
def _cumulative_distribution(t, tau):
    return 1 - np.exp(- t/tau)

def _empirical_cumulative_distribution(esc_times):
    if type(esc_times) is list:
        esc_times = np.array(esc_times)
    ECDF = np.arange(esc_times.size)
    return ECDF/ECDF[-1]

def _curve_fit_ECDF(esc_times, ECDF, mu):
    from scipy.optimize import curve_fit
    tau, pcov = curve_fit(_cumulative_distribution,  np.sort(esc_times), ECDF, p0=mu)
    return tau[0]


def check_poisson(esc_times, plot=False, log=False):
    mu           = np.mean(esc_times)
    sigma        = np.std(esc_times)
    median       = np.median(esc_times)
    ECDF         = _empirical_cumulative_distribution(esc_times)
    tau          = _curve_fit_ECDF(esc_times, ECDF, mu)
    rate         = _esc_rate(esc_times)
    sigmaMu      = sigma/mu
    medianMulog2 = median/(mu*np.log(2))

    if plot:
        CDF = _cumulative_distribution
        plot_PoissonCeck(esc_times, ECDF, CDF, tau)
    if log:
        return  mu, sigma, median, ECDF, tau, rate, sigmaMu, medianMulog2 
         

