import numpy as np
from scipy.stats import norm

def vectorized_truncnorm(loc:float, scale:float, upperbound:float=np.inf, lowerbound:float=-np.inf, no_samples:int=100):
    """
    a vectorized version of a truncated normal distribution
    it first calculates the multiplier (estimate on how many samples have to be taken to have enough after applying bounds)
    then it iteratively samples and resamples until enough values within bounds are available
    """
    # calculate multiplier (how many distributions have to be generated to be able to make the cut with the given bounds)
    try:
        multiplier = 1 / (norm(loc, scale).cdf(upperbound) - norm(loc, scale).cdf(lowerbound))
    except Exception:
        multiplier = 1e-15 # if the cdfs both return 0
    # initialize storage:
    samples = []
    total_no_samples = 0
    while True:
        # number of samples pulled this round:
        iteration_no_samples = min( int( np.ceil( (no_samples-total_no_samples) * multiplier ) ), 100000) # limit is in order to not overflow the buffer
        # sample values:
        s = np.random.normal(loc, scale, iteration_no_samples )
        # sort out values that don't fit:
        s = s[(lowerbound <= s) * (s <= upperbound)]
        # if too many samples were generated:
        if  total_no_samples + s.shape[0] > no_samples:
            s = s[:no_samples-total_no_samples]
        # update storage:
        if s.shape[0] > 0:
            samples.append(s)
            total_no_samples += s.shape[0]
        if total_no_samples == no_samples:
            break
    samples = np.concatenate(samples)
    return samples