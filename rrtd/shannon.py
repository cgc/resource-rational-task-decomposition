import numpy as np
# thinkin bout claude

wikiP = np.array([9/25, 12/25, 4/25])
wikiQ = np.ones(3)/3

def symkldiv(P, Q):
    return kldiv(P, Q) + kldiv(Q, P)

def kldiv(P, Q):
    '''
    > The Kullback–Leibler divergence is then interpreted as the average
    > difference of the number of bits required for encoding samples of
    > P using a code optimized for Q rather than one optimized for P.
    https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence

    >>> assert np.isclose(kldiv(wikiP, wikiQ), 0.0852996)
    >>> assert np.isclose(kldiv(wikiQ, wikiP), 0.097455)
    '''
    #return P @ (np.log(P) - np.log(Q))
    return crossent(P, Q) - crossent(P, P)

def jsdiv(*Ps):
    '''
    The Jensen-Shannon divergence, for arbitrary number of distributions.
    using example from https://stats.stackexchange.com/questions/29578/jensen-shannon-divergence-calculation-for-3-prob-distributions-is-this-ok
    >>> assert np.isclose(jsdiv(np.array([1/2,1/2,0]),np.array([0,1/10,9/10]),np.ones(3)/3), 0.378889)
    '''
    M = np.mean(np.stack(Ps), axis=0)
    return sum(kldiv(P, M)/len(Ps) for P in Ps)

def crossent(P, Q=None):
    '''
    >>> assert np.isclose(crossent(wikiP, wikiQ) - crossent(wikiP), kldiv(wikiP, wikiQ))
    '''
    if Q is None:
        Q = P
    logQ = np.log(Q)
    valid = ~np.isneginf(logQ)
    return -P[valid] @ logQ[valid]

def softmax(z):
    z = z - np.max(z) # please never do -= on this line again...
    e = np.exp(z)
    return e / e.sum()

def conditional_entropy(PQ):
    '''
    Conditional entropy of Q given P, H(Q|P)
    Need to specify a joint over P, Q, p(P, Q)

    an example from https://math.stackexchange.com/questions/848158/when-it-is-conditional-entropy-minimized

    >>> assert np.isclose(conditional_entropy(np.array([
    ...     [1/8, 1/16, 1/32, 1/32],
    ...     [1/16, 1/8, 1/32, 1/32],
    ...     [1/16, 1/16, 1/16, 1/16],
    ...     [1/4, 0, 0, 0],
    ... ])), 11/8 * np.log(2))
    '''
    assert np.isclose(PQ.sum(), 1)
    P = PQ.sum(1, keepdims=True) # get the marginal
    with np.errstate(invalid='ignore'):
        return -np.nansum((PQ * (np.log(PQ) - np.log(P))))


PQ = np.array([
    [1/8, 2/8],
    [1/8, 1/16],
    [3/8, 1/16],
])

def mutual_information(PQ):
    '''
    >>> byhand = np.sum(PQ * np.log(PQ/(PQ.sum(1, keepdims=True)*PQ.sum(0, keepdims=True))))
    >>> assert np.allclose(mutual_information(PQ), byhand)
    >>> assert np.allclose(mutual_information(PQ), mutual_information(PQ.T)) # symmetric
    '''
    return crossent(PQ.sum(0)) - conditional_entropy(PQ)

def mutual_information2(PQ):
    '''
    >>> assert np.allclose(mutual_information2(PQ), mutual_information(PQ))
    '''
    return kldiv(
        PQ.flatten(),
        (PQ.sum(1, keepdims=True) * PQ.sum(0, keepdims=True)).flatten(),
    )
