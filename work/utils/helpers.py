import numpy

def vcol(x):
    return x.reshape((x.size, 1))

def vrow(x):
    return x.reshape((1, x.size))

def compute_mu_C(D):
    """
    Compute the mean mu and covariance matrix C of the data matrix D of size (d, n)
    """
    mu = vcol(D.mean(1))
    C = ((D-mu) @ (D-mu).T) / float(D.shape[1])
    return mu, C

def compute_pca(D, m):
    """
    Compute the PCA projection matrix P of size (d, m) from the data matrix D of size (d, n)
    """
    mu, C = compute_mu_C(D)
    U, s, Vh = numpy.linalg.svd(C)
    P = U[:, 0:m]
    return P

def compute_Sb_Sw(D, L):
    Sb = 0
    Sw = 0
    muGlobal = vcol(D.mean(1))
    for i in numpy.unique(L):
        DCls = D[:, L == i]
        mu = vcol(DCls.mean(1))
        Sb += (mu - muGlobal) @ (mu - muGlobal).T * DCls.shape[1]
        Sw += (DCls - mu) @ (DCls - mu).T
    return Sb / D.shape[1], Sw / D.shape[1]


def apply_pca(P, D):
    return P.T @ D

def compute_lda_JointDiag(D, L, m):

    Sb, Sw = compute_Sb_Sw(D, L)

    U, s, _ = numpy.linalg.svd(Sw)
    P = numpy.dot(U * vrow(1.0/(s**0.5)), U.T)

    Sb2 = numpy.dot(P, numpy.dot(Sb, P.T))
    U2, s2, _ = numpy.linalg.svd(Sb2)

    P2 = U2[:, 0:m]
    return numpy.dot(P2.T, P).T

def apply_lda(U, D):
    return U.T @ D
