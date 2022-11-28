import numpy as np
import pickle
import scipy
from scipy import linalg

from sklearn import metrics
from scipy.stats import ortho_group
import sys

from utils import *
from sklearn.model_selection import RepeatedStratifiedKFold
from scipy.linalg import orth


def EProjSimplex_new(v, k=1, xorig=None, verbos=False):
    #
    #  Problem
    #  min  1/2 || x - v||^2
    #  s.t. x>=0, 1'x=1
    #
    n = v.shape[0]
    v = np.reshape(v, (n, 1))
    ft = 1
    v0 = v - np.mean(v) + k / n
    vmin = np.min(v0)
    vmax = np.max(v0)
    errold = np.linalg.norm(xorig-v)**2
    if vmin < 0:
        f = 1
        lambda_m = 0
        while abs(f) > 1e-7:
            v1 = v0 - lambda_m
            x = v1 * (v1 > 0)
            posidx = np.where(v1[:, 0] >= 0)[0]
            npos = len(posidx)
            g = -1 * npos
            f = np.sum(v1[posidx, 0]) - k
            lambda_m = lambda_m - f / g
            ft = ft + 1
            if ft > 200:
                x = v1 * (v1 > 0)
                break
        x = v1 * (v1 > 0)
    else:
        x = v0
    errnew = np.linalg.norm(x-v)**2

    return x


def GPI(A, B, s=0,Worig = None):
    m, k = B.shape
    if s == 0:
        alpha = max(abs(np.linalg.eigvals(A)))
    if s == 1:
        ww = np.random.randn(m, 1)
        for i in range(10):
            m1 = np.matmul(A, ww)
            q = m1 / np.linalg.norm(m1)
            ww = q.copy()
        alpha = abs(np.matmul(np.matmul(ww.transpose(), A), ww))

    err = 1
    t = 1
    W = orth(np.random.randn(B.shape[0], B.shape[1]))

    A_til = alpha * np.eye(m) - A
    obj = list()
    obj.append(np.trace(np.matmul(np.matmul(Worig.transpose(), A), Worig) - 2 * np.matmul(Worig.transpose(), B)))

    while t<1200 :
        M = 2 * np.matmul(A_til, W) + 2 * B
        u, s, vh = np.linalg.svd(M)
        #print(M.shape,u.shape,s.shape, vh.shape)
        u = u[:, 0:k]
        W = np.matmul(u, vh)
        obj.append(np.trace(np.matmul(np.matmul(W.transpose(), A), W) - 2 * np.matmul(W.transpose(), B)))
        if t >= 2:
            err = abs(obj[ - 1] - obj[ - 2])
        t = t + 1
        if err<1e-6:
            break

    return W

def solveTheta(d, Q, s,ThetaOrig=None):
    ThetaOrig = np.diag(ThetaOrig).reshape(ThetaOrig.shape[0],1)
    Theta_ = np.ones((d, 1)) / d
    V = np.ones((d, 1)) / d
    sigma2 = 0
    Sigma1 = np.zeros((d, 1))
    ite = 0
    niu = 1e1
    niumax = 1e7
    rho = 1.1
    Id = np.ones((d, 1))
    objvalue = list()

    objvalue.append(0)
    while ite < 1000:
        ite += 1
        objvalue.append(
            np.matmul(Theta_.transpose(), np.matmul(Q, Theta_))[0, 0] - np.matmul(Theta_.transpose(), s))
        if abs(objvalue[-1] - objvalue[-2]) < 1e-5 and abs(np.sum(Theta_) - 1)<0.0001 and ite>2:
            break
        E = 2 * Q + niu * np.eye(d) + niu * np.matmul(Id, Id.transpose())
        f = niu * V + niu * Id - sigma2 * Id - Sigma1 + s
        Theta_ = np.matmul(np.linalg.inv(E), f)
        V = Theta_ + 1 / niu * Sigma1
        V = (V > 0) * V
        Sigma1 = Sigma1 + niu * (Theta_ - V)
        sigma2 = sigma2 + niu * (np.matmul(Theta_.transpose(), Id) - 1)[0, 0]
        niu = rho * niu

    return Theta_

def run(X, Y, testX, testY, alpha=1, beta=1e-2, h=3, lamda1=1, lamda2=100):
    # X: training data, X[0]=MRI feature matrix, X[1] = PET feature matrix 
    # Y: training label
    n_views = len(X)
    d = np.ones((n_views, 1)) 
    Slsit = list()

    for i_view in range(n_views):
        St = scipy.spatial.distance.cdist(X[i_view].transpose(), X[i_view].transpose())
        St = St * St
        St = np.exp(-1 * St)
        St = St - np.diag(np.diag(St))
        Slsit.append(St)
        if i_view == 0:
            Ss = St.copy()
        else:
            Ss += St
    Ss /= n_views
    Ss = Ss - np.diag(np.diag(Ss))
    S = Ss.copy()

    Sk = (S.transpose() + S) / 2
    D_ = np.diag(np.sum(Sk, axis=1))
    LS = D_ - Sk

    W = list()
    Theta = list()
    blist = list()
    Htemp = list()
    b = np.zeros((Y.shape[0], 1)) 


    H = np.zeros([h, X[i_view].shape[1]])
    P = np.zeros([Y.shape[0], h])
    for i_view in range(n_views):
        blist.append(np.zeros((h, 1)))
        Htemp.append(H)
        W.append(np.zeros([X[i_view].shape[0], h]))
        Theta.append(np.diag(np.ones(X[i_view].shape[0]) / X[i_view].shape[0]))

    Ytemp = Y.copy()
    err = list()
    err.append(1e9)

    lossite = list()
    for ite in range(200):
        loss = [0, 0, 0, 0, 0]
        loss[0] = 0.5 * (np.linalg.norm(np.matmul(P, H) + np.matmul(b, np.ones((1, Y.shape[1]))  )- Y)**2)
        loss[1] = 0
        loss[2] = 0
        for i_view in range(n_views):
            loss[1] += 0.5  * (np.linalg.norm(
                np.matmul(np.matmul(W[i_view].transpose(), Theta[i_view]), X[i_view]) + np.matmul(blist[i_view],np.ones((1, X[i_view].shape[1])) )- H)**2)
            loss[2] += np.trace(np.matmul(np.matmul(
                np.matmul(np.matmul(np.matmul(np.matmul(W[i_view].transpose(), Theta[i_view]), X[i_view]), LS),
                          X[i_view].transpose()), Theta[i_view].transpose()), W[i_view]))
        loss[3] = (np.linalg.norm(P)**2)  / 2
        loss[4] = (np.linalg.norm(S - Ss)**2)
        lossall = loss[0] +alpha*loss[1] +lamda1*loss[2] + beta * loss[3] + lamda2*loss[4]
        err.append(lossall)
        if abs(err[ite + 1] - err[ite]) < 1e-3:
            break

        # update Wv, v=1,2,...,V
        for i_view in range(n_views):
            A = np.matmul(np.matmul(np.matmul(Theta[i_view], np.matmul(X[i_view],  np.eye(S.shape[0]) + 2*lamda1/alpha* LS)),
                                    X[i_view].transpose()), Theta[i_view].transpose())
            B = np.matmul(Theta[i_view], np.matmul(X[i_view], Htemp[i_view].transpose()))
            W[i_view] = GPI(A, B,Worig = W[i_view]*1)

        # update Theta v=1,2,...,V
        for i_view in range(n_views):
            Q = np.matmul(np.matmul(X[i_view], np.eye(S.shape[0]) + 2*lamda1/alpha * LS), X[i_view].transpose()) * np.matmul(
                W[i_view], W[i_view].transpose())
            s = np.diag(2 * np.matmul(np.matmul(X[i_view], Htemp[i_view].transpose()), W[i_view].transpose()))
            s = s.reshape(s.shape[0], 1)
            Theta[i_view] = solveTheta(s.shape[0], Q, s, ThetaOrig=Theta[i_view])
            Theta[i_view] = Theta[i_view].reshape(Theta[i_view].shape[0])
            Theta[i_view] = np.diag(Theta[i_view])

        # update bv v=1,2,...,V
        for i_view in range(n_views):
            blist[i_view] = (np.matmul(H, np.ones((X[i_view].shape[1], 1)) ) - np.matmul(np.matmul(np.matmul(W[i_view].transpose(), Theta[i_view]), X[i_view]), np.ones((X[i_view].shape[1], 1))  ))/X[i_view].shape[1]
            Htemp[i_view] = H - np.matmul(blist[i_view],np.ones((1, X[i_view].shape[1])) )



        # update H
        H1 = alpha * n_views * np.eye(P.shape[1]) + np.matmul(P.transpose(), P)
        H2 = np.matmul(P.transpose(), Ytemp)
        for i_view in range(n_views):
            H2 = H2 + alpha * (np.matmul(np.matmul(W[i_view].transpose(), Theta[i_view]), X[i_view]) + np.matmul(blist[i_view], np.ones((1, X[i_view].shape[1])) ))
        H = np.matmul(np.linalg.inv(H1), H2)

        for i_view in range(n_views):
            Htemp[i_view] = H - np.matmul(blist[i_view],np.ones((1, X[i_view].shape[1])))



        # update P
        P = np.matmul(np.matmul(Ytemp, H.transpose()),
                      np.linalg.inv(np.matmul(H, H.transpose()) + beta * np.eye(H.shape[0])))

        # update b
        b = (np.matmul(Y, np.ones((Y.shape[1], 1))) - np.matmul(np.matmul(P, H), np.ones((Y.shape[1], 1)))  ) / Y.shape[1]
        Ytemp = Y- np.matmul(b, np.ones((1, Y.shape[1])))

        # Update S
        for i_view in range(n_views):
            temp = np.matmul(np.matmul(W[i_view].transpose(), Theta[i_view]), X[i_view])
            if i_view == 0:
                tE = scipy.spatial.distance.cdist(temp.transpose(), temp.transpose())
                E = tE * tE
            else:
                tE = scipy.spatial.distance.cdist(temp.transpose(), temp.transpose())
                E += tE * tE
        for i_row in range(S.shape[0]):
            if lamda2 != 0:
                vtemp = Ss[i_row, :] - lamda1 / lamda2  * E[i_row, :]/4
            else:
                vtemp = -1 * lamda1 * E[i_row, :]/4
            ttt = EProjSimplex_new(vtemp,  k=1, xorig=S[i_row, :])
            tttr = ttt.reshape(ttt.shape[0])
            S[i_row, :] = tttr.copy()
        Sk = (S.transpose() + S) / 2
        D_ = np.diag(np.sum(Sk, axis=1))
        LS = D_ - Sk



    for i_view in range(n_views):
        if i_view == 0:
            Hall  = np.matmul(W[i_view].transpose(), np.matmul(Theta[i_view], testX[i_view])) + np.matmul(blist[i_view], np.ones((1, testX[i_view].shape[1] )))
        else:
            Hall += np.matmul(W[i_view].transpose(), np.matmul(Theta[i_view], testX[i_view])) + np.matmul(blist[i_view], np.ones((1, testX[i_view].shape[1] )))


    preds = np.matmul(P,Hall/n_views) + np.matmul(b, np.ones((1, testX[i_view].shape[1] )))
    prob = scipy.special.softmax(preds.transpose(), axis = 1)[:,1]
    ans1 = evaluate_cls(np.argmax(testY.transpose(), axis=1), np.argmax(preds.transpose(), axis=1), prob) 


    return ans1

