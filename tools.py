import numpy as np
# import settings
import torch
import matplotlib.pyplot as plt
# from settings import a, b,c
from scipy.stats import norm
def build_G_from_S(S, k, eplison):
    # k = 3
# def build_G_from_S(S, k):
    # S: similarity matrix
    # k: number of nearest neighbors
    # G: graph
    # G = torch.ones(S.shape).cuda() * -1.5
    # # G = torch.zeros(S.shape).cuda()
    # G_ = S
    # # G_ = S_
        # S: similarity matrix
    # k: number of nearest neighbors
    # G: graph
    G = torch.ones(S.shape).cuda() * -1.5
    # G = torch.zeros(S.shape).cuda()
    eplison = torch.tensor(eplison, dtype=S.dtype)

    G_ = torch.where(S > eplison
                     , S.cuda(), torch.tensor(0, dtype=torch.float).cuda()).cuda()
    # G_ = torch.where(S > settings.threshold, S.to(torch.float), float(-1.50)).cuda()
    # G_ = S_
    # S_ = torch.where(S > 0 and S > settings.rthreshold, 1, S).cuda()
    # S_ = torch.where(S < 0 and S < settings.lthreshold, -1, S_).cuda()
    # G_= torch.where(S_ != 0, S, 0).cuda()
    for i in range(G_.shape[0]):
        idx = torch.argsort(-G_[i])[:k]
        G[i][idx] = G_[i][idx]
    del G_
    torch.cuda.empty_cache()
    return G
    # # S_ = torch.where(S > 0 and S > settings.rthreshold, 1, S).cuda()
    # # S_ = torch.where(S < 0 and S < settings.lthreshold, -1, S_).cuda()
    # # G_= torch.where(S_ != 0, S, 0).cuda()
    # for i in range(G_.shape[0]):
    #     idx = torch.argsort(-G_[i])[:k]
    #     G[i][idx] = G_[i][idx]
    # del G_
    # torch.cuda.empty_cache()
    # return G
    G = torch.ones(S.shape).cuda() * -1.5
    # G = torch.zeros(S.shape).cuda()

    # G_ = torch.where(S > settings.threshold, S, -1.5).cuda()
    G_ = S.cuda()

    # G_ = torch.where(S==1, S, -1.5).cuda()
    # G = G_

    # G_ = S_
    # S_ = torch.where(S > 0 and S > settings.rthreshold, 1, S).cuda()
    # S_ = torch.where(S < 0 and S < settings.lthreshold, -1, S_).cuda()
    # G_= torch.where(S_ != 0, S, 0).cuda()
    for i in range(G_.shape[0]):
        idx = torch.argsort(-G_[i])[:k]
        G[i][idx] = G_[i][idx]
    del G_
    torch.cuda.empty_cache()
    return G

def generate_robust_S(s, alpha=2, beta=2):
    # G: graph
    # S: similarity matrix
    # alpha: parameter for postive robustness
    # beta: parameter for negative robustness
    # S_robust: robust similarity matrix

    S = s

    # find maximum count of cosine distance

    max_count = 0
    max_cos = 0

    interval = 1/1000
    cur = -1.0
    for i in range(2000):
        cur_cnt = np.sum((S>cur) & (S<cur+interval))
        if max_count < cur_cnt:
            max_count = cur_cnt
            max_cos = cur
        cur += interval

    # split positive and negative similarity matrix

    flat_S = S.reshape((-1,1))
    left = flat_S[np.where(flat_S <= max_cos)[0]]
    right = flat_S[np.where(flat_S >= max_cos)[0]]

    # reconstruct
    left = np.concatenate([left, 2 * max_cos - left])
    right = np.concatenate([max_cos-np.maximum(right-max_cos,max_cos-right), right])

    # fit to gaussian distribution
    left_mean, left_std = norm.fit(left)
    right_mean, right_std = norm.fit(right)

    print('left mean: ', left_mean)
    print('left std: ', left_std)
    print('threshold:',left_mean-alpha*left_std)
    print('right mean: ', right_mean)
    print('right std: ', right_std)
    print('threshold:',right_mean+beta*right_std)

    S = np.where(S >= right_mean + beta * right_std, 1, S)
    # S = np.where(S <= left_mean - alpha * left_std, -1, S)
    S = np.where(S <= left_mean - alpha * left_std, 0, S)
    # 不显示 y轴
    plt.gca().axes.get_yaxis().set_visible(False)
        # draw the histogram 
    plt.hist(left, bins=10000, density=True, alpha=0.6, color='g')
    plt.savefig('left.svg', format='svg')
    plt.close()
    # x_major_locator=MultipleLocator(0.5)
    # ax=plt.gca()
    # ax.xaxis.set_major_locator(x_major_locator)
    plt.hist(right, bins=10000, density=True, alpha=0.6, color='r')
    plt.savefig('right.svg', format='svg')
    plt.close()
    # x_major_locator=MultipleLocator(0.5)
    # ax=plt.gca()
    # ax.xaxis.set_major_locator(x_major_locator)
    plt.hist(flat_S, bins=10000, density=True, alpha=0.6, color='b')
    plt.savefig('flat_S.svg', format='svg')
    plt.close()
    return S

    # generate robust similarity matrix

    # show the fitting result
    # plt.hist(left, bins=100, density=True, alpha=0.6, color='g')
    # plt.hist(right, bins=100, density=True, alpha=0.6, color='r')
    # plt.show()
    # plt.close()

def edge_list(G):
    # list_e = []
    # cla = []
    # N = len(G[0])
    # for i in range(N):
    #     for j in range(N):
    #         if G[i][j] != -1.5:
    #             list_e.append(i)
    #             cla.append(j)
    #
    # res = []
    # res.append(list_e)
    # res.append(cla)
    # list_e, cla = zip(*[(i, j) for i in range(N) for j in range(N) if G[i][j] != -1.5])
    #
    # # Package results into a single list
    # res = [list(list_e), list(cla)]
    mask = G >0#!= 0#-1.5

    # Use the mask to find the indices
    list_e, cla = mask.nonzero(as_tuple=True)

    # Convert indices to lists if needed (they are returned as tensors)
    list_e = list_e.tolist()
    cla = cla.tolist()

    # Combine the lists into a final result
    res = [list_e, cla]
    return res





# def cal_sim(S_I, S_T, S_F):
#     return settings.a * S_I + settings.b * S_T + settings.c * (S_F + S_F.t()) / 2