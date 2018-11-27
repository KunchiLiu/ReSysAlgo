# -*- coding=utf-8 -*-
import random
from operator import itemgetter

import math

from usersim import Recommend

# 均方根误差 RMSE, 书本有误, recoders[i] = [u, i, rui, pui]
def RMSE(recoders):
    return math.sqrt(sum([(rui-pui)*(rui-pui) for u, i, rui, pui in recoders])) / float(len(recoders))

# 平均绝对误差 MAE
def MAE(recoders):
    return sum([abs(rui-pui) for u, i, rui, pui in recoders]) / float(len(recoders))

# TopN推荐预测准确率 = 准确率（precosion）/召回率（recall）
# 准确率和召回率是广泛用于信息检索和统计学分类领域的两个度量值，用来评价结果的质量。
# 其中精度是检索出相关文档数与检索出的文档总数的比率，衡量的是检索系统的查准率；
# 召回率是指检索出的相关文档数和文档库中所有的相关文档数的比率，衡量的是检索系统的查全率。
# 1. 正确率 = 提取出的正确信息条数 /  提取出的信息条数
# 2. 召回率 = 提取出的正确信息条数 /  样本中的信息条数
# https://blog.csdn.net/xwd18280820053/article/details/70674256
def PrecisionRecall(test, N):
    hit = 0
    n_recall = 0
    n_precision = 0
    for user, items in test.items():
        rank = Recommend(user, N)
        hit += len(rank & items)
        n_recall += len(items)
        n_precision += N
    return [hit / (1.0 * n_recall), hit / (1.0 * n_precision)]

# 基尼系数
def GiniIndex(p):
    j = 1
    n = len(p)
    G = 0
    for item, weight in sorted(p.items, key=itemgetter(1)):
        G += (2 * j - n - 1) * weight
    return G / float(n - 1)

# 马太效应， 比如新浪的热搜
# 长尾效应

# 将数据集随机分训练集和测试集
# 感觉seed没啥用，注意的地方是k应该属于[0,M-2]
def SplitData(data, M, k, seed):
    test = []
    train = []
    random.seed(seed)
    for user, items in data.items():
        if random.randint(0,M) == k:
            print(user, items)
            test.append([user,items])
        else:
            print(user, items)
            train.append([user,items])
    return train, test

# 召回率
def Recall(train, test, N):
    hit = 0
    all = 0
    for user in train.keys():
        tu = test[user]
        rank = Recommend(user, N)
        #rank = GetRecommendation(user, N)
        for item, pui in rank:
            if item in tu:
                hit += 1
        all += len(tu)
    return hit / (all * 1.0)

# 准确率
def Precision(train, test, N):
    hit = 0
    all = 0
    for user in train.keys():
        tu = test[user]
        rank = Recommend(user, N)
        # rank = GetRecommendation(user, N)
        for item, pui in rank:
            if item in tu:
                hit += 1
        all += N
    return hit / (all * 1.0)

# Coverage
def Coverage(train, test, N):
    # set类似于list，但不能重复
    recommend_items = set()
    all_items = set()
    for user in train.keys():
        for item in train[user].keys():
            all_items.add(item)
        rank = Recommend(user, N)
        for item, pui in rank:
            recommend_items.add(item)
    return len(recommend_items) / len(all_items)

# 物品流行度分布满足长尾分布，在取对数后，流行度的平均值更加稳定
def Popularity(train, test, N):
    item_popularity = dict()
    for user, items in train.items():
        for item in items.keys():
            if item not in item_popularity:
                item_popularity[item] = 0
            item_popularity[item] += 1
    ret = 0
    n = 0
    for user in train.keys():
        rank = Recommend(user, N)
        for item, pui in rank:
            ret = math.log(1 + item_popularity[item])
            n += 1
    ret /= n*1.0
    return ret

# 基于上述用户相似度公式的UserCF算法记为User-IIF算法
def UserSimilarity(train):
    # build inverse table for item_users，主要是降低时间复杂度
    item_users = dict()
    for user, items in train.items():
        for i in items.keys():
            if i not in item_users:
                item_users[i] = set()
            item_users[i].add(user)

    # calculate co-rate items between users
    C = dict()
    N = dict()
    for i, users in item_users.items():
        for u in users:
            N[u] += 1 # 字典，用户：用户购买的商品数
            for v in users:
                if u == v:
                    continue
                C[u, v] = 1 / math.log(1 + len(users))

    # calculate finial similarity matrix W
    W = dict()
    for co_user, cuv in C.items():
            W[co_user] = cuv / math.sqrt(N[co_user[0]] * N[co_user[1]])
    return  W

# ItemCF
def ItemSimilarity(train):
    # calculate co-rated users between items
    C = dict()
    N = dict()
    for u, items in train.items():
        for i in items:
            N[i] += 1
            for j in items:
                if i == j:
                    continue
                C[i][j] += 1
    # calculate finial similarity matrix W
    W = dict()
    for i, related_items in C.items():
        for j, cij in related_items.items():
            W[i][j] = cij / math.sqrt(N[i] * N[j])
    return W

def Recommendation(train, user_id, W, K):
    rank = dict()
    ru = train[user_id]
    for i, pi in ru.items():
        for j, wj in sorted(W[i].items(), key=itemgetter(1), reverse=True)[0:K]:
            if j in ru:
                continue
            rank[j].weight += pi * wj
            rank[j].reason[i] = pi * wj
    return rank

# 实现基于随机游走的PersonalRank算法
def PersonalRank(G, alpha, root):
    rank = dict()
    rank = {x:0 for x in G.keys()}
    rank[root] = 1
    for k in range(20):
        tmp = {x:0 for x in G.keys()}
        for i, ri in G.items():
            for j, wij in ri.items():
                if j not in tmp:
                    tmp[j] = 0
                tmp[j] += 0.6 * rank[i] / (1.0 * len(ri))
        rank = tmp
    return rank

# 冷启动问题，抽取关键词及其权重，计算物品相似度
# def CalculatesSimilarity(D):
#     for di in D:
#         for dj in D:
#             w[i][j] = CosineSimilarity(di, dj)
#     return w

# 计算每个标签的流行度
def TagPopularity(records):
    tagfreq = dict()
    for user, item, tag in records:
        if tag not in tagfreq:
            tagfreq[tag] = 1
        else:
            tagfreq[tag] += 1
    return tagfreq

# 对于每个物品i，item_tags[i]存储了物品i的标签向量，其中item_tags[i][b]是对物品i打标签b的次数，那么物品i和j的余弦相似度
# 可以通过如下程序计算。
def CosinSim(item_tages, i, j):
    ret = 0
    for b, wib in item_tages[i].items():
        if b in item_tages[j]:
            ret += wib * item_tages[j][b]
    ni = 0
    nj = 0
    for b, w in item_tages[i].items():
        ni += w * w
    for b, w in item_tages[j].items():
        nj += w * w
    if ret == 0:
        return 0
    return ret / math.sqrt(ni * nj)

# 计算推荐列表的多样性
def Diversity(item_tages, recommend_items):
    ret = 0
    n = 0
    for i in recommend_items.keys():
        for j in recommend_items.keys():
            if i == j:
                continue
            ret += CosinSim(item_tages, i, j)
            n += 1
    return ret / (n * 1.0)

# SimpleTagBased标记算法
# 用records 存储标签数据的三元组，其中records[i] = [user, item, tag];
# 用user_tags 存储nu,b，其中user_tags[u][b] = nu,b;
# 用tag_items存储nb,i，其中tag_items[b][i] = nb,i。

def addValueToMat(theMat, key, value, incr):
    if key not in theMat:
        theMat[key] = value
        theMat[key][value] = incr
    else:
        if value not in theMat[key]:
            theMat[key][value] = incr
        else:
            theMat[key][value] += incr

def InitStat(records):
    user_tags = dict()
    tag_items = dict()
    user_items = dict()
    for user, item, tag in records.items():
        addValueToMat(user_tags, user, tag, 1)
        addValueToMat(tag_items, tag, item, 1)
        addValueToMat(user_items, user, item, 1)

# def RecommendByTag(user):
#     recommend_items = dict()
#     tagged_items = user_items[user]
#     for tag, wut in user_tags[user].items():
#         for item, wti in tag_items.items():
#             if item in tagged_items:
#                 continue
#             if item not in recommend_items:
#                 recommend_items = wut * wti
#             else:
#                 recommend_items += wut * wti
#     return recommend_items

# TF-IDF

# 基于好友的好友推荐算法
def FriendSuggestion(user, G, GT):
    suggestions = dict()
    friends = G[user]
    for fid in G[user]:
        for ffid in fid:
            if ffid in friends:
                continue
            if ffid not in suggestions:
                suggestions[ffid] = 0
            suggestions[ffid] += 1
    suggestions = {x: y / math.sqrt(len(G[user]) * len(G[x])) for x, y in suggestions}

# 推荐结果多样性
def ReasonDiversity(recommendations):
    reasons = set()
    for i in recommendations:
        if i.reason in reasons:
            i.weight /= 2
            reasons.add(i.reason)
    #recommendations = sortByWeight(recommendations)

# LFM(Latent Factor Model)
# train是训练集，user_items   F是引类格式   n是迭代次数    aloha是学习sulv   lamda正则化参数
# def LearningLFM(train, F, n, alpha, lamda):
#     # 初始化p,q矩阵
#     # [p, q] = InitLFM(train, F)


if __name__ == "__main__":
    d = {'Adam': 95, 'Lisa': 85, 'Bart': 59, 'Paul': 74}
    sum = 0
    for key, value in d.items():
        sum = sum + value
        print(key, ':', value)
    print('平均分为:', sum / len(d))

    l = {x:0 for x in [1,2,3,4,5,10]}
    print(l)
