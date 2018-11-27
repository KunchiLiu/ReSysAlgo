# -*- coding=utf-8 -*-
# https://blog.csdn.net/guanbai4146/article/details/78016778
import math
from operator import itemgetter

#例子数据相当于一个用户dict，如下存储原始数据


dic = {'A':('a','b','d'),'B':('a','c'),'C':('b','e'),'D':('c','d','e')}

# 计算用户兴趣相似度
def Usersim(dicc):
    # 把用户-商品字典转化为商品-用户字典
    item_user = dict()
    for u,items in dicc.items():
        for i in items:
            if i not in item_user.keys():
                item_user[i] = set() #i键所对应的值是一个集合
            item_user[i].add(u)

    C = dict() # {('A','B'):0.4,('A','C'):0.5}
    N = dict() # {'A':3,'B':2,'C':2,'D':3}
    for item,users in item_user.items():
        for u in users:
            if u not in N.keys():
                N[u] = 0 # 字典没有初值不可以相加
            N[u] += 1    # 每个商品下，用户出现一次加一次，可计算每个用户一共购买的商品个数

            # 但是这个值亦可以从刚开始的用户表中获得
            # for u in dic.keys():
            #     N[u] = len(dic[u])

            for v in users:
                if u == v:
                    continue
                if (u,v) not in C.keys():
                    C[u,v] = 0
                C[u,v] += 1
    # 至此，倒排建立完成
    W = dict()
    for co_user,cuv in C.items():
        W[co_user] = cuv / math.sqrt(N[co_user[0]]*N[co_user[1]])
    return W

def Recommend(user,dicc,W2,K):
    rvi = 1 # 这里都是1，实际上则不一定，每个人喜欢beautiful girl，但有的哥们喜欢可带的多一点，有个喜欢御姐型多一点
    rank = dict()
    related_user = []
    interacted_items = dicc[user]
    for co_user,items in W2.items():
        if user == co_user[0]:
            related_user.append((co_user[1],items)) # 建立一个和待推荐用户兴趣相关的所有的用户列表
        for v,wuv in sorted(related_user,key = itemgetter(1),reverse = True)[0:K]:
    # key = itemgetter(1)，排序，根据related_user第一个域进行排序
    # 按兴趣度从大到小排序，选取前K个
            for i in dicc[v]:
                if i in interacted_items:
                    continue
                if i not in rank.keys():
                    rank[i] = 0
                rank[i] += wuv*rvi
    return rank

if __name__ == '__main__':
    W3 = Usersim(dic)
    Last_Rank = Recommend('A',dic,W3,2)
    print(Last_Rank)









