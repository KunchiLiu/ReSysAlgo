# 推荐系统算法用到的Python库Surprise(Simple Python Recommendation System Engine)
# https://blog.csdn.net/mycafe_/article/details/79146764#11-movielens%E7%9A%84%E4%BE%8B%E5%AD%90
# 可使用上面提到的各种推荐系统算法


# 默认载入movielens数据集，会提示是否下载这个数据集，这是非常经典的公开推荐系统数据集——MovieLens数据集之一
import Dataset as Dataset

data = Dataset.load_builtin('ml-100k')
# k折交叉验证(k=3)
data.split(n_folds = 3)
# 试一试把SVD矩阵分解
algo = SVD()
# 在数据集上测试一下效果
perf = evaluate(algo, data, measuers = ['RMSE', 'MAE'])
# 输出结果
print_perf(perf)
