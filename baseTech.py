
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
import numpy as np
from sklearn.metrics import mean_squared_error
import pandas as pd
from scipy.stats import norm, skew

#stack回归器，相当于重写一个类
class StackingAveragedModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, base_models, meta_model, n_folds=5):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds

    # We again fit the data on clones of the original models
    def fit(self, X, y):
        self.base_models_ = [list() for x in self.base_models]
        self.meta_model_ = clone(self.meta_model)
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=156)

        # Train cloned base models then create out-of-fold predictions
        # that are needed to train the cloned meta-model
        out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))
        #各个模型进行训练，预测
        for i, model in enumerate(self.base_models):
            #存储了一个模型的n_splits折，用于后面各模型取平均
            for train_index, holdout_index in kfold.split(X, y):
                instance = clone(model)
                self.base_models_[i].append(instance)
                instance.fit(X[train_index], y[train_index])
                y_pred = instance.predict(X[holdout_index])
                out_of_fold_predictions[holdout_index, i] = y_pred

        # Now train the cloned  meta-model using the out-of-fold predictions as new feature
        self.meta_model_.fit(out_of_fold_predictions, y)
        return self

    # Do the predictions of all base models on the test data and use the averaged predictions as
    # meta-features for the final prediction which is done by the meta-model
    def predict(self, X):
        #一个模型有n_split个预测结果，因此需要取平均
        meta_features = np.column_stack([
            np.column_stack([model.predict(X) for model in base_models]).mean(axis=1)
            for base_models in self.base_models_ ])
        return self.meta_model_.predict(meta_features)


#用自己搭建的模型进行预测，返回的是交叉验证的误差
def rmsle_cv(model,n_folds,train,y_train):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train.values)
    rmse= np.sqrt(-cross_val_score(model, train.values, y_train, scoring="neg_mean_squared_error", cv = kf))
    return(rmse)

#计算残差，即平方误差
def rmsle(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))

#alpha为阈值
def skewfy(all_data,alpha=0.75,lam=0.15):
    numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index
    # Check the skew of all numerical features
    skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
    skewness = pd.DataFrame({'Skew' :skewed_feats})
    skewness = skewness[abs(skewness) > alpha]
    print("There are {} skewed numerical features to Box Cox transform".format(skewness.shape[0]))
    from scipy.special import boxcox1p
    skewed_features = skewness.index
    for feat in skewed_features:
        # all_data[feat] += 1
        all_data[feat] = boxcox1p(all_data[feat], lam)
    return all_data

# stacked_averaged_models = StackingAveragedModels(base_models = (ENet, GBoost, KRR),
#                                                  meta_model = lasso)
# score = rmsle_cv(stacked_averaged_models)
# print("Stacking Averaged models score: {:.4f} ({:.4f})".format(score.mean(), score.std()))