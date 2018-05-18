
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
import numpy as np
from sklearn.metrics import mean_squared_error
import pandas as pd
from scipy.stats import norm, skew

#几个常用的模型
# KNN
# Support Vector Machine
# Gradient Boosting Classifier
# XGBoost
# Multi Layer Perceptron
# Linear SVMC
# Random Forest
# Logistic Regression
# Decision Tree
# Adaboost
# Extra Tree

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

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='white', context='notebook', palette='deep')
mycols = ["#76CDE9", "#FFE178", "#9CE995", "#E97A70"]
sns.set_palette(palette = mycols, n_colors = 4)

#画卡方图，看属性的重要性，用于削减特征
def chi2test(X_train, Y_train,X_test):
    Kbest = SelectKBest(score_func=chi2, k=10)
    fit = Kbest.fit(X_train, Y_train)
    # Create a table with the results and score for each features
    scores = pd.DataFrame({'Columns': X_test.columns.values, 'Score': fit.scores_})
    # Visualise the scores of dependence with a barplot
    plt.subplots(figsize=(15, 10))
    g = sns.barplot('Score', 'Columns', data=scores, palette=mycols, orient="h")
    g.set_xlabel("Importance")
    g = g.set_title("Feature Importances using Chi-squared")

from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier, RandomForestClassifier, VotingClassifier
#画extra tree，看属性的重要性，用于削减特征
def extreeTest(X_train, Y_train,X_test):
    model = ExtraTreesClassifier()
    model.fit(X_train, Y_train)
    # And create a table with the importances
    scores = pd.DataFrame({'Columns': X_test.columns.values, 'Score': model.feature_importances_})
    scores.sort_values(by='Score', ascending=False)
    # Finally let's visualise this
    plt.subplots(figsize=(15, 10))
    g = sns.barplot('Score', 'Columns', data=scores, palette=mycols, orient="h")
    g.set_xlabel("Importance")
    g = g.set_title("Feature Importances using Trees")


#部分模型包含feature importance_，可以不用上述两种方法检验
#gbc = GradientBoostingClassifier(random_state=0)
#xgb = XGBClassifier()
#rf = RandomForestClassifier()
#dt = DecisionTreeClassifier()
#ada = AdaBoostClassifier()
#etc = ExtraTreesClassifier()
def importanceTest(model,X_train):
    indices = np.argsort(model.feature_importances_)[::-1]
    # Visualise these with a barplot
    plt.subplots(figsize=(15, 10))
    g = sns.barplot(model=X_train.columns[indices], x=model.feature_importances_[indices], orient='h', palette=mycols)
    g.set_xlabel("Relative importance", fontsize=12)
    g.set_ylabel("Features", fontsize=12)
    g.tick_params(labelsize=9)
    g.set_title("GBC feature importance")

#同上，计算rfecv重要性
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import ShuffleSplit
from sklearn.feature_selection import RFECV
from sklearn import metrics
def featureReduction(model,X_train,Y_train,X_test,Y_test,test):
    shuff = ShuffleSplit(n_splits=3, test_size=0.2, random_state=50)
    # Take some copies
    model_red_train = X_train.copy()
    model_red_test = X_test.copy()
    model_final_test = test.copy()
    # Fit a model to the estimation data
    modelFit = model.fit(model_red_train, Y_train)
    # Allow the feature importances attribute to select the most important features
    gbc_feat_red = SelectFromModel(modelFit, prefit=True)
    # Reduce estimation, validation and test datasets
    model_X_train = gbc_feat_red.transform(model_red_train)
    model_X_test = gbc_feat_red.transform(model_red_test)
    model_final_test = gbc_feat_red.transform(model_final_test)

    print("Results of 'feature_importances_':")
    print('X_train: ', model_X_train.shape, '\nX_test: ', model_X_test.shape, '\nY_train: ', Y_train.shape, '\nY_test: ',
          Y_test.shape)
    print("-" * 75)

    ##############################################################################

    # Take come copies
    model_rfecv_train = X_train.copy()
    model_rfecv_test = X_test.copy()
    model_rfecv_final_test = test.copy()

    # Initialise RFECV
    model_rfecv = RFECV(estimator=model, step=1, cv=shuff, scoring='accuracy')

    # Fit RFECV to the estimation data
    model_rfecv.fit(model_rfecv_train, Y_train)

    # Now reduce estimation, validation and test datasets
    model_rfecv_X_train = model_rfecv.transform(model_rfecv_train)
    model_rfecv_X_test = model_rfecv.transform(model_rfecv_test)
    model_rfecv_final_test = model_rfecv.transform(model_rfecv_final_test)
    print("Results of 'RFECV':")
    # Let's see the results of RFECV
    print(model_rfecv.support_)
    print(model_rfecv.ranking_)
    print('X_train: ', model_rfecv_X_train.shape, '\nX_test: ', model_rfecv_X_test.shape, '\nY_train: ', Y_train.shape,
          '\nY_test: ', Y_test.shape)

    # Fit estimator to reduced dataset
    model.fit(model_X_train, Y_train)

    # Compute cross validated scores and take the mean
    model_scores = cross_val_score(model, model_X_train, Y_train, cv=shuff)
    model_scores = model_scores.mean()

    print('feature_importances_ - Mean Cross Validated Score: {:.2f}'.format(model_scores * 100))
    model_apply_acc = metrics.accuracy_score(Y_test, model.predict(model_X_test))
    print('feature_importances_ - Accuracy when applied to Test: {:.2f}'.format(model_apply_acc * 100))
    print("-" * 50)

    ##############################################################################
    # Fit estimator to reduced dataset
    model.fit(model_rfecv_X_train, Y_train)

    # Compute cross validated scores and take the mean
    model_scores = cross_val_score(model, model_rfecv_X_train, Y_train, cv=shuff)
    model_scores = model_scores.mean()

    print('RFECV - Mean Cross Validated Score: {:.2f}'.format(model_scores * 100))
    model_apply_acc = metrics.accuracy_score(Y_test, model.predict(model_rfecv_X_test))
    print('RFECV - Accuracy when applied to Test: {:.2f}'.format(model_apply_acc * 100))

# stacked_averaged_models = StackingAveragedModels(base_models = (ENet, GBoost, KRR),
#                                                  meta_model = lasso)
# score = rmsle_cv(stacked_averaged_models)
# print("Stacking Averaged models score: {:.4f} ({:.4f})".format(score.mean(), score.std()))