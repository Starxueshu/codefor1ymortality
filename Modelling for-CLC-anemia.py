#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import f1_score,precision_score,recall_score,roc_auc_score,accuracy_score,roc_curve
from sklearn.metrics import brier_score_loss
from sklearn.metrics import log_loss

print('Numpy version:', np.__version__)
print('Pandas version:', pd.__version__)


# In[2]:


#conda install scikit-learn


# In[3]:


#conda install scikit-learn-intelex
#pip install scipy


# In[4]:


#from sklearn.base import BaseEstimator, TransformerMixin


# In[5]:


#import pandas as pd


# In[6]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = 12, 8
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12


# import dataset

# In[7]:


data_train = pd.read_excel(r'F:/materi/3-CLC/CLC 2.0/start0.7.xlsx')
data_test = pd.read_excel(r'F:/materi/3-CLC/CLC 2.0/starv0.3.xlsx')


# In[ ]:





# In[8]:


pd.set_option('display.max_columns', data_train.shape[1])
pd.set_option('max_colwidth', 1000)


# In[9]:


data_train.head()


# In[10]:


data_train.info()


# Data preprocessing piprlines
# Prepare the data to a format that can be fit into scikit learn algorithms

# Categorical variable encoder

# In[11]:


categorical_vars = [
"Tachycardia3",
"CHF1",
"Aspirin",
"Clopidogrel",
"Bblocker",
"Cablocker",
"ACEIARB",
"Statins"]


# In[12]:


data_train[categorical_vars].head()


# In[13]:


# to make a custom transformer to fit into a pipeline
class Vars_selector(BaseEstimator, TransformerMixin):
    '''Returns a subset of variables in a dataframe'''
    def __init__(self, var_names):
        '''var_names is a list of categorical variables names'''
        self.var_names = var_names
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        '''returns a dataframe with selected variables'''
        return X[self.var_names]


# In[14]:


class Cat_vars_encoder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        '''X is a dataframe'''

        return X.values


# Transform data in a pipeline

# In[15]:


# categorical variables preprocessing
cat_vars_pipeline = Pipeline([
    ('selector', Vars_selector(categorical_vars)),
    ('encoder', Cat_vars_encoder())
])


# For many machine learning algorithms, gradient descent is the preferred or even the only optimization method to learn the model parameters. Gradient descent is highly sensitive to feature scaling.

# ** Continuous vars **

# In[16]:


continuous_vars = ["HR",
"Hb",
"Albumin",
"HDL",
"cMDRDeGRR",
"LVEF"]


# In[17]:


data_train[continuous_vars].describe()


# The scales among the continuous variables vary a lot, we need to standardize them prior to modelling.

# In[18]:


# continuous variables preprocessing
cont_vars_pipeline = Pipeline([
    ('selector', Vars_selector(continuous_vars)),
    ('standardizer', StandardScaler())
])


# To transform the two types of variables in one step

# In[19]:


preproc_pipeline = FeatureUnion(transformer_list=[
    ('cat_pipeline', cat_vars_pipeline),
    ('cont_pipeline', cont_vars_pipeline)
])


# In[20]:


data_train_X = pd.DataFrame(preproc_pipeline.fit_transform(data_train), 
                            columns=categorical_vars + continuous_vars)


# In[21]:


data_train_X.head()


# Fitting classifiers

# In[22]:


from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.model_selection import learning_curve, RandomizedSearchCV, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score


# In[23]:


data_train['Death1y'].value_counts()


# This is a fairly balanced dataset(i.e., the number of positive and negative cases are roughly the same), and we'll use AUC as our metric to optimise the model performance.

# Assessing learning curve using the model default settings
# Tuning the model hyper-parameters are always difficult, so a good starting point is to see how the Scikit-learn default settings for the model performs, i.e., to see if it overfits or underfits, or is just right. This will give a good indication as to the direction of tuning.

# In[24]:


def plot_learning_curves(model, X, y):
    train_sizes, train_scores, val_scores = learning_curve(model, X, y, cv=5, scoring='roc_auc',
                                                           random_state=42, n_jobs=-1)
    plt.plot(train_sizes, np.mean(train_scores, axis=1), "o-", label="training scores")
    plt.plot(train_sizes, np.mean(val_scores, axis=1), "o-", label="x-val scores")
    plt.legend(fontsize=14).get_frame().set_facecolor('white')
    plt.xlabel("Training set size")
    plt.ylabel("Area under Curve")
    plt.title('{} learning curve'.format(model.__class__.__name__))


# # Compute and compare test metrics
# Transform test data set

# In[25]:


data_test_X = pd.DataFrame(preproc_pipeline.transform(data_test), # it's imperative not to do fit_transfomr again
                           columns=categorical_vars + continuous_vars)


# In[26]:


data_test_X.shape


# In[27]:


data_test_X.head()


# In[28]:


def plot_roc_curve(fpr, tpr, auc, model=None):
    if model == None:
        title = None
    elif isinstance(model, str):
        title = model
    else:
        title = model.__class__.__name__
#    title = None if model == None else model.__class__.__name__
    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, linewidth=2, label='auc: {}'.format(auc))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([-.01, 1.01, -.01, 1.01])
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.legend(fontsize=14).get_frame().set_facecolor('white')
    plt.title('{} - ROC Curve'.format(title))


# # Logistic Regression---网格搜索模型调参GridSearchCV

# In[26]:


lr_clf = LogisticRegression(n_jobs = -1)
plot_learning_curves(lr_clf, data_train_X, data_train['Death1y'])


# In[ ]:





# Let's see if we can squeeze some more performance out by optimising C

# In[62]:


param_grid = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
    }
lr_clf = LogisticRegression(random_state=42)
grid_search = GridSearchCV(lr_clf, param_grid=param_grid, return_train_score=True,
                                cv=5, scoring='roc_auc', n_jobs=-1)
grid_search.fit(data_train_X, data_train['Death1y'])


# In[63]:


cv_rlt = grid_search.cv_results_
pd.DataFrame(sorted(list(zip(cv_rlt["mean_test_score"], cv_rlt['mean_train_score'], cv_rlt["params"])), 
                    key=lambda x: x[0], reverse=True), columns=['X-val scores', 'Train scores', 'Params'])


# # Looks like C=100? is our best value.

# 下方with open,"wb",确实是保存了新的pkl模型文件，下一次使用该文件只需要直接调用即可

# In[66]:


lr_clf = grid_search.best_estimator_
with open('F:/materi/3-CLC/CLC 2.0/lr_clf_final_round.pkl', 'wb') as f:
    pickle.dump(lr_clf, f)


# In[65]:


plot_learning_curves(lr_clf, data_train_X, data_train['Death1y'])


# Looks like the logistic regression model would benefit from additional data.

# # Logistic Regression model-ROC

# In[35]:


# Import model and retrain
with open('F:/materi/3-CLC/CLC 2.0/lr_clf_final_round.pkl', 'rb') as f:
    lr_clf = pickle.load(f)
lr_clf.fit(data_train_X, data_train['Death1y'])


# Accuracy scores

# In[26]:


accu_lr = accuracy_score(data_test['Death1y'], lr_clf.predict(data_test_X))


# In[27]:


round(accu_lr,3)


# In[28]:


pd.crosstab(data_test['Death1y'], lr_clf.predict(data_test_X))


# In[29]:


pred_proba_lr = lr_clf.predict_proba(data_test_X)


# In[30]:


lr_clf.predict(data_test_X)


# In[31]:


data_test['Death1y']


# In[32]:


fpr, tpr, _ = roc_curve(data_test['Death1y'], pred_proba_lr[:, 1])
auc_lr = roc_auc_score(data_test['Death1y'], pred_proba_lr[:, 1])


# In[33]:


lr_score = lr_clf.score(data_test_X, data_test['Death1y'])
lr_accuracy_score=accuracy_score(data_test['Death1y'],lr_clf.predict(data_test_X))
lr_preci_score=precision_score(data_test['Death1y'],lr_clf.predict(data_test_X))
lr_recall_score=recall_score(data_test['Death1y'],lr_clf.predict(data_test_X))
lr_f1_score=f1_score(data_test['Death1y'],lr_clf.predict(data_test_X))
lr_auc=roc_auc_score(data_test['Death1y'],pred_proba_lr[:, 1])
print('lr_accuracy_score: %f,lr_preci_score: %f,lr_recall_score: %f,lr_f1_score: %f,lr_auc: %f'
      %(lr_accuracy_score,lr_preci_score,lr_recall_score,lr_f1_score,lr_auc))


# In[ ]:





# In[74]:


brier_score_loss(data_test['Death1y'], pred_proba_lr[:, 1])


# In[75]:


print('loss lr:', log_loss(data_test['Death1y'], pred_proba_lr[:, 1]))


# In[76]:


round(auc_lr,3)


# In[77]:


plot_roc_curve(fpr, tpr, round(auc_lr,3), lr_clf)


# In[78]:


data_test['lr_pred_proba'] = pred_proba_lr[:, 1]


# In[79]:


data_test.to_csv('F:/materi/3-CLC/CLC 2.0/test_set_with_predictions-lr.csv'.format(len(data_train)), index=False)


# In[ ]:





# In[ ]:





# In[ ]:





# ROC曲线+95%CI Bootstrap方法

# In[43]:


X_train = data_train_X
y_train = data_train['Death1y']
X_test = data_test_X
y_test = data_test['Death1y']


# In[44]:


y = data_test['Death1y']
pred = pred_proba_lr[:, 1]


# In[80]:


def bootstrap_auc(y, pred, classes, bootstraps = 100, fold_size = 1000):
    statistics = np.zeros((len(classes), bootstraps))

    for c in range(len(classes)):
        df = pd.DataFrame(columns=['y', 'pred'])
        # df.
        df.loc[:, 'y'] = y
        df.loc[:, 'pred'] = pred
        df_pos = df[df.y == 1]
        df_neg = df[df.y == 0]
        prevalence = len(df_pos) / len(df)
        for i in range(bootstraps):
            pos_sample = df_pos.sample(n = int(fold_size * prevalence), replace=True)
            neg_sample = df_neg.sample(n = int(fold_size * (1-prevalence)), replace=True)

            y_sample = np.concatenate([pos_sample.y.values, neg_sample.y.values])
            pred_sample = np.concatenate([pos_sample.pred.values, neg_sample.pred.values])
            score = roc_auc_score(y_sample, pred_sample)
            statistics[c][i] = score
    return statistics


# In[81]:


y = data_test['Death1y']
scores_lr = pred_proba_lr[:, 1]
statistics_lr = bootstrap_auc(y,scores_lr,[0,1])
print("均值:",np.mean(statistics_lr,axis=1))
print("最大值:",np.max(statistics_lr,axis=1))
print("最小值:",np.min(statistics_lr,axis=1))


# In[ ]:





# In[ ]:





# ROC曲线+95%CI 方法2

# In[97]:


from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt


# In[ ]:


from sklearn.metrics import roc_curve, roc_auc_score
#label是每个样本对应的真实标签(0或1)，pred_prob是模型输出的对每个样本的预测概率
FPR, TPR, _ = roc_curve(y, pred, pos_label = 1)
AUC = roc_auc_score(y, pred)


# In[ ]:


from scipy.stats import norm
import numpy as np
def AUC_CI(auc, label, alpha = 0.05):
	label = np.array(label)#防止label不是array类型
	n1, n2 = np.sum(label == 1), np.sum(label == 0)
	q1 = auc / (2-auc)
	q2 = (2 * auc ** 2) / (1 + auc)
	se = np.sqrt((auc * (1 - auc) + (n1 - 1) * (q1 - auc ** 2) + (n2 -1) * (q2 - auc ** 2)) / (n1 * n2))
	confidence_level = 1 - alpha
	z_lower, z_upper = norm.interval(confidence_level)
	lowerb, upperb = auc + z_lower * se, auc + z_upper * se
	return (lowerb, upperb)


# In[98]:


import matplotlib.pyplot as plt
def plot_AUC(ax, FPR, TPR, AUC, CI, label):
	label = '{}: {} ({}-{})'.format(str(label), round(AUC, 3), round(CI[0], 3), round(CI[1], 3))
	ax.plot(FPR, TPR, label = label)
	return ax


# In[130]:


FPR_test, TPR_test, _ = roc_curve(y, pred, pos_label = 1)
AUC_test = roc_auc_score(y, pred)
CI_test = AUC_CI(AUC_test, y, 0.05)


# In[133]:


AUC_test


# In[132]:


CI_test


# In[131]:


plt.style.use('ggplot')
fig, ax = plt.subplots()
#ax = plot_AUC(ax, FPR_train, TPR_train, AUC_train, CI_train, label = 'train')
ax = plot_AUC(ax, FPR_test, TPR_test, AUC_test, CI_test, label = 'test')
ax.plot((0, 1), (0, 1), ':', color = 'grey')
ax.set_xlim(-0.01, 1.01)
ax.set_ylim(-0.01, 1.01)
ax.set_aspect('equal')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.legend()
plt.show()


# In[126]:





# In[ ]:





# In[ ]:





# # XGboot

# In[83]:


from xgboost.sklearn import XGBClassifier


# In[84]:


Xgbc_clf=XGBClassifier(random_state=42)  #Xgbc
plot_learning_curves(Xgbc_clf, data_train_X, data_train['Death1y'])


# max_depth = 5 ：这应该在3-10之间。我从5开始，但你也可以选择不同的数字。4-6可以是很好的起点。
# min_child_weight = 1 ：选择较小的值是因为它是高度不平衡的类问题，并且叶节点可以具有较小的大小组。
# gamma = 0.1 ：也可以选择较小的值，如0.1-0.2来启动。无论如何，这将在以后进行调整。
# subsample，colsample_bytree = 0.8：这是一个常用的使用起始值。典型值介于0.5-0.9之间。
# scale_pos_weight = 1：由于高级别的不平衡。
# colsample_bytree = 0.5,gamma=0.2

# In[85]:


param_distribs = {
     'n_estimators': stats.randint(low=20, high=120),      
    'max_depth': stats.randint(low=1, high=100),
    'min_child_weight': stats.randint(low=1, high=100)
    }
Xgbc_clf=XGBClassifier(random_state=42,learning_rate=0.125,use_label_encoder=False)
Xgbc_search = RandomizedSearchCV(Xgbc_clf, param_distributions=param_distribs, return_train_score=True,
                                n_iter=100, cv=5, scoring='roc_auc', random_state=42, n_jobs=-1)
Xgbc_gs=Xgbc_search.fit(data_train_X, data_train['Death1y'])


# In[86]:


print(Xgbc_gs.best_score_)


# In[87]:


print(Xgbc_gs.best_params_)


# In[88]:


cv_rlt = Xgbc_search.cv_results_
pd.DataFrame(sorted(list(zip(cv_rlt["mean_test_score"], cv_rlt['mean_train_score'], cv_rlt["params"])), 
                    key=lambda x: x[0], reverse=True), columns=['X-val scores', 'Train scores', 'Params'])


# In[91]:


Xgbc_clf = Xgbc_search.best_estimator_
Xgbc_clf.fit(data_train_X, data_train['Death1y'])
with open('F:/materi/3-CLC/CLC 2.0/Xgbc_clf_final_round.pkl', 'wb') as f:
    pickle.dump(Xgbc_clf, f)


# In[92]:


Xgbc_clf


# In[93]:


plot_learning_curves(rf_clf, data_train_X, data_train['Death1y'])


# # XGBoost-ROC

# In[34]:


# Import model and retrain
with open('F:/materi/3-CLC/CLC 2.0/Xgbc_clf_final_round.pkl', 'rb') as f:
    Xgbc_clf = pickle.load(f)
Xgbc_clf.fit(data_train_X, data_train['Death1y'])


# In[95]:


accu_Xgbc = accuracy_score(data_test['Death1y'], Xgbc_clf.predict(data_test_X))
round(accu_Xgbc,3)


# In[96]:


pd.crosstab(data_test['Death1y'], Xgbc_clf.predict(data_test_X))


# In[97]:


pred_proba_Xgbc = Xgbc_clf.predict_proba(data_test_X)


# In[98]:


fpr, tpr, _ = roc_curve(data_test['Death1y'], pred_proba_Xgbc[:, 1])
auc_Xgbc = roc_auc_score(data_test['Death1y'], pred_proba_Xgbc[:, 1])
round(auc_Xgbc,3)


# In[99]:


Xgbc_score = Xgbc_clf.score(data_test_X, data_test['Death1y'])
Xgbc_accuracy_score=accuracy_score(data_test['Death1y'],Xgbc_clf.predict(data_test_X))
Xgbc_preci_score=precision_score(data_test['Death1y'],Xgbc_clf.predict(data_test_X))
Xgbc_recall_score=recall_score(data_test['Death1y'],Xgbc_clf.predict(data_test_X))
Xgbc_f1_score=f1_score(data_test['Death1y'],Xgbc_clf.predict(data_test_X))
Xgbc_auc=roc_auc_score(data_test['Death1y'],pred_proba_Xgbc[:, 1])
print('Xgbc_accuracy_score: %f,Xgbc_preci_score: %f,Xgbc_recall_score: %f,Xgbc_f1_score: %f,Xgbc_auc: %f'
      %(Xgbc_accuracy_score,Xgbc_preci_score,Xgbc_recall_score,Xgbc_f1_score,Xgbc_auc))


# In[100]:


brier_score_loss(data_test['Death1y'], pred_proba_Xgbc[:, 1])


# In[101]:


print('loss Xgbc:', log_loss(data_test['Death1y'], pred_proba_Xgbc[:, 1]))


# In[102]:


plot_roc_curve(fpr, tpr, round(auc_Xgbc,3), Xgbc_clf)


# In[103]:


data_test['lr_pred_proba'] = pred_proba_Xgbc[:, 1]
data_test.to_csv('F:/materi/3-CLC/CLC 2.0/test_set_with_predictions-Xgbc.csv'.format(len(data_train)), index=False)


# In[104]:


y_Xgbc = data_test['Death1y']
scores_Xgbc = pred_proba_Xgbc[:, 1]
statistics_Xgbc = bootstrap_auc(y_Xgbc,scores_Xgbc,[0,1])
print("均值:",np.mean(statistics_Xgbc,axis=1))
print("最大值:",np.max(statistics_Xgbc,axis=1))
print("最小值:",np.min(statistics_Xgbc,axis=1))


# In[ ]:





# In[ ]:





# # DecisionTreeClassifier

# In[105]:


from sklearn.tree import DecisionTreeClassifier


# In[106]:


tr_clf=DecisionTreeClassifier(random_state=42)  # 决策树模型
plot_learning_curves(tr_clf, data_train_X, data_train['Death1y'])


# In[107]:


param_distribs = {
         'max_features': ['auto', 'log2'],
        'max_depth': stats.randint(low=1, high=50),
        'min_samples_split': stats.randint(low=2, high=200), 
        'min_samples_leaf': stats.randint(low=2, high=200)
    }
dt_clf = DecisionTreeClassifier(random_state=42,criterion='gini', splitter='best')
rnd_search = RandomizedSearchCV(dt_clf, param_distributions=param_distribs, return_train_score=True,
                                n_iter=100, cv=5, scoring='roc_auc', random_state=42, n_jobs=-1)
gsdt=rnd_search.fit(data_train_X, data_train['Death1y'])


# In[108]:


print(gsdt.best_score_)


# In[109]:


print(gsdt.best_params_)


# In[110]:


cv_rlt = rnd_search.cv_results_
pd.DataFrame(sorted(list(zip(cv_rlt["mean_test_score"], cv_rlt['mean_train_score'], cv_rlt["params"])), 
                    key=lambda x: x[0], reverse=True), columns=['X-val scores', 'Train scores', 'Params'])


# In[111]:


dt_clf = rnd_search.best_estimator_
dt_clf.fit(data_train_X, data_train['Death1y'])
with open('F:/materi/3-CLC/CLC 2.0/dt_clf_final_round.pkl', 'wb') as f:
    pickle.dump(dt_clf, f)


# In[112]:


plot_learning_curves(dt_clf, data_train_X, data_train['Death1y'])


# # Decision Tree ROC计算

# In[36]:


# Import model and retrain
with open('F:/materi/3-CLC/CLC 2.0/dt_clf_final_round.pkl', 'rb') as f:
    dt_clf = pickle.load(f)
dt_clf.fit(data_train_X, data_train['Death1y'])


# In[114]:


accu_dt = accuracy_score(data_test['Death1y'], dt_clf.predict(data_test_X))
round(accu_dt,3)


# In[115]:


pd.crosstab(data_test['Death1y'], dt_clf.predict(data_test_X))


# In[116]:


pred_proba_dt = dt_clf.predict_proba(data_test_X)
fpr, tpr, _ = roc_curve(data_test['Death1y'], pred_proba_dt[:, 1])
auc_dt = roc_auc_score(data_test['Death1y'], pred_proba_dt[:, 1])
round(auc_dt,3)


# In[117]:


dt_score = dt_clf.score(data_test_X, data_test['Death1y'])
dt_accuracy_score=accuracy_score(data_test['Death1y'],dt_clf.predict(data_test_X))
dt_preci_score=precision_score(data_test['Death1y'],dt_clf.predict(data_test_X))
dt_recall_score=recall_score(data_test['Death1y'],dt_clf.predict(data_test_X))
dt_f1_score=f1_score(data_test['Death1y'],dt_clf.predict(data_test_X))
dt_auc=roc_auc_score(data_test['Death1y'],pred_proba_dt[:, 1])
print('dt_accuracy_score: %f,dt_preci_score: %f,dt_recall_score: %f,dt_f1_score: %f,dt_auc: %f'
      %(dt_accuracy_score,dt_preci_score,dt_recall_score,dt_f1_score,dt_auc))


# In[118]:


brier_score_loss(data_test['Death1y'], pred_proba_dt[:, 1])


# In[119]:


print('loss dt:', log_loss(data_test['Death1y'], pred_proba_dt[:, 1]))


# In[120]:


plot_roc_curve(fpr, tpr, round(auc_dt,3), dt_clf)


# In[121]:


data_test['lr_pred_proba'] = pred_proba_dt[:, 1]


# In[122]:


data_test.to_csv('F:/materi/3-CLC/CLC 2.0/test_set_with_predictions-decision tree.csv'.format(len(data_train)), index=False)


# In[123]:


y_dt = data_test['Death1y']
scores_dt = pred_proba_dt[:, 1]
statistics_dt = bootstrap_auc(y_dt,scores_dt,[0,1])
print("均值:",np.mean(statistics_dt,axis=1))
print("最大值:",np.max(statistics_dt,axis=1))
print("最小值:",np.min(statistics_dt,axis=1))


# # 决策树图片

# In[114]:


from sklearn.tree import DecisionTreeClassifier
from IPython.display import Image  
from sklearn import tree
import pydotplus


# In[135]:


# 创建 DOT data
dot_data = tree.export_graphviz(dt_clf, out_file=None, 
                                feature_names=data_train_X.columns,  
                                class_names=['Alive','Death'],
                                filled=True, rounded=True,  
                                special_characters=True)
# 绘图
graph = pydotplus.graph_from_dot_data(dot_data)  


# In[136]:


from IPython.display import Image


# In[137]:


# 展现图形
Image(graph.create_png()) 


# In[ ]:





# # KNN机器学习算法

# In[124]:


from sklearn.neighbors import KNeighborsClassifier


# In[125]:


knn_clf=KNeighborsClassifier()  # 决策树模型
plot_learning_curves(knn_clf, data_train_X, data_train['Death1y'])


# In[132]:


from sklearn.model_selection import GridSearchCV
param_grid = [
    { 'weights':['uniform'],
     'n_neighbors':[i for i in range (1,11)]##网上多为设置为11
    },
    {
        'weights':['distance'],
        'n_neighbors':[i for i in range (1,11)],##网上多为设置为11
        'p':[i for i in range(1,6)]
    }
]

grid_search_knn = GridSearchCV(knn_clf, param_grid, cv=5, n_jobs=-1, verbose=10)

grid_search_knn.fit(data_train_X, data_train['Death1y'])


# In[133]:


grid_search_knn.best_score_


# In[134]:


grid_search_knn.best_params_


# In[135]:


grid_search_knn.best_estimator_


# In[136]:


knn_clf = grid_search_knn.best_estimator_
knn_clf.fit(data_train_X, data_train['Death1y'])
with open('F:/materi/3-CLC/CLC 2.0/knn_clf_final_round.pkl', 'wb') as f:
    pickle.dump(knn_clf, f)


# In[137]:


plot_learning_curves(knn_clf, data_train_X, data_train['Death1y'])


# # KNN-ROC

# In[138]:


# Import model and retrain
with open('F:/materi/3-CLC/CLC 2.0/knn_clf_final_round.pkl', 'rb') as f:
    knn_clf = pickle.load(f)
knn_clf.fit(data_train_X, data_train['Death1y'])


# In[139]:


accu_knn = accuracy_score(data_test['Death1y'], knn_clf.predict(data_test_X))
round(accu_knn,3)


# In[140]:


pd.crosstab(data_test['Death1y'], knn_clf.predict(data_test_X))


# In[141]:


pred_proba_knn = knn_clf.predict_proba(data_test_X)
fpr, tpr, _ = roc_curve(data_test['Death1y'], pred_proba_knn[:, 1])
auc_knn = roc_auc_score(data_test['Death1y'], pred_proba_knn[:, 1])
round(auc_knn,3)


# In[142]:


knn_score = knn_clf.score(data_test_X, data_test['Death1y'])
knn_accuracy_score=accuracy_score(data_test['Death1y'],knn_clf.predict(data_test_X))
knn_preci_score=precision_score(data_test['Death1y'],knn_clf.predict(data_test_X))
knn_recall_score=recall_score(data_test['Death1y'],knn_clf.predict(data_test_X))
knn_f1_score=f1_score(data_test['Death1y'],knn_clf.predict(data_test_X))
knn_auc=roc_auc_score(data_test['Death1y'],pred_proba_knn[:, 1])
print('knn_accuracy_score: %f,knn_preci_score: %f,knn_recall_score: %f,knn_f1_score: %f,knn_auc: %f'
      %(knn_accuracy_score,knn_preci_score,knn_recall_score,knn_f1_score,knn_auc))


# In[143]:


brier_score_loss(data_test['Death1y'], pred_proba_knn[:, 1])


# In[144]:


print('loss knn:', log_loss(data_test['Death1y'], pred_proba_knn[:, 1]))


# In[145]:


plot_roc_curve(fpr, tpr, round(auc_knn,3), knn_clf)


# In[146]:


data_test['lr_pred_proba'] = pred_proba_knn[:, 1]


# In[147]:


data_test.to_csv('F:/materi/3-CLC/CLC 2.0/test_set_with_predictions-knn.csv'.format(len(data_train)), index=False)


# In[148]:


y_knn = data_test['Death1y']
scores_knn = pred_proba_knn[:, 1]
statistics_knn = bootstrap_auc(y_knn,scores_knn,[0,1])
print("均值:",np.mean(statistics_knn,axis=1))
print("最大值:",np.max(statistics_knn,axis=1))
print("最小值:",np.min(statistics_knn,axis=1))


# In[ ]:





# # 朴素叶贝斯

# 朴素贝叶斯有没有超参数可以调？
# 朴素贝叶斯是没有超参数可以调的，所以它不需要调参，朴素贝叶斯是根据训练集进行分类，分类出来的结果基本上就是确定了的，拉普拉斯估计器不是朴素贝叶斯中的参数，不能通过拉普拉斯估计器来对朴素贝叶斯调参。

# In[ ]:


mlt = MultinomialNB(alpha=1.0) # alpha拉普拉斯平滑系数

print(x_train.toarray())

mlt.fit(x_train, y_train)

y_predict = mlt.predict(x_test)

print("预测的文章类别为：", y_predict)


# In[150]:


from sklearn.naive_bayes import GaussianNB


# In[151]:


gnb_clf=GaussianNB() 
plot_learning_curves(gnb_clf, data_train_X, data_train['Death1y'])


# In[152]:


gnb_clf = gnb_clf.fit(data_train_X, data_train['Death1y'])


# In[153]:


gnb_clf.fit(data_train_X, data_train['Death1y'])
with open('F:/materi/3-CLC/CLC 2.0/gnb_clf_final_round.pkl', 'wb') as f:
    pickle.dump(gnb_clf, f)


# In[38]:


# Import model and retrain
with open('F:/materi/3-CLC/CLC 2.0/gnb_clf_final_round.pkl', 'rb') as f:
    gnb_clf = pickle.load(f)
gnb_clf.fit(data_train_X, data_train['Death1y'])


# In[155]:


accu_gnb = accuracy_score(data_test['Death1y'], gnb_clf.predict(data_test_X))
round(accu_gnb,3)


# In[156]:


pd.crosstab(data_test['Death1y'], gnb_clf.predict(data_test_X))


# In[157]:


pred_proba_gnb = gnb_clf.predict_proba(data_test_X)
fpr, tpr, _ = roc_curve(data_test['Death1y'], pred_proba_gnb[:, 1])
auc_gnb = roc_auc_score(data_test['Death1y'], pred_proba_gnb[:, 1])
round(auc_gnb,3)


# In[158]:


gnb_score = gnb_clf.score(data_test_X, data_test['Death1y'])
gnb_accuracy_score=accuracy_score(data_test['Death1y'],gnb_clf.predict(data_test_X))
gnb_preci_score=precision_score(data_test['Death1y'],gnb_clf.predict(data_test_X))
gnb_recall_score=recall_score(data_test['Death1y'],gnb_clf.predict(data_test_X))
gnb_f1_score=f1_score(data_test['Death1y'],gnb_clf.predict(data_test_X))
gnb_auc=roc_auc_score(data_test['Death1y'],pred_proba_gnb[:, 1])
print('gnb_accuracy_score: %f,gnb_preci_score: %f,gnb_recall_score: %f,gnb_f1_score: %f,gnb_auc: %f'
      %(gnb_accuracy_score,gnb_preci_score,gnb_recall_score,gnb_f1_score,gnb_auc))


# In[159]:


brier_score_loss(data_test['Death1y'], pred_proba_gnb[:, 1])


# In[160]:


print('loss gnb:', log_loss(data_test['Death1y'], pred_proba_gnb[:, 1]))


# In[161]:


plot_roc_curve(fpr, tpr, round(auc_gnb,3), gnb_clf)


# In[162]:


data_test['lr_pred_proba'] = pred_proba_gnb[:, 1]


# In[163]:


data_test.to_csv('F:/materi/3-CLC/CLC 2.0/test_set_with_predictions-gnb.csv'.format(len(data_train)), index=False)


# In[164]:


y_gnb = data_test['Death1y']
scores_gnb = pred_proba_gnb[:, 1]
statistics_gnb = bootstrap_auc(y_gnb,scores_gnb,[0,1])
print("均值:",np.mean(statistics_gnb,axis=1))
print("最大值:",np.max(statistics_gnb,axis=1))
print("最小值:",np.min(statistics_gnb,axis=1))


# In[ ]:





# In[ ]:





# # Random Forests classifier---随机搜索模型调参RandomizedSearchCV
# Random forests classifier is an ensemble tree-based model that reduces the variance of the predictors.
# 
# plot the learning curve to find out where the default model is at

# In[ ]:





# In[165]:


rf_clf = RandomForestClassifier(random_state=42)
plot_learning_curves(rf_clf, data_train_X, data_train['Death1y'])


# In[166]:


param_distribs = {
        'n_estimators': stats.randint(low=1, high=50),
         'max_features': ['auto', 'log2'],
        'max_depth': stats.randint(low=1, high=100),
        'min_samples_split': stats.randint(low=2, high=200), 
        'min_samples_leaf': stats.randint(low=2, high=200)
    }
rf_clf = RandomForestClassifier(random_state=42)
rnd_search = RandomizedSearchCV(rf_clf, param_distributions=param_distribs, return_train_score=True,
                                n_iter=100, cv=5, scoring='roc_auc', random_state=42, n_jobs=-1)
gs=rnd_search.fit(data_train_X, data_train['Death1y'])


# In[167]:


print(gs.best_score_)


# In[168]:


print(gs.best_params_)


# In[169]:


cv_rlt = rnd_search.cv_results_
pd.DataFrame(sorted(list(zip(cv_rlt["mean_test_score"], cv_rlt['mean_train_score'], cv_rlt["params"])), 
                    key=lambda x: x[0], reverse=True), columns=['X-val scores', 'Train scores', 'Params'])


# In[170]:


rf_clf = rnd_search.best_estimator_
rf_clf.fit(data_train_X, data_train['Death1y'])
with open('F:/materi/3-CLC/CLC 2.0/rf_clf_final_round.pkl', 'wb') as f:
    pickle.dump(rf_clf, f)


# In[171]:


plot_learning_curves(rf_clf, data_train_X, data_train['Death1y'])


# # Random forests model-ROC

# In[172]:


# Import model and retrain
with open('F:/materi/3-CLC/CLC 2.0/rf_clf_final_round.pkl', 'rb') as f:
    rf_clf = pickle.load(f)
rf_clf.fit(data_train_X, data_train['Death1y'])


# Accuracy scores

# In[173]:


accu_rf = accuracy_score(data_test['Death1y'], rf_clf.predict(data_test_X))


# In[174]:


round(accu_rf,3)


# In[175]:


pd.crosstab(data_test['Death1y'], rf_clf.predict(data_test_X))


# ROC and AUC

# In[176]:


pred_proba_rf = rf_clf.predict_proba(data_test_X)


# In[177]:


fpr, tpr, _ = roc_curve(data_test['Death1y'], pred_proba_rf[:, 1])
auc_rf = roc_auc_score(data_test['Death1y'], pred_proba_rf[:, 1])


# In[178]:


rf_score = rf_clf.score(data_test_X, data_test['Death1y'])
rf_accuracy_score=accuracy_score(data_test['Death1y'],rf_clf.predict(data_test_X))
rf_preci_score=precision_score(data_test['Death1y'],rf_clf.predict(data_test_X))
rf_recall_score=recall_score(data_test['Death1y'],rf_clf.predict(data_test_X))
rf_f1_score=f1_score(data_test['Death1y'],rf_clf.predict(data_test_X))
rf_auc=roc_auc_score(data_test['Death1y'],pred_proba_rf[:, 1])
print('rf_accuracy_score: %f,rf_preci_score: %f,rf_recall_score: %f,rf_f1_score: %f,rf_auc: %f'
      %(rf_accuracy_score,rf_preci_score,rf_recall_score,rf_f1_score,rf_auc))


# accuracy计算第二种方法

# In[179]:


from sklearn import metrics
def calculate_accuracy_sklearn(y, y_pred):
    return metrics.accuracy_score(y, y_pred)
calculate_accuracy_sklearn(data_test['Death1y'],rf_clf.predict(data_test_X))


# Brier score计算

# In[180]:


from sklearn.metrics import brier_score_loss
brier_score_loss(data_test['Death1y'], pred_proba_rf[:, 1])


# 第二种方法计算Brier score

# In[223]:


def brier_score(y, y_pred):
    s=0
    for i, j in zip(y, y_pred):
        s += (j-i)**2
    return s * (1/len(y))


# In[224]:


brier_score(data_test['Death1y'], pred_proba_rf[:, 1])


# Log loss计算

# In[181]:


from sklearn.metrics import log_loss
print('loss rf:', log_loss(data_test['Death1y'], pred_proba_rf[:, 1]))


# In[ ]:





# In[182]:


round(auc_rf,3)


# In[183]:


plot_roc_curve(fpr, tpr, round(auc_rf,3), rf_clf)


# In[184]:


data_test['lr_pred_proba'] = pred_proba_rf[:, 1]


# In[185]:


data_test.to_csv('F:/materi/3-CLC/CLC 2.0/test_set_with_predictions-Random forests model.csv'.format(len(data_train)), index=False)


# In[186]:


y_rf = data_test['Death1y']
scores_rf = pred_proba_rf[:, 1]
statistics_rf = bootstrap_auc(y_rf,scores_rf,[0,1])
print("均值:",np.mean(statistics_rf,axis=1))
print("最大值:",np.max(statistics_rf,axis=1))
print("最小值:",np.min(statistics_rf,axis=1))


# In[ ]:





# # Neural Network

# In[29]:


from sklearn.neural_network import MLPClassifier


# In[30]:


nn_clf = MLPClassifier(random_state=42)
plot_learning_curves(nn_clf, data_train_X, data_train['Death1y'])


# In[90]:


from keras.wrappers.scikit_learn import KerasRegressor###目前的存在的问题tensorflow 2.0版本以及以下才有compat


# In[87]:


conda install tensorflow==2.0.0


# In[91]:


from tensorflow import compat


# In[89]:


conda install tensorflow


# In[83]:


from keras.wrappers.scikit_learn import KerasClassifier


# In[75]:


import tensorflow as tf2
tf = tf2.compat.v1
tf.disable_v2_behavior()
# 跟原来的在理论上没有区别


# In[79]:


import tensorflow._api.v2.compat.v1 as tf
#tf.disable_v2_behavior()


# In[ ]:





# In[3]:


#conda install tensorflow ##加载工具包，最好使用conda pip install加载不上的，用这个可以加载上去。


# In[51]:


def create_network(optimizer='rmsprop', neurons=16, learning_rate=0.001):
    
    # Start Artificial Neural Network
    network = Sequential()
    
    # Adding the input layer and the first hidden layer
    network.add(Dense(units = neurons, 
                  activation = tf.keras.layers.LeakyReLU(alpha=0.3)))

    # Adding the second hidden layer
    network.add(Dense(units = neurons, 
                  activation = tf.keras.layers.LeakyReLU(alpha=0.3)))

    # Adding the third hidden layer
    network.add(Dense(units = neurons, 
                  activation = tf.keras.layers.LeakyReLU(alpha=0.3)))

    # Adding the output layer
    network.add(Dense(units = 1))

    ###############################################
    # Add optimizer with learning rate
    if optimizer == 'rmsprop':
        opt = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
    elif optimizer == 'adam':
        opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer == 'SGD':
        opt = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    else:
        raise ValueError('optimizer {} unrecognized'.format(optimizer))
    ##############################################    

    # Compile NN
    network.compile(optimizer = opt, 
                loss = 'mean_squared_error', 
                metrics=['mae', tf.keras.metrics.RootMeanSquaredError()])
    
    # Return compiled network
    return network

# Wrap Keras model so it can be used by scikit-learn
ann = KerasRegressor(build_fn=create_network, verbose=0)

# Create hyperparameter space
epoch_values = [10, 25, 50, 100, 150, 200]
batches = [10, 20, 30, 40, 50, 100, 1000]
optimizers = ['rmsprop', 'adam', 'SGD']
neuron_list = [16, 32, 64, 128, 256]
lr_values = [0.001, 0.01, 0.1, 0.2, 0.3]

# Create hyperparameter options
hyperparameters = dict(
    epochs=epoch_values, 
    batch_size=batches, 
    optimizer=optimizers, 
    neurons=neuron_list,
    learning_rate=lr_values)

# Create grid search
# cv=5 is the default 5-fold
grid = GridSearchCV(estimator=ann, cv=5, param_grid=hyperparameters)

# Fit grid search
grid_result = grid.fit(data_train_X, data_train['Death1y'])


# In[ ]:





# In[ ]:





# In[ ]:





# In[2]:


from keras.optimizers import SGD


# In[ ]:


# 构建模型的函数
def create_model(learn_rate=0.01, momentum=0):
    # 创建模型
    model = Sequential()
    model.add(Dense(12, input_dim=8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    # 编译模型
    optimizer = SGD(lr=learn_rate, momentum=momentum)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

# 为了复现，设置随机种子
seed = 7
np.random.seed(seed)

# 加载数据
dataset = np.loadtxt("pima-indians-diabetes.csv", delimiter=",")
# 切分数据为输入 X 和输出 Y
X = dataset[:,0:8]
Y = dataset[:,8]

# 创建模型，使用到了上一步找出的 epochs、batch size 最优参数
model = KerasClassifier(build_fn=create_model, epochs=100, batch_size=80, verbose=0)
# 定义网格搜索参数
learn_rate = [0.001, 0.01, 0.1, 0.2, 0.3]
momentum = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]
param_grid = dict(learn_rate=learn_rate, momentum=momentum)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1)
grid_result = grid.fit(X, Y)

# 总结结果
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[31]:


nn_clf = MLPClassifier(random_state=42,activation='relu',alpha=0.0001,batch_size='auto',beta_1=0.9, beta_2=0.999, 
                       early_stopping=False,epsilon=1e-08,hidden_layer_sizes=(100),
                       learning_rate='constant', learning_rate_init=0.001,max_iter=200, momentum=0.9, 
                       n_iter_no_change=10, nesterovs_momentum=True, power_t=0.5,shuffle=True,  
                       tol=0.0001, validation_fraction=0.1,verbose=False, warm_start=False)


# In[32]:


nn_clf.fit(data_train_X, data_train['Death1y'])
nn_clf_y_pre=nn_clf.predict(data_test_X)
nn_clf_y_proba=nn_clf.predict_proba(data_test_X)


# In[33]:


from sklearn.metrics import f1_score,precision_score,recall_score,roc_auc_score,accuracy_score,roc_curve


# In[34]:


nn_clf_accuracy_score=accuracy_score(data_test['Death1y'],nn_clf_y_pre)
nn_clf_preci_score=precision_score(data_test['Death1y'],nn_clf_y_pre)
nn_clf_recall_score=recall_score(data_test['Death1y'],nn_clf_y_pre)
nn_clf_f1_score=f1_score(data_test['Death1y'],nn_clf_y_pre)
nn_clf_auc=roc_auc_score(data_test['Death1y'],nn_clf_y_proba[:,1])
print('nn_clf_accuracy_score: %f,nn_clf_preci_score: %f,nn_clf_recall_score: %f,nn_clf_f1_score: %f,nn_clf_auc: %f'
      %(nn_clf_accuracy_score,nn_clf_preci_score,nn_clf_recall_score,nn_clf_f1_score,nn_clf_auc))


# In[35]:


nn_clf_fpr,nn_clf_tpr,nn_clf_threasholds=roc_curve(data_test['Death1y'],nn_clf_y_proba[:,1]) # 计算ROC的值,svm_threasholds为阈值
plt.title("roc_curve of %s(AUC=%.4f)" %('nn_clf',nn_clf_auc))
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.plot(nn_clf_fpr,nn_clf_tpr)
plt.show()


# In[36]:


data_test['lr_pred_proba'] = nn_clf_y_proba[:,1]


# In[37]:


data_test.to_csv('F:/materi/3-CLC/CLC 2.0/test_set_with_predictions-nn.csv'.format(len(data_train)), index=False)


# In[38]:


fpr, tpr, _ = roc_curve(data_test['Death1y'], nn_clf_y_proba[:,1])
auc_nn = roc_auc_score(data_test['Death1y'], nn_clf_y_proba[:,1])


# In[40]:


brier_score_loss(data_test['Death1y'], nn_clf_y_proba[:,1])


# In[41]:


#round(auc_nn,3)


# In[42]:


print('loss dt:', log_loss(data_test['Death1y'], nn_clf_y_proba[:,1]))


# In[ ]:





# In[ ]:




直接形成
# In[280]:


continuous_vars = ["Hb","HDL","albumin","Scr","NT.proBNP"]
categorical_vars = ["CHF1","Statins"]
data_train_X = pd.DataFrame(data_train, 
                            columns=continuous_vars + categorical_vars)


# In[281]:


data_train_X.head


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# # Gradient boosting classifier---随机搜索模型调参RandomizedSearchCV
# Gradient boosting classifier is an ensemble tree-based model that reduces the bias of the predictors.

# In[187]:


plot_learning_curves(GradientBoostingClassifier(random_state=42), data_train_X, data_train['Death1y'])


# In[188]:


param_distribs = {
        'n_estimators': stats.randint(low=80, high=200),
         'max_features': ['auto', 'log2'],
        'max_depth': stats.randint(low=1, high=100),
        'min_samples_split': stats.randint(low=2, high=200), 
        'min_samples_leaf': stats.randint(low=2, high=200),
    }

rnd_search = RandomizedSearchCV(GradientBoostingClassifier(random_state=42), 
                                param_distributions=param_distribs, return_train_score=True,
                                n_iter=100, cv=5, scoring='roc_auc', random_state=42, n_jobs=-1)
# this will take a long time
gsgbm = rnd_search.fit(data_train_X, data_train['Death1y'])


# In[189]:


print(gsgbm.best_score_)


# In[190]:


print(gsgbm.best_params_)


# In[191]:


cv_rlt = rnd_search.cv_results_
pd.DataFrame(sorted(list(zip(cv_rlt["mean_test_score"], cv_rlt['mean_train_score'], cv_rlt["params"])), 
                    key=lambda x: x[0], reverse=True), columns=['X-val scores', 'Train scores', 'Params'])


# In[196]:


gbm_clf = rnd_search.best_estimator_
gbm_clf.fit(data_train_X, data_train['Death1y'])
with open('F:/materi/3-CLC/CLC 2.0/gbm_clf_final_round.pkl', 'wb') as f:
    pickle.dump(gbm_clf, f)


# In[197]:


plot_learning_curves(gbm_clf, data_train_X, data_train['Death1y'])


# # Gradient boosting machine model-ROC

# In[198]:


# Import model and retrain
with open('F:/materi/3-CLC/CLC 2.0/gbm_clf_final_round.pkl', 'rb') as f:
    gbm_clf = pickle.load(f)
gbm_clf.fit(data_train_X, data_train['Death1y'])


# Accuracy scores

# In[199]:


accu_gbm = accuracy_score(data_test['Death1y'], gbm_clf.predict(data_test_X))


# In[200]:


round(accu_gbm,3)


# In[201]:


pd.crosstab(data_test['Death1y'], gbm_clf.predict(data_test_X))


# ROC and AUC

# In[ ]:





# In[202]:


pred_proba_gbm = gbm_clf.predict_proba(data_test_X)


# In[203]:


fpr, tpr, _ = roc_curve(data_test['Death1y'], pred_proba_gbm[:, 1])
auc_gbm = roc_auc_score(data_test['Death1y'], pred_proba_gbm[:, 1])


# In[204]:


round(auc_gbm,3)


# In[205]:


gbm_score = gbm_clf.score(data_test_X, data_test['Death1y'])
gbm_accuracy_score=accuracy_score(data_test['Death1y'],gbm_clf.predict(data_test_X))
gbm_preci_score=precision_score(data_test['Death1y'],gbm_clf.predict(data_test_X))
gbm_recall_score=recall_score(data_test['Death1y'],gbm_clf.predict(data_test_X))
gbm_f1_score=f1_score(data_test['Death1y'],gbm_clf.predict(data_test_X))
gbm_auc=roc_auc_score(data_test['Death1y'],pred_proba_gbm[:, 1])
print('gbm_accuracy_score: %f,gbm_preci_score: %f,gbm_recall_score: %f,gbm_f1_score: %f,gbm_auc: %f'
      %(gbm_accuracy_score,gbm_preci_score,gbm_recall_score,gbm_f1_score,gbm_auc))


# In[206]:


brier_score_loss(data_test['Death1y'], pred_proba_gbm[:, 1])


# In[207]:


print('loss gbm:', log_loss(data_test['Death1y'], pred_proba_gbm[:, 1]))


# In[208]:


plot_roc_curve(fpr, tpr, round(auc_gbm,3), gbm_clf)


# In[209]:


data_test['lr_pred_proba'] = pred_proba_gbm[:, 1]


# In[210]:


data_test.to_csv('F:/materi/3-CLC/CLC 2.0/test_set_with_predictions-Gradient boosting machine model.csv'.format(len(data_train)), index=False)


# In[211]:


y = data_test['Death1y']
scores_gbm = pred_proba_gbm[:, 1]
statistics_gbm = bootstrap_auc(y,scores_gbm,[0,1])
print("均值:",np.mean(statistics_gbm,axis=1))
print("最大值:",np.max(statistics_gbm,axis=1))
print("最小值:",np.min(statistics_gbm,axis=1))


# In[ ]:





# In[234]:


import shap


# In[235]:


shap.initjs()


# In[256]:


# train an model
model = gbm_clf.fit(data_train_X, data_train['Death1y'])


# In[257]:


# explain the model's predictions using SHAP
# (same syntax works for LightGBM, CatBoost, scikit-learn, transformers, Spark, etc.)
explainer = shap.Explainer(model)
shap_values = explainer(data_test_X)###修改此处为不同数据集里面进行运算哦！！！

# visualize the first prediction's explanation
shap.waterfall_plot(explainer.base_values[0])
# In[238]:


shap.initjs()
# visualize the first prediction's explanation with a force plot
shap.plots.force(shap_values[0])


# In[245]:


shap.initjs()
# visualize the first prediction's explanation with a force plot
shap.plots.force(shap_values[38])


# In[262]:


import os
plt.tight_layout()
#plt.savefig('images.png')
plt.savefig(os.path.join('E:/R code-LMX/Yimin-medical disputes/Models', 'GBM-julei.png'))
plt.rcParams['savefig.dpi'] = 1200 #图片像素
plt.rcParams['figure.dpi'] = 1200 #分辨率
plt.figure(figsize=(8,10))
plt.show()


# In[ ]:


shap.plots.beeswarm(shap_values)###


# In[ ]:


shap.plots.bar(shap_values)


# In[ ]:





# In[265]:


shap_values = explainer(data_test_X[:1000])##数据库前1000个


# In[ ]:


shap.plots.heatmap(shap_values)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# # Support vector machine classifier---网格搜索模型调参GridSearchCV
# Support vector machine classifier is a powerful classifier that works best on small to medium size complex data set. Our training set is medium size to SVMs.
# 
# plot the learning curve to find out where the default model is at

# In[ ]:





# Try Linear SVC fist

# In[212]:


plot_learning_curves(LinearSVC(loss='hinge', random_state=42), data_train_X, data_train['Death1y'])


# Try Polynomial kernel

# In[213]:


plot_learning_curves(SVC(kernel='poly', random_state=42), data_train_X, data_train['Death1y'])


# Try Gaussian RBF kernel

# In[214]:


plot_learning_curves(SVC(random_state=42), data_train_X, data_train['Death1y'])


# In[ ]:





# 第二调参

# In[215]:


hyperparameters = {
 "C": stats.uniform(0.001, 0.1),
 "gamma": stats.uniform(0, 0.5),
 'kernel': ('linear', 'rbf')
}
random = RandomizedSearchCV(estimator = SVC(probability=True), param_distributions = hyperparameters, n_iter = 100, 
                            cv = 5, return_train_score=True, random_state=42, n_jobs = -1)
gssvm = random.fit(data_train_X, data_train['Death1y'])


# In[216]:


print(gssvm.best_score_)


# In[217]:


print(gssvm.best_params_)


# In[218]:


cv_rlt = random.cv_results_
pd.DataFrame(sorted(list(zip(cv_rlt["mean_test_score"], cv_rlt['mean_train_score'], cv_rlt["params"])), 
                    key=lambda x: x[0], reverse=True), columns=['X-val scores', 'Train scores', 'Params'])


# In[219]:


svc_clf = random.best_estimator_
#svc_clf.fit(data_train_X, data_train['Death1y'])
with open('F:/materi/3-CLC/CLC 2.0/svc_clf_final_round.pkl', 'wb') as f:
    pickle.dump(svc_clf, f)


# In[220]:


# best model is the default RBF kernal SVM
plot_learning_curves(svc_clf, data_train_X, data_train['Death1y']) 


# # SVM-ROC

# In[37]:


# Import model and retrain
with open('F:/materi/3-CLC/CLC 2.0/svc_clf_final_round.pkl', 'rb') as f:
    svc_clf = pickle.load(f)
svc_clf.fit(data_train_X, data_train['Death1y'])


# In[222]:


accu_svc = accuracy_score(data_test['Death1y'], svc_clf.predict(data_test_X))


# In[223]:


round(accu_svc,3)


# In[224]:


pd.crosstab(data_test['Death1y'], svc_clf.predict(data_test_X))


# In[225]:


pred_proba_svc = svc_clf.predict_proba(data_test_X) 


# In[226]:


fpr, tpr, _ = roc_curve(data_test['Death1y'], pred_proba_svc[:, 1])
auc_svc = roc_auc_score(data_test['Death1y'], pred_proba_svc[:, 1])


# In[227]:


round(auc_svc,3)


# In[228]:


svc_score = svc_clf.score(data_test_X, data_test['Death1y'])
svc_accuracy_score=accuracy_score(data_test['Death1y'],svc_clf.predict(data_test_X))
svc_preci_score=precision_score(data_test['Death1y'],svc_clf.predict(data_test_X))
svc_recall_score=recall_score(data_test['Death1y'],svc_clf.predict(data_test_X))
svc_f1_score=f1_score(data_test['Death1y'],svc_clf.predict(data_test_X))
svc_auc=roc_auc_score(data_test['Death1y'],pred_proba_svc[:, 1])
print('svc_accuracy_score: %f,svc_preci_score: %f,svc_recall_score: %f,svc_f1_score: %f,svc_auc: %f'
      %(svc_accuracy_score,svc_preci_score,svc_recall_score,svc_f1_score,svc_auc))


# In[229]:


brier_score_loss(data_test['Death1y'], pred_proba_svc[:, 1])


# In[230]:


print('loss svc:', log_loss(data_test['Death1y'], pred_proba_svc[:, 1]))


# In[231]:


plot_roc_curve(fpr, tpr, round(auc_svc,3), svc_clf)


# In[232]:


data_test['lr_pred_proba'] = pred_proba_svc[:, 1]


# In[233]:


data_test.to_csv('F:/materi/3-CLC/CLC 2.0/test_set_with_predictions-Support vector machine model.csv'.format(len(data_train)), index=False)


# In[234]:


y = data_test['Death1y']
scores_svc = pred_proba_svc[:, 1]
statistics_svc = bootstrap_auc(y,scores_svc,[0,1])
print("均值:",np.mean(statistics_svc,axis=1))
print("最大值:",np.max(statistics_svc,axis=1))
print("最小值:",np.min(statistics_svc,axis=1))


# In[ ]:





# # Ensemble classifier
# Scikit-learn offers a voting classifier which aggregates the prediction of multiple predictors and is a flexible ensemble technique that allows an ensemble of different models.
# For the final classifier, simply aggregate the predictions of the three best models, i.e., random forests, gradien boosting machine and the support vector machine.

# In[39]:


ensemble_clf = VotingClassifier(estimators=[('lr', lr_clf),('gnb',gnb_clf), ('Xgbc', Xgbc_clf), ('svc', svc_clf),('dt',dt_clf)],
                             voting='soft')
ensemble_clf.fit(data_train_X, data_train['Death1y'])


# Check out its learning curve.

# In[236]:


plot_learning_curves(ensemble_clf, data_train_X, data_train['Death1y'])


# In[237]:


with open('F:/materi/3-CLC/CLC 2.0/ensemble_clf_final_round.pkl', 'wb') as f:
    pickle.dump(ensemble_clf, f)


# # Ensemble-ROC

# In[32]:


# Import model and retrain
with open('F:/materi/3-CLC/CLC 2.0/ensemble_clf_final_round.pkl', 'rb') as f:
    ensemble_clf = pickle.load(f)
ensemble_clf.fit(data_train_X, data_train['Death1y'])


# In[239]:


ensemble_clf


# In[240]:


accu_ensemble = accuracy_score(data_test['Death1y'], ensemble_clf.predict(data_test_X))


# In[241]:


round(accu_ensemble,3)


# In[242]:


pd.crosstab(data_test['Death1y'], ensemble_clf.predict(data_test_X))


# In[243]:


pred_proba_ensemble = ensemble_clf.predict_proba(data_test_X)


# In[244]:


fpr, tpr, _ = roc_curve(data_test['Death1y'], pred_proba_ensemble[:, 1])
auc_ensemble = roc_auc_score(data_test['Death1y'], pred_proba_ensemble[:, 1])


# In[245]:


round(auc_ensemble,3)


# In[246]:


ensemble_score = ensemble_clf.score(data_test_X, data_test['Death1y'])
ensemble_accuracy_score=accuracy_score(data_test['Death1y'],ensemble_clf.predict(data_test_X))
ensemble_preci_score=precision_score(data_test['Death1y'],ensemble_clf.predict(data_test_X))
ensemble_recall_score=recall_score(data_test['Death1y'],ensemble_clf.predict(data_test_X))
ensemble_f1_score=f1_score(data_test['Death1y'],ensemble_clf.predict(data_test_X))
ensemble_auc=roc_auc_score(data_test['Death1y'],pred_proba_ensemble[:, 1])
print('ensemble_accuracy_score: %f,ensemble_preci_score: %f,ensemble_recall_score: %f,ensemble_f1_score: %f,ensemble_auc: %f'
      %(ensemble_accuracy_score,ensemble_preci_score,ensemble_recall_score,ensemble_f1_score,ensemble_auc))


# In[247]:


brier_score_loss(data_test['Death1y'], pred_proba_ensemble[:, 1])


# In[248]:


print('loss ensemble:', log_loss(data_test['Death1y'], pred_proba_ensemble[:, 1]))


# In[249]:


plot_roc_curve(fpr, tpr, round(auc_ensemble,3), ensemble_clf)


# In[250]:


data_test['lr_pred_proba'] = pred_proba_ensemble[:, 1]


# In[251]:


data_test.to_csv('F:/materi/3-CLC/CLC 2.0/test_set_with_predictions-The ensemble model.csv'.format(len(data_train)), index=False)


# In[252]:


y = data_test['Death1y']
scores_ensemble = pred_proba_ensemble[:, 1]
statistics_ensemble = bootstrap_auc(y,scores_ensemble,[0,1])
print("均值:",np.mean(statistics_ensemble,axis=1))
print("最大值:",np.max(statistics_ensemble,axis=1))
print("最小值:",np.min(statistics_ensemble,axis=1))


# In[ ]:





# In[40]:


import shap
shap.initjs()
model = ensemble_clf.fit(data_train_X, data_train['Death1y'])


# In[41]:


X_train_summary = shap.kmeans(data_train_X, 10)


# In[47]:


explainer = shap.KernelExplainer(model.predict, X_train_summary)


# In[42]:


# explain the model's predictions using SHAP
# (same syntax works for LightGBM, CatBoost, scikit-learn, transformers, Spark, etc.)
explainer = shap.KernelExplainer(model.predict, data_train_X)


# In[45]:


shap_values = explainer.shap_values(data_test_X)


# In[48]:


shap.summary_plot(shap_values, data_train_X)


# In[46]:


shap.summary_plot(shap_values, data_test_X)


# In[ ]:





# # 总结100 bootstrap

# In[ ]:





# In[255]:


# coding=utf-8
import matplotlib.pyplot as plt


game = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,
              38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,
              76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100])

plt.figure(figsize=(20, 10), dpi=100)
#game = ['1', '2', '1-G3', '1-G4', '1-G5', '2-G1', '2-G2', '2-G3', '2-G4', '2-G5', '3-G1', '3-G2', '3-G3',
       # '3-G4', '3-G5', '总决赛-G1', '总决赛-G2', '总决赛-G3', '总决赛-G4', '总决赛-G5', '总决赛-G6']
#scores = statistics[1, :]
#plt.plot(game, scores)
plt.ylim(0, 1)

plt.plot(game, statistics_lr[1, :], c='red', label="Logistic Regression (95% CI:)")
plt.plot(game, statistics_Xgbc[1, :], c='green', label="XGBoosting Machine (95% CI:)")#linestyle='--',
plt.plot(game, statistics_dt[1, :], c='blue', label="Decision Tree (95% CI:)")#linestyle='-.', 
plt.plot(game, statistics_rf[1, :], c='purple', label="Random Forest (95% CI:)")
plt.plot(game, statistics_gbm[1, :], c='black', label="Gradient Boosting Machine (95% CI:)")
plt.plot(game, statistics_svc[1, :], c='chocolate', label="Support Vector Machine (95% CI:)")
plt.plot(game, statistics_ensemble[1, :], c='darkorange', label="Ensemble model (95% CI:)")

plt.scatter(game, statistics_lr[1, :], c='red')
plt.scatter(game, statistics_Xgbc[1, :], c='green')
plt.scatter(game, statistics_dt[1, :], c='blue')
plt.scatter(game, statistics_rf[1, :], c='purple')
plt.scatter(game, statistics_gbm[1, :], c='black')
plt.scatter(game, statistics_svc[1, :], c='chocolate')
plt.scatter(game, statistics_ensemble[1, :], c='darkorange')

plt.legend(loc='best')
#plt.yticks(range(0, 50, 5))
plt.grid(True, linestyle='--', alpha=0.5)
plt.xlabel("Number of bootstraps", fontdict={'size': 16})
plt.ylabel("AUC", fontdict={'size': 16})
plt.title("AUC (Bootstraps=100)", fontdict={'size': 20})

plt.legend(loc='lower right',fontsize=16)
#plt.xlabel('X-axis',fontproperties=font_set) #X轴标签
#plt.ylabel("Y-axis",fontproperties=font_set) #Y轴标签
#plt.legend(loc='upper left', bbox_to_anchor=(1, 1))#图注的位置设置：upper right，upper left，lower left，
#lower right，right，center left，center right，lower center，upper center，center

plt.yticks(size=15)#设置大小及加粗fontproperties='Times New Roman', ,weight='bold'
plt.xticks(size=15)

plt.show()


# JAMA配色

# In[267]:


# coding=utf-8
import matplotlib.pyplot as plt


game = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,
              38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,
              76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100])

plt.figure(figsize=(20, 10), dpi=600)##设置画布大小和像素参数。
#game = ['1', '2', '1-G3', '1-G4', '1-G5', '2-G1', '2-G2', '2-G3', '2-G4', '2-G5', '3-G1', '3-G2', '3-G3',
       # '3-G4', '3-G5', '总决赛-G1', '总决赛-G2', '总决赛-G3', '总决赛-G4', '总决赛-G5', '总决赛-G6']
#scores = statistics[1, :]
#plt.plot(game, scores)
plt.ylim(0, 1)

plt.plot(game, statistics_lr[1, :], c='#BC3C29FF', label="Logistic Regression (0.795, 95% CI: 0.756-0.829)")
plt.plot(game, statistics_gnb[1, :], c='#0072B5FF', label="Naive Bayesian (0.828, 95% CI: 0.793-0.863)")#linestyle='--',
plt.plot(game, statistics_Xgbc[1, :], c='#E18727FF', label="XGBoosting Machine (0.763, 95% CI: 0.724-0.791)")#linestyle='-.', 
plt.plot(game, statistics_svc[1, :], c='#20854EFF', label="Support Vector Machine (0.805, 95% CI: 0.771-0.839)")
plt.plot(game, statistics_dt[1, :], c='#7876B1FF', label="Decision Tree (0.673, 95% CI: 0.634-0.715)")
plt.plot(game, statistics_ensemble[0, :], c='#6F99ADFF', label="Ensemble Model (0.828, 95% CI: 0.805-0.870)")
#plt.plot(game, statistics_ensemble[1, :], c='#FFDC91FF', label="Ensemble model (95% CI:)")

plt.scatter(game, statistics_lr[1, :], c='#BC3C29FF')
plt.scatter(game, statistics_gnb[1, :], c='#0072B5FF')
plt.scatter(game, statistics_Xgbc[1, :], c='#E18727FF')
plt.scatter(game, statistics_svc[1, :], c='#20854EFF')
plt.scatter(game, statistics_dt[1, :], c='#7876B1FF')
plt.scatter(game, statistics_ensemble[0, :], c='#6F99ADFF')
#plt.scatter(game, statistics_ensemble[1, :], c='#FFDC91FF')

plt.legend(loc='best')
#plt.yticks(range(0, 50, 5))
plt.grid(True, linestyle='--', alpha=0.5)
plt.xlabel("Number of bootstraps", fontdict={'size': 20})
plt.ylabel("AUC", fontdict={'size': 20})
plt.title("AUC (Bootstraps=100)", fontdict={'size': 20})

plt.legend(loc='lower right',fontsize=16)
#plt.xlabel('X-axis',fontproperties=font_set) #X轴标签
#plt.ylabel("Y-axis",fontproperties=font_set) #Y轴标签
#plt.legend(loc='upper left', bbox_to_anchor=(1, 1))#图注的位置设置：upper right，upper left，lower left，
#lower right，right，center left，center right，lower center，upper center，center

plt.yticks(size=15)#设置大小及加粗fontproperties='Times New Roman', ,weight='bold'
plt.xticks(size=15)

plt.show()


# In[ ]:


#BC3C29FF
#0072B5FF
#E18727FF
#20854EFF
#7876B1FF
#6F99ADFF
#FFDC91FF
#EE4C97FF

[('lr', lr_clf), ('Xgbc', Xgbc_clf), ('rf', rf_clf), ('gbm', gbm_clf), ('svc', svc_clf),('dt',dt_clf)],
                             voting='soft')
    
plt.plot(game, statistics_lr[1, :], c='#BC3C29FF', label="Logistic Regression (0.795, 95% CI: 0.756-0.829)")
plt.plot(game, statistics_gnb[1, :], c='#0072B5FF', label="Naive Bayesian (0.828, 95% CI: 0.793-0.863)")#linestyle='--',
plt.plot(game, statistics_Xgbc[1, :], c='#E18727FF', label="XGBoosting Machine (0.763, 95% CI: 0.724-0.791)")#linestyle='-.', 
plt.plot(game, statistics_svc[1, :], c='#20854EFF', label="Support Vector Machine (0.805, 95% CI: 0.771-0.839)")
plt.plot(game, statistics_dt[1, :], c='#7876B1FF', label="Decision Tree (0.673, 95% CI: 0.634-0.715)")
plt.plot(game, statistics_ensemble[0, :], c='#6F99ADFF', label="Ensemble model (0.828, 95% CI: 0.805-0.870)")


# 绘制模型calibration曲线

# In[33]:


clf_list = [
    (lr_clf, "Logistic Regression"),
    (gnb_clf, "Naive Bayesian"),
    (Xgbc_clf, "XGBoosting Machine"),
    (svc_clf, "Support Vector Machine"),
    (dt_clf, "Decision Tree"),
    (ensemble_clf, "Ensemble Model"),
]


# In[34]:


import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


# In[35]:


from sklearn.calibration import CalibrationDisplay


# In[36]:


fig = plt.figure(figsize=(10, 12), dpi=900)
gs = GridSpec(6, 2)###确定框架图6行，2列
#colors = plt.cm.get_cmap("Dark2")
colors = ['#BC3C29FF','#0072B5FF','#E18727FF','#20854EFF','#7876B1FF','#6F99ADFF','#FFDC91FF']

ax_calibration_curve = fig.add_subplot(gs[:3, :2])#calibration图占据前3行，前2列
calibration_displays = {}

for i, (clf, name) in enumerate(clf_list):
    clf.fit(data_train_X, data_train['Death1y'])
    display = CalibrationDisplay.from_estimator(
        clf,
        data_test_X,
        data_test['Death1y'],
        n_bins=10,
        name=name,
        ax=ax_calibration_curve,
        color=colors[i],
    )
    calibration_displays[name] = display

ax_calibration_curve.grid()
ax_calibration_curve.set_title("Calibration plots")

# Add histogram
grid_positions = [(3, 0), (3, 1), (4, 0), (4, 1),(5, 0),(5, 1)]##分别为第4行左、右；第5行左、右；第6行左、右
for i, (_, name) in enumerate(clf_list):
    row, col = grid_positions[i]
    ax = fig.add_subplot(gs[row, col])

    ax.hist(
        calibration_displays[name].y_prob,
        range=(0, 1),
        bins=10,
        label=name,
        #lable
        #fontdict={'size': 10},
        color=colors[i],
    )
    ax.set_title(name,fontsize = 12)
    ax.set_xlabel('Mean predicted probability', fontsize = 12)
    ax.set_ylabel('Count', fontsize = 12)

plt.tight_layout()
plt.show()


# In[306]:





# In[283]:


fig = plt.figure(figsize=(10, 16))
gs = GridSpec(4, 2)
#colors = plt.cm.get_cmap('#BC3C29FF','#0072B5FF')
colors = ['#BC3C29FF','#0072B5FF','#E18727FF','#20854EFF','#7876B1FF','#6F99ADFF','#FFDC91FF']
 
ax_calibration_curve = fig.add_subplot(gs[:2, :2])
calibration_displays = {}
for i, (clf, name) in enumerate(clf_list):
    clf.fit(data_train_X, data_train['Death1y'])
    display = CalibrationDisplay.from_estimator(
        clf,
        data_test_X,
        data_test['Death1y'],
        n_bins=10,
        name=name,
        ax=ax_calibration_curve,
        color=colors[i],
    )
    calibration_displays[name] = display
 
ax_calibration_curve.grid()
ax_calibration_curve.set_title("Calibration plots")

plt.xlabel("Mean predicted probability", fontdict={'size': 16})
plt.ylabel("Mean actual probability", fontdict={'size': 16})


plt.tight_layout()
plt.show()


# In[ ]:


ValueError: '#BC3C29FF' is not a valid value for name; supported values are 'Accent', 'Accent_r', 'Blues', 'Blues_r', 'BrBG', 
    'BrBG_r', 'BuGn', 'BuGn_r', 'BuPu', 'BuPu_r', 'CMRmap', 'CMRmap_r', 'Dark2', 'Dark2_r', 'GnBu', 'GnBu_r', 'Greens', 'Greens_r', 
    'Greys', 'Greys_r', 'OrRd', 'OrRd_r', 'Oranges', 'Oranges_r', 'PRGn', 'PRGn_r', 'Paired', 'Paired_r', 'Pastel1', 'Pastel1_r',
    'Pastel2', 'Pastel2_r', 'PiYG', 'PiYG_r', 'PuBu', 'PuBuGn', 'PuBuGn_r', 'PuBu_r', 'PuOr', 'PuOr_r', 'PuRd', 'PuRd_r', 'Purples',
    'Purples_r', 'RdBu', 'RdBu_r', 'RdGy', 'RdGy_r', 'RdPu', 'RdPu_r', 'RdYlBu', 'RdYlBu_r', 'RdYlGn', 'RdYlGn_r', 'Reds', 'Reds_r',
    'Set1', 'Set1_r', 'Set2', 'Set2_r', 'Set3', 'Set3_r', 'Spectral', 'Spectral_r', 'Wistia', 'Wistia_r', 'YlGn', 'YlGnBu', 'YlGnBu_r',
    'YlGn_r', 'YlOrBr', 'YlOrBr_r', 'YlOrRd', 'YlOrRd_r', 'afmhot', 'afmhot_r', 'autumn', 'autumn_r', 'binary', 'binary_r', 'bone', 
    'bone_r', 'brg', 'brg_r', 'bwr', 'bwr_r', 'cividis', 'cividis_r', 'cool', 'cool_r', 'coolwarm', 'coolwarm_r', 'copper', 'copper_r',
    'cubehelix', 'cubehelix_r', 'flag', 'flag_r', 'gist_earth', 'gist_earth_r', 'gist_gray', 'gist_gray_r', 'gist_heat', 'gist_heat_r',
    'gist_ncar', 'gist_ncar_r', 'gist_rainbow', 'gist_rainbow_r', 'gist_stern', 'gist_stern_r', 'gist_yarg', 'gist_yarg_r', 'gnuplot',
    'gnuplot2', 'gnuplot2_r', 'gnuplot_r', 'gray', 'gray_r', 'hot', 'hot_r', 'hsv', 'hsv_r', 'inferno', 'inferno_r', 'jet', 'jet_r',
    'magma', 'magma_r', 'nipy_spectral', 'nipy_spectral_r', 'ocean', 'ocean_r', 'pink', 'pink_r', 'plasma', 'plasma_r', 'prism', 'prism_r', 
    'rainbow', 'rainbow_r', 'seismic', 'seismic_r', 'spring', 'spring_r', 'summer', 'summer_r', 'tab10', 'tab10_r', 'tab20', 'tab20_r', 
    'tab20b', 'tab20b_r', 'tab20c', 'tab20c_r', 'terrain', 'terrain_r', 'turbo', 'turbo_r', 'twilight', 'twilight_r', 
    'twilight_shifted', 'twilight_shifted_r', 'viridis', 'viridis_r', 'winter', 'winter_r'


# In[ ]:





# In[247]:





# In[ ]:





# In[ ]:





# # h2o AutoML

# In[155]:


import h2o
from h2o.automl import H2OAutoML


# In[156]:


import h2o
from h2o.estimators.gbm import H2OGradientBoostingEstimator


# In[161]:


h2o.init()


# In[ ]:


data_train = h2o.import_file(r'E:/R code-LMX/Yimin-medical disputes/Data/start0.9.csv')##只能是csv文件，excle文件还不行哦！！！
#data_test = h2o.import_file(r'E:/R code-LMX/R code for SEER-Bone metastasis-LYS/Data/starv0.1.csv')
data_test = h2o.import_file(r'E:/R code-LMX/Yimin-medical disputes/External validation/4-data.csv')


# In[163]:


##airlines = h2o.import_file('https://s3.amazonaws.com/h2o-public-test-data/smalldata/airlines/allyears2k_headers.zip')


# In[164]:


data_train.head()


# In[ ]:





# In[17]:


data_train["Marital.status"] = data_train["Marital.status"].asfactor()
data_train["Rural.urban"] = data_train["Rural.urban"].asfactor()
data_train["Brain.m"] = data_train["Brain.m"].asfactor()
data_train["Liver.m"] = data_train["Liver.m"].asfactor()
data_train["Lung.m"] = data_train["Lung.m"].asfactor()
data_train["Primary.site"] = data_train["Primary.site"].asfactor()
data_train["Race"] = data_train["Race"].asfactor()
data_train["Tstage"] = data_train["Tstage"].asfactor()
data_train["Nstage"] = data_train["Nstage"].asfactor()
data_train["Cancer.directed.surgery"] = data_train["Cancer.directed.surgery"].asfactor()
data_train['Radiation'] = data_train['Radiation'].asfactor()
data_train['Chemotherapy'] = data_train['Chemotherapy'].asfactor()
data_train['Sex'] = data_train['Sex'].asfactor()


# In[21]:


predictors = ["Rural.urban",
"Marital.status",
"Race",
"Brain.m",
"Liver.m",
"Lung.m",
"Cancer.directed.surgery",
"Radiation",
"Chemotherapy",
"Tstage",
"Nstage",
"Primary.site",
"Sex","Age"]
response = 'Death1y'


# In[22]:


bin_num = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
label = ["8", "16", "32", "64", "128", "256", "512", "1024", "2048", "4096"]


# In[23]:


model = H2OGradientBoostingEstimator(seed=1234)
model.train(x=predictors,y=response,training_frame=data_train,validation_frame=data_test)


# In[26]:


aml = H2OAutoML(max_runtime_secs=60, seed=1)
aml.train(x=predictors,y=response, training_frame=data_train)


# In[165]:


# Create h2o dataframes. Make sure to run the "Compute and Compare test metrics" cells to create data_test_X 
# before running these cells##data_test在下方绘制ROC曲线前面有。

htrain = h2o.H2OFrame(pd.concat([data_train_X, data_train['Y']], axis=1))
htest = h2o.H2OFrame(pd.concat([data_test_X, data_test['Y']], axis=1))


# In[166]:


# define cols
x = htrain.columns
y = 'Y'
x.remove(y)


# In[167]:


htrain[y] = htrain[y].asfactor()
htest[y] = htest[y].asfactor()


# In[34]:


# Train Deep Learners for 5 hous##确实需要5个小时
aml_gbm_deep = H2OAutoML(max_runtime_secs = 18000, exclude_algos=['GLM','GBM','DRF','StackedEnsemble'])
aml_gbm_deep.train(x=x, y=y, training_frame=htrain, leaderboard_frame=htest)


# In[35]:


aml_gbm_deep.leaderboard


# In[36]:


# Save best deep learner predictions
h2o_deep_pred = aml_gbm_deep.leader.predict(htest)


# In[37]:


# Save the model
model_path = h2o.save_model(model=aml_gbm_deep.leader, path='E:/R code-LMX/R code for SEER for publishing paper/Data/h2o_deep_learner_may31', force=True)


# In[38]:


model_path


# In[138]:





# # H2O Deep Learner

# In[154]:


model_path


# In[171]:


# Use manual path if model_path is not defined
h2o_deep_learner = h2o.load_model('E:\\R code-LMX\\Yimin-medical disputes\\Models\\h2o_deep_learner_may31\\DeepLearning_grid_3_AutoML_1_20220720_10323_model_3')


# In[172]:


h2o_deep_learner.train(x=x, y=y, training_frame=htrain)


# In[173]:


# Get predictions
h2o_deep_pred = h2o_deep_learner.predict(htest)


# In[174]:


# Convert to pandas df
h2o_deep_pred = h2o_deep_pred['p1'].as_data_frame()


# In[175]:


h2o_deep_pred


# In[176]:


accu_h2o_deep = accuracy_score(data_test['Y'], round(h2o_deep_pred))
accu_h2o_deep


# In[178]:


##下面没有计算出来，有问题
pd.crosstab(data_test['Y'], round(h2o_deep_pred))


# In[181]:


fpr, tpr, _ = roc_curve(data_test['Y'], h2o_deep_pred)
auc_h2o_deep = roc_auc_score(data_test['Y'], h2o_deep_pred)


# In[185]:


round(auc_h2o_deep,3)


# In[186]:


plot_roc_curve(fpr, tpr, round(auc_h2o_deep,3), h2o_deep_learner)


# In[187]:


data_test['lr_pred_proba'] = h2o_deep_pred


# In[188]:


data_test.to_csv('E:/R code-LMX/Yimin-medical disputes/External validation/test_set_with_predictions-h2o_deep_pred-external validation.csv'.format(len(data_train)), index=False)


# In[ ]:





# Class breakdown per model
# 下面的代码中
# [df.g == 0]，df.后面为y变量，此处为Death1y

# In[88]:


def plot_class_breakdown_hist(df, var, var_name, plot_title, xlog=False, ylog=False, **histkwargs):
    df[var][df.Death1y == 0].hist(alpha=.5, label='Negative', color = "green", **histkwargs)
    df[var][df.Death1y == 1].hist(alpha=.5, label='Positive', color = "red", **histkwargs)
    plt.xlabel(var_name)
    plt.title(plot_title)
    if xlog:
        plt.xscale('log')
    if ylog:
        plt.yscale('log')
    plt.ylim(ymax=35, ymin=0)
    plt.legend()
    plt.savefig(var_name + ' Class Breakdown.png');


# In[89]:


plot_class_breakdown_hist(data_test, 'lr_pred_proba', var_name='Logistic Regression Risk', 
                          plot_title='Logistic Regression Class Breakdown', bins=100)


# In[ ]:





# In[ ]:





# In[ ]:




