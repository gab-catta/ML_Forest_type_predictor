from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score,f1_score, recall_score

from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC

# Load dataset 
dataset_file = '/home/gabri/Desktop/Machine_learning_py/Progetto/Progetto2/covtype.csv' 
df = pd.read_csv(dataset_file,header=None)
df.columns = [f'feature_{i}' for i in range(df.shape[1])]
df_reduced, _ = train_test_split(df, train_size=50000, random_state=42) #i need to reduce the number of samples, my pc is melting 
df = pd.DataFrame(df)

# Separate features and labels
x_full = df.iloc[:, :-1]
y_full = df.iloc[:, -1]
print(x_full.shape)
print(y_full.shape)
x = df_reduced.iloc[:, :-1]
y = df_reduced.iloc[:, -1]
print(x.shape)
print(y.shape) 
'''
sns.set_theme(style="whitegrid")
for i in range(x.shape[1]):
    data = x.iloc[:, i].to_numpy(dtype=float)  # ensure it's a NumPy array
    plt.figure(figsize=(8, 4))
    plt.hist(data, bins=30, edgecolor='black', alpha=0.7)
    plt.title(f'Distribution of Feature {i}')
    plt.xlabel(f'Feature {i}')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.show()
'''
######EDA#######
#Nan values not present (dataset confirmed it) 
#Split dataset in train and test 
X_tr, X_ts, y_tr, y_ts= train_test_split(x, y, test_size=0.35, random_state=2)
y_tr = y_tr.to_numpy()
y_ts = y_ts.to_numpy()
print(X_tr.shape)
print(y_tr.shape) 
#Scaling feature matrix
scaler = StandardScaler()
X_tr_transf = scaler.fit_transform(X_tr)
X_ts_transf = scaler.transform(X_ts)
###########TRAINING#################
#########MODELS############
softmax_reg_model = LogisticRegression(multi_class='multinomial',n_jobs=-1,max_iter=10000) #haven't transformed y into OneHotEncoded vector because sklearn manages it automatically. Argue about the doc saying that dual solution is preferred when n_samples>n_features, should it be the opposite? 
svm_model = SVC(decision_function_shape='ovr',max_iter=2000,cache_size=1000) 
random_forest_model = RandomForestClassifier(n_jobs=-1)
mlp_model = MLPClassifier(max_iter=2000,early_stopping=True)
#add NN.

#####Parameter tuning (since i'm using >= 500K samples i need to limit the GridSearchCV, otherwise it will take too long, i'll do a double search with RandomizedSearchCV to get indicative paramteres to insert in the limited GridSearchCV)
print('########################')
print('####Cross-validation####')
print('########################')
'''
hparameters = {'max_features': ['sqrt',0.1,0.3,0.01,0.4],'n_estimators': [200,500,700]}
clf = RandomizedSearchCV(estimator=random_forest_model, param_distributions=hparameters, scoring='accuracy', cv=3,verbose=True) 
clf.fit(X_tr_transf, y_tr)
print('Random Forest .................')
print('The best max_features is ',clf.best_params_.get('max_features'))
print('The best n_estimators is ',clf.best_params_.get('n_estimators'))
print('Which corresponds to an Accuracy of ', clf.best_score_)
'''
'''
hparameters = {'max_features': [0.4,0.41,0.39],'n_estimators': [600,500,400]}
clf = GridSearchCV(estimator=random_forest_model, param_grid=hparameters, scoring='accuracy', cv=3,verbose=True) 
clf.fit(X_tr_transf, y_tr)
print('Random Forest .................')
print('The best max_features is ',clf.best_params_.get('max_features'))
print('The best n_estimators is ',clf.best_params_.get('n_estimators'))
print('Which corresponds to an Accuracy of ', clf.best_score_)
'''
'''
After RandomizedSearchCV (FULL DATA_SET): 
    The best max_features is  0.4
    The best n_estimators is  200
    Which corresponds to an Accuracy of  0.9554039773673142
After GriSearchCV (FULL DATA_SET): 
    The best max_features is  0.41
    The best n_estimators is  180
    Which corresponds to an Accuracy of  0.9560129951687388
'''
'''
After RandomizedSearchCV (50k  REDUCED_DATA_SET): 
    The best max_features is  0.4
    The best n_estimators is  500
    Which corresponds to an Accuracy of  0.858061604462938
After GridSearchCV (50k  REDUCED_DATA_SET): 
    The best max_features is 0.39
    The best n_estimators is  600 
    Which corresponds to an Accuracy of  0.858954564654765
'''

#for softmax_regression the GridSearchCV is kinda fast, so we can skip the preliminary RandomizedSearchCV! 
'''
hparameters = {'tol': [1e-4, 1e-3, 5e-3, 1e-1,1,1e1,1e2],'penalty': ('l1', 'l2','None'), 'C': [1e-4, 1e-3, 5e-3, 1e-1,1,1e1,1e2]}
clf = GridSearchCV(estimator=softmax_reg_model, param_grid=hparameters, scoring='accuracy', cv=3,n_jobs=-1) 
clf.fit(X_tr_transf, y_tr)
print('Softmax Regression ..................')
print('Overall, the best choice for parameter penalty is ', clf.best_params_.get('penalty'))
print('Overall, the best choice for parameter tol is ', clf.best_params_.get('tol'))
if clf.best_params_.get('penalty') != None:
    print('with a best C value equal to ', clf.best_params_.get('C'))
print('Which corresponds to an Accuracy of ', clf.best_score_)
'''
'''
After GriSearchCV (FULL DATA_SET): 
    Overall, the best choice for parameter tol is  0.0001
    Overall, the best choice for parameter penalty is  l2
    with a best C value equal to  100.0
    Which corresponds to an Accuracy of  0.724559590289871
'''
'''
After GridSearchCV (50k  REDUCED_DATA_SET): 
    Overall, the best choice for parameter penalty is  l2
    Overall, the best choice for parameter tol is  0.0001
    with a best C value equal to  10.0
    Which corresponds to an Accuracy of  0.7301538333355969 #HIGHER THAN THE FULL_DATASET ONE?
'''

'''
hparameters = {'tol': [1e-4, 1e-3, 5e-3, 1e-1,1,1e1,1e2],'C': [1e-4,1e-3,1e-2,1,10],'kernel': ['linear', 'poly', 'rbf']}
clf = RandomizedSearchCV(estimator=svm_model, param_distributions=hparameters, scoring='accuracy', cv=3,verbose=True,n_jobs=-1) 
clf.fit(X_tr_transf, y_tr)
print('SVC ..................')
print('Overall, the best choice for tol is ', clf.best_params_.get('tol'))
print('Overall, the best choice for C is ', clf.best_params_.get('C'))
print('Overall, the best choice for kernel is ', clf.best_params_.get('kernel'))
print('Which corresponds to an Accuracy of ', clf.best_score_)
'''
'''
hparameters = {'tol': [1, 1.5, 0.5], 'C': [0.001, 0.003, 0.0007], 'kernel': ['rbf']}
clf = GridSearchCV(estimator=svm_model, param_grid=hparameters, scoring='accuracy', cv=3,verbose=True,n_jobs=-1) 
clf.fit(X_tr_transf, y_tr)
print('SVC ..................')
print('Overall, the best choice for C is ', clf.best_params_.get('C'))
print('Overall, the best choice for tol is ', clf.best_params_.get('tol'))
print('Which corresponds to an Accuracy of ', clf.best_score_)
'''
'''
After RandomizedSearchCV (FULL DATA_SET): ####My hardware can't support these workloads, SVC's parameter tuning revealed to be particurarly RAM-hungry (94% of RAM usage with 16GB capcity.), this is probably due to the necessity to store all the support vectors since we are using kernels.
    Overall, the best choice for tol is  1
    Overall, the best choice for C is  0.001
    Overall, the best choice for kernel is  rbf
    Which corresponds to an Accuracy of  0.5067895308945213
After GriSearchCV (FULL DATA_SET): 
    Overall, the best choice for C is  0.001
    Overall, the best choice for tol is  1
    Overall, the best choice for kernel is  rbf
    Which corresponds to an Accuracy of  0.5069363285812726
'''
'''
After RandomizedSearchCV (50k  REDUCED_DATA_SET): 
    Overall, the best choice for C is  1
    Overall, the best choice for kernel is  rbf
    Which corresponds to an Accuracy of  0.7473540047238386
After GridSearchCV (50k  REDUCED_DATA_SET): 
    Overall, the best choice for C is  1.5
    Overall, the best choice for kernel is  rbf
    Which corresponds to an Accuracy of  0.753907822488601
'''
'''
hparameters = {'hidden_layer_sizes': [100,1000,500,10,300],'activation': ['tanh','identity','logistic','relu'],'solver': ['lbfgs', 'adam'],'alpha': [1e-4,1e-3,1e-2,1e-1,1,10]}
clf = RandomizedSearchCV(estimator=mlp_model, param_distributions=hparameters, scoring='accuracy', cv=3,verbose=True,n_jobs=-1) 
clf.fit(X_tr_transf, y_tr)
print('MLP ..................')
print('Overall, the best choice for hidden_layer_sizes is ', clf.best_params_.get('hidden_layer_sizes'))
print('Overall, the best choice for activation is ', clf.best_params_.get('activation'))
print('Overall, the best choice for solver is ', clf.best_params_.get('solver'))
print('Overall, the best choice for alpha is ', clf.best_params_.get('alpha'))
print('Which corresponds to an Accuracy of ', clf.best_score_)
'''
'''
hparameters = {'hidden_layer_sizes': [500,550,450],'activation': ['relu'],'solver': ['lbfgs'],'alpha': [10,20,50,100]}
clf = GridSearchCV(estimator=mlp_model, param_grid=hparameters, scoring='accuracy', cv=3,verbose=True,n_jobs=-1) 
clf.fit(X_tr_transf, y_tr)
print('MLP ..................')
print('Overall, the best choice for hidden_layer_sizes is ', clf.best_params_.get('hidden_layer_sizes'))
print('Overall, the best choice for activation is ', clf.best_params_.get('activation'))
print('Overall, the best choice for solver is ', clf.best_params_.get('solver'))
print('Overall, the best choice for alpha is ', clf.best_params_.get('alpha'))
print('Which corresponds to an Accuracy of ', clf.best_score_)
'''
'''
After RandomizedSearchCV (50k  REDUCED_DATA_SET): 
    Overall, the best choice for hidden_layer_sizes is  500
    Overall, the best choice for activation is  relu
    Overall, the best choice for solver is  lbfgs
    Overall, the best choice for alpha is  10
    Which corresponds to an Accuracy of  0.8424924427745276
After GridSearchCV (50k  REDUCED_DATA_SET): 
    Overall, the best choice for hidden_layer_sizes is  500
    Overall, the best choice for activation is  relu
    Overall, the best choice for solver is  lbfgs
    Overall, the best choice for alpha is  10
    Which corresponds to an Accuracy of  0.8424924427745276
'''

print('After GriSearchCV:')
print('RandomForestClassifier won!')
print('    The best max_features is  0.41')
print('    The best n_estimators is  180')
print('    Which corresponds to an Accuracy of  0.9560129951687388')
button = input('#-----------------Premi un tasto per proseguire-----------------#')
best_model = RandomForestClassifier(n_jobs=-1,max_features=0.41,n_estimators=180)
##########TESTING (RandomForestClassifier won!)#############
best_model.fit(X_tr_transf,y_tr)
y_pred = best_model.predict(X_ts_transf)
print('###############')
print('####TESTING####')
print('###############')
print('The final testing Accuracy is ', accuracy_score(y_ts, y_pred))
print('The final testing Precision is ', precision_score(y_ts, y_pred,average='weighted'))
print('The final testing Recall is ', recall_score(y_ts, y_pred,average='weighted'))
print('The final testing F1_score is ', f1_score(y_ts, y_pred,average='weighted'))

# Plotting the feature importance
importances = best_model.feature_importances_
indices = pd.Series(importances, index=x.columns).sort_values(ascending=False)
plt.figure(figsize=(12, 6))
sns.barplot(x=indices.values, y=indices.index, hue=indices.values,legend=False ,palette='viridis')
plt.title('Feature Importance')
plt.xlabel('Importance')
plt.ylabel('Features')
plt.tight_layout()
plt.show()


