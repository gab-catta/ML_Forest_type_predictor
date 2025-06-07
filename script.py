from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score,f1_score, recall_score

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC




def Dataset_resizing(df):
    while True:
        while True:
            n_samples = input("How many samples to you want to work with? (for full dataset type 'max'):  ")
            if n_samples == "max":
                x = df.iloc[:, :-1]
                y = df.iloc[:, -1]
                print(x.shape)
                print(y.shape)
                return x,y 
            else: 
                try:
                    n_samples = int(n_samples)
                    break 
                except ValueError:
                    print("#-------------------------------------------------------------------#")
                    print("Invalid input. Please enter a integer.")  
        if n_samples == "max":
            break
        elif n_samples != "max" and 10000<n_samples<581012:
            df_reduced, _ = train_test_split(df, train_size=n_samples, random_state=42)
            df = pd.DataFrame(df)
            x = df_reduced.iloc[:, :-1]
            y = df_reduced.iloc[:, -1]
            print(x.shape)
            print(y.shape) 
            return x,y
        else:
            print("#-------------------------------------------------------------------#")
            print("Please choose a number of samples between 10000 and 581012")

def random_forest_model_grid(X_tr_transf,y_tr):
    random_forest_model = RandomForestClassifier(n_jobs=-1)
    hparameters = {'criterion': ['gini', 'entropy'],'max_features': ['sqrt',0.1,0.3,0.6,1],'n_estimators': [100,200,500,700,1000]}
    clf = GridSearchCV(estimator=random_forest_model, param_grid=hparameters, scoring='accuracy', cv=3,verbose=True) 
    clf.fit(X_tr_transf, y_tr)
    print('Random Forest .................')
    print('The best criterion is ',clf.best_params_.get('criterion'))
    print('The best max_features is ',clf.best_params_.get('max_features'))
    print('The best n_estimators is ',clf.best_params_.get('n_estimators'))
    print('Which corresponds to an Accuracy of ', clf.best_score_)
    return clf.best_params_.get('criterion'), clf.best_params_.get('max_features'), clf.best_params_.get('n_estimators'), clf.best_score_

def softmaxregress_grid(X_tr_transf,y_tr):
    softmax_reg_model = LogisticRegression(multi_class='multinomial',n_jobs=-1) #haven't transformed y into OneHotEncoded vector because sklearn manages it automatically. Argue about the doc saying that dual solution is preferred when n_samples>n_features, should it be the opposite? 
    hparameters = {'solver': ['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga'], 'tol': [1e-4, 1e-3, 5e-3, 1e-1,1,1e1,1e2],'penalty': ('l1', 'l2',None), 'C': [1e-4, 1e-3, 5e-3, 1e-1,1,1e1,1e2]}
    clf = GridSearchCV(estimator=softmax_reg_model, param_grid=hparameters, scoring='accuracy', cv=3,n_jobs=-1) 
    clf.fit(X_tr_transf, y_tr)
    print('Softmax Regression ..................')
    print('Overall, the best choice for parameter solver is ', clf.best_params_.get('solver'))
    print('Overall, the best choice for parameter penalty is ', clf.best_params_.get('penalty'))
    print('Overall, the best choice for parameter tol is ', clf.best_params_.get('tol'))
    if clf.best_params_.get('penalty') != None:
        print('with a best C value equal to ', clf.best_params_.get('C'))
    print('Which corresponds to an Accuracy of ', clf.best_score_)
    return clf.best_params_.get('solver'), clf.best_params_.get('penalty'), clf.best_params_.get('tol'), clf.best_params_.get('C'), clf.best_score_

def svc_grid(X_tr_transf,y_tr):
    svm_model = SVC(decision_function_shape='ovr',cache_size=1000) #one vs the rest by default because it's more efficient to process
    hparameters = {'C': [1e-4,1e-3,1e-2,1,10],'kernel': ['linear', 'poly', 'rbf','sigmoid']}
    clf = GridSearchCV(estimator=svm_model, param_grid=hparameters, scoring='accuracy', cv=3,verbose=True,n_jobs=-1) 
    clf.fit(X_tr_transf, y_tr)
    print('SVC ..................')
    print('Overall, the best choice for C is ', clf.best_params_.get('C'))
    print('Overall, the best choice for kernel is ', clf.best_params_.get('kernel'))
    print('Which corresponds to an Accuracy of ', clf.best_score_)
    return clf.best_params_.get('C'), clf.best_params_.get('kernel'), clf.best_score_

def mlp_grid(X_tr_transf,y_tr):
    mlp_model = MLPClassifier(early_stopping=True)
    hparameters = {'hidden_layer_sizes': [5,10,100,300,500,700,1000],'activation': ['tanh','identity','logistic','relu'],'solver': ['lbfgs', 'adam'],'alpha': [1e-4,1e-3,1e-2,1e-1,1,10]}
    clf = GridSearchCV(estimator=mlp_model, param_grid=hparameters, scoring='accuracy', cv=3,verbose=True,n_jobs=-1) 
    clf.fit(X_tr_transf, y_tr)
    print('MLP ..................')
    print('Overall, the best choice for hidden_layer_sizes is ', clf.best_params_.get('hidden_layer_sizes'))
    print('Overall, the best choice for activation is ', clf.best_params_.get('activation'))
    print('Overall, the best choice for solver is ', clf.best_params_.get('solver'))
    print('Overall, the best choice for alpha is ', clf.best_params_.get('alpha'))
    print('Which corresponds to an Accuracy of ', clf.best_score_)
    return clf.best_params_.get('hidden_layer_sizes'), clf.best_params_.get('activation'), clf.best_params_.get('solver'), clf.best_params_.get('alpha'), clf.best_score_



if __name__ == "__main__":

    # Load dataset 
    dataset_file = '/home/gabri/Desktop/Machine_learning_py/Progetto/Progetto2/covtype.csv' 
    df = pd.read_csv(dataset_file,header=None)
    df.columns = [f'feature_{i}' for i in range(df.shape[1])]

    ######EDA#######
    # User decides how many samples he wants to work with
    x,y = Dataset_resizing(df)

    # Plotting distribution of features
    '''
    sns.set_theme(style="whitegrid")
    for i in range(df):
        data = x.iloc[:, i].to_numpy(dtype=float)  # ensure it's a NumPy array
        plt.figure(figsize=(8, 4))
        plt.hist(data, bins=30, edgecolor='black', alpha=0.7)
        plt.title(f'Distribution of Feature {i}')
        plt.xlabel(f'Feature {i}')
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.show()
    '''

    # Nan values not present (dataset confirmed it) 

    # Split dataset in train and test 
    X_tr, X_ts, y_tr, y_ts= train_test_split(x, y, test_size=0.35, random_state=2)
    y_tr = y_tr.to_numpy()
    y_ts = y_ts.to_numpy()

    # Scaling feature matrix
    scaler = StandardScaler()
    X_tr_transf = scaler.fit_transform(X_tr)
    X_ts_transf = scaler.transform(X_ts)

    ###########TRAINING#################
    print('########################')
    print('####Cross-validation####')
    print('########################')
    best_scores_list = [] # list containing all the best accuracy stores of the models

    # Random Forest Classifier
    criterion_randforest,max_feature_randforest, n_estimators_randforest, score_randforest = random_forest_model_grid(X_tr_transf,y_tr)
    best_scores_list.append(score_randforest)
    
    # Softmax Regression 
    solver_softmax, penalty_softmax, tol_softmax, C_softmax, score_softmax = softmaxregress_grid(X_tr_transf,y_tr)
    best_scores_list.append(score_softmax)
    
    # Support vector machine Classifier
    C_svc, kernel_svc, score_svc = svc_grid(X_tr_transf,y_tr)
    best_scores_list.append(score_svc)

    # Multiple layer processor Classifier
    hidden_layer_sizes_mlp, activation_mlp, solver_mlp, alpha_mlp, score_mlp = mlp_grid(X_tr_transf,y_tr)
    best_scores_list.append(score_mlp)

    # Choice of the model with the highest accuracy, fitting and testing for the final evaluation
    more_accurate = max(best_scores_list)
    if more_accurate == score_softmax:
        print 
        best_model = LogisticRegression(multi_class='multinomial',n_jobs=-1,solver=solver_softmax,penalty=penalty_softmax,tol=tol_softmax,C=C_softmax) 
        best_model.fit(X_tr_transf,y_tr)
        y_pred = best_model.predict(X_ts_transf)
        print('##################################')
        print('####TESTING SOFTMAX REGRESSION####')
        print('##################################')
        print('The final testing Accuracy is ', accuracy_score(y_ts, y_pred))
        print('The final testing Precision is ', precision_score(y_ts, y_pred,average='weighted'))
        print('The final testing Recall is ', recall_score(y_ts, y_pred,average='weighted'))
        print('The final testing F1_score is ', f1_score(y_ts, y_pred,average='weighted'))
    elif more_accurate == score_randforest:
        best_model = RandomForestClassifier(n_jobs=-1,criterion=criterion_randforest,max_features=max_feature_randforest,n_estimators=n_estimators_randforest)
        best_model.fit(X_tr_transf,y_tr)
        y_pred = best_model.predict(X_ts_transf)
        print('#############################')
        print('####TESTING RANDOM FOREST####')
        print('#############################')
        print('The final testing Accuracy is ', accuracy_score(y_ts, y_pred))
        print('The final testing Precision is ', precision_score(y_ts, y_pred,average='weighted'))
        print('The final testing Recall is ', recall_score(y_ts, y_pred,average='weighted'))
        print('The final testing F1_score is ', f1_score(y_ts, y_pred,average='weighted'))            
    elif more_accurate == score_svc:
        best_model= SVC(decision_function_shape='ovr',cache_size=1000,C=C_svc,kernel=kernel_svc)
        best_model.fit(X_tr_transf,y_tr)
        y_pred = best_model.predict(X_ts_transf)
        print('###################')
        print('####TESTING SVC####')
        print('###################')
        print('The final testing Accuracy is ', accuracy_score(y_ts, y_pred))
        print('The final testing Precision is ', precision_score(y_ts, y_pred,average='weighted'))
        print('The final testing Recall is ', recall_score(y_ts, y_pred,average='weighted'))
        print('The final testing F1_score is ', f1_score(y_ts, y_pred,average='weighted'))      
    elif more_accurate == score_mlp:
        best_model = MLPClassifier(early_stopping=True,hidden_layer_sizes=hidden_layer_sizes_mlp,activation=activation_mlp,solver=solver_mlp,alpha=alpha_mlp)
        best_model.fit(X_tr_transf,y_tr)
        y_pred = best_model.predict(X_ts_transf)
        print('###################')
        print('####TESTING MLP####')
        print('###################')
        print('The final testing Accuracy is ', accuracy_score(y_ts, y_pred))
        print('The final testing Precision is ', precision_score(y_ts, y_pred,average='weighted'))
        print('The final testing Recall is ', recall_score(y_ts, y_pred,average='weighted'))
        print('The final testing F1_score is ', f1_score(y_ts, y_pred,average='weighted')) 




