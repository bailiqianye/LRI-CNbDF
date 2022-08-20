
from sklearn.model_selection import StratifiedKFold, GridSearchCV
import xgboost as xgb
from until.until import  *



dataname= 'human'

Related = pd.read_csv("dataset/"+dataname+"/Related.csv", header=None).to_numpy()
Line_feature1= pd.read_csv("dataset/"+dataname+"/receptor_feature_CTD+pyfet.csv", header=None).to_numpy()
Line_feature2= pd.read_csv("dataset/"+dataname+"/receptor_PseudoAAC_feature.csv", header=None).to_numpy()
Row_feature1= pd.read_csv("dataset/"+dataname+"/ligand_feature_CTD+pyfet.csv", header=None).to_numpy()
Row_feature2 = pd.read_csv("dataset/"+dataname+"/ligand_PseudoAAC_feature.csv", header=None).to_numpy()


Line_feature=np.hstack([Line_feature1,Line_feature2])
Row_feature=np.hstack([Row_feature1,Row_feature2])

def perform(y_proba, y_true,name):  # Performance evaluation
    ACC = accuracy_score(y_true, y_proba.argmax(axis=1))
    fpr, tpr, threshold = roc_curve(y_true, y_proba[:, 1])
    pre, rec, _ = precision_recall_curve(y_true, y_proba[:, 1])
    RECALL = recall_score(y_true, y_proba.argmax(axis=1))
    AUC = auc(fpr, tpr)
    AUPR = auc(rec, pre)
    f1 = f1_score(y_true, y_proba.argmax(axis=1))
    Pre = precision_score(y_true, y_proba.argmax(axis=1))
    #plt
    curve_1 = np.vstack([fpr, tpr])
    curve_1 = pd.DataFrame(curve_1.T)
    curve_1.to_csv('out_new/human/' +str(name)+ '/c' + "cv3" + '_au' + str(AUC) + '.csv', header=None, index=None)

    curve_2 = np.vstack([rec, pre])
    curve_2 = pd.DataFrame(curve_2.T)
    curve_2.to_csv('out_new/human/' +str(name)+ '/c' + "cv3" + '_aupr' + str(AUPR) + '.csv', header=None, index=None)
    # print("ACC=",ACC)
    # print("AUC=", AUC)
    # print("AUPR=", AUPR)
    # print("RECALL=", RECALL)
    # print("f1=", f1)
    # print("Pre=", Pre)
    return ACC, AUC, AUPR, RECALL, f1, Pre

X, y = CV3_Generate_data(Related,Row_feature,Line_feature,feature_dim)

xgb_model=xgb.XGBClassifier()
xgbout=xgb_model.fit(X, y)
feature_importance=xgbout.feature_importances_
feature_number=-feature_importance
H1=np.argsort(feature_number)
mask=H1[:300]
X=X[:,mask]

print("After feature selection:",X.shape)


ACC_array_20 = []
AUC_array_20 = []
RECALL_array_20 = []
AUPR_array_20 = []
f1_array_20 = []
Pre_array_20 = []

ngb_ACC_array_20 = []
ngb_AUC_array_20 = []
ngb_RECALL_array_20 = []
ngb_AUPR_array_20 = []
ngb_f1_array_20 = []
ngb_Pre_array_20 = []


cat_ACC_array_20 = []
cat_AUC_array_20 = []
cat_RECALL_array_20 = []
cat_AUPR_array_20 = []
cat_f1_array_20 = []
cat_Pre_array_20 = []


cf_ACC_array_20 = []
cf_AUC_array_20 = []
cf_RECALL_array_20 = []
cf_AUPR_array_20 = []
cf_f1_array_20 = []
cf_Pre_array_20 = []



for i in range(20):
    ACC_array = []
    AUC_array = []
    RECALL_array = []
    AUPR_array = []
    f1_array = []
    Pre_array = []

    cf_ACC_array = []
    cf_AUC_array = []
    cf_RECALL_array = []
    cf_AUPR_array = []
    cf_f1_array = []
    cf_Pre_array = []

    ngb_ACC_array = []
    ngb_AUC_array = []
    ngb_RECALL_array = []
    ngb_AUPR_array = []
    ngb_f1_array = []
    ngb_Pre_array = []

    cat_ACC_array = []
    cat_AUC_array = []
    cat_RECALL_array = []
    cat_AUPR_array = []
    cat_f1_array = []
    cat_Pre_array = []


    skf = StratifiedKFold(n_splits=5, random_state=np.random.randint(10000), shuffle=True)
    for ci, (train_index, test_index) in enumerate(skf.split(X, y)):
        X_train,X_test = X[train_index],X[test_index]
        y_train,y_test = y[train_index],y[test_index]

        from ngboost import NGBClassifier
        ngb = NGBClassifier()
        ngb.fit(X_train,y_train)
        ngb_y_preds = ngb.predict_proba(X_test)
        ngb_ACC, ngb_AUC, ngb_AUPR, ngb_RECALL, ngb_f1, ngb_Pre = perform(ngb_y_preds, y_test,"ngboost")

        #############################################################################
        from catboost import CatBoostClassifier
        cat = CatBoostClassifier(boosting_type='Ordered',max_depth =10,n_estimators=2000)

        cat.fit(X_train, y_train)
        cat_y_preds = cat.predict_proba(X_test)
        cat_ACC, cat_AUC, cat_AUPR, cat_RECALL, cat_f1, cat_Pre = perform(cat_y_preds, y_test,"catboost")
        #############################################################################

        from deepforest import CascadeForestClassifier
        model = CascadeForestClassifier()
        model.fit(X_train, y_train)
        cf_y_pred = model.predict_proba(X_test)
        cf_ACC, cf_AUC, cf_AUPR, cf_RECALL, cf_f1, cf_Pre = perform(cf_y_pred, y_test,"deepforest")

        #############################################################################
        y_preds = (ngb_y_preds*w1+cat_y_preds*w2+cf_y_pred*w3)
        ACC, AUC, AUPR, RECALL, f1, Pre = perform(y_preds, y_test,"integrated")

        ngb_ACC_array.append(ngb_ACC)
        ngb_AUC_array.append(ngb_AUC)
        ngb_AUPR_array.append(ngb_AUPR)
        ngb_RECALL_array.append(ngb_RECALL)
        ngb_f1_array.append(ngb_f1)
        ngb_Pre_array.append(ngb_Pre)

        cf_ACC_array.append(cf_ACC)
        cf_AUC_array.append(cf_AUC)
        cf_AUPR_array.append(cf_AUPR)
        cf_RECALL_array.append(cf_RECALL)
        cf_f1_array.append(cf_f1)
        cf_Pre_array.append(cf_Pre)

        cat_ACC_array.append(cat_ACC)
        cat_AUC_array.append(cat_AUC)
        cat_AUPR_array.append(cat_AUPR)
        cat_RECALL_array.append(cat_RECALL)
        cat_f1_array.append(cat_f1)
        cat_Pre_array.append(cat_Pre)


        ACC_array.append(ACC)
        AUC_array.append(AUC)
        AUPR_array.append(AUPR)
        RECALL_array.append(RECALL)
        f1_array.append(f1)
        Pre_array.append(Pre)


    print("###############################NGBoost_Five-fold cross validation—Result############################################")
    print("Five-fold cross validation—Result--ACC:", np.mean(ngb_ACC_array))
    print("Five-fold cross validation—Result--AUC:", np.mean(ngb_AUC_array))
    print("Five-fold cross validation—Result--AUPR:", np.mean(ngb_AUPR_array))
    print("Five-fold cross validation—Result--RECALL:", np.mean(ngb_RECALL_array))
    print("Five-fold cross validation—Result--f1：", np.mean(ngb_f1_array))
    print("Five-fold cross validation—Result--Pre", np.mean(ngb_Pre_array))

    print("###############################CatBoost-Five-fold cross validation—Result############################################")
    print("Five-fold cross validation—Result--ACC:", np.mean(cat_ACC_array))
    print("Five-fold cross validation—Result--AUC:", np.mean(cat_AUC_array))
    print("Five-fold cross validation—Result--AUPR:", np.mean(cat_AUPR_array))
    print("Five-fold cross validation—Result--RECALL:", np.mean(cat_RECALL_array))
    print("Five-fold cross validation—Result--f1：", np.mean(cat_f1_array))
    print("Five-fold cross validation—Result--Pre", np.mean(cat_Pre_array))

    print("###############################deepforest-Five-fold cross validation—Result############################################")
    print("Five-fold cross validation—Result--ACC:", np.mean(cf_ACC_array))
    print("Five-fold cross validation—Result--AUC:", np.mean(cf_AUC_array))
    print("Five-fold cross validation—Result--AUPR:", np.mean(cf_AUPR_array))
    print("Five-fold cross validation—Result--RECALL:", np.mean(cf_RECALL_array))
    print("Five-fold cross validation—Result--f1：", np.mean(cf_f1_array))
    print("Five-fold cross validation—Result--Pre", np.mean(cf_Pre_array))

    print("###############################Five-fold cross validation—Result############################################")
    print("Five-fold cross validation—Result--ACC:", np.mean(ACC_array))
    print("Five-fold cross validation—Result--AUC:", np.mean(AUC_array))
    print("Five-fold cross validation—Result--AUPR:", np.mean(AUPR_array))
    print("Five-fold cross validation—Result--RECALL:", np.mean(RECALL_array))
    print("Five-fold cross validation—Result--f1：", np.mean(f1_array))
    print("Five-fold cross validation—Result--Pre", np.mean(Pre_array))

    ngb_ACC_array_20.append(np.mean(ngb_ACC_array))
    ngb_AUC_array_20.append(np.mean(ngb_AUC_array))
    ngb_AUPR_array_20.append(np.mean(ngb_AUPR_array))
    ngb_RECALL_array_20.append(np.mean(ngb_RECALL_array))
    ngb_f1_array_20.append(np.mean(ngb_f1_array))
    ngb_Pre_array_20.append(np.mean(ngb_Pre_array))

    cat_ACC_array_20.append(np.mean(cat_ACC_array))
    cat_AUC_array_20.append(np.mean(cat_AUC_array))
    cat_AUPR_array_20.append(np.mean(cat_AUPR_array))
    cat_RECALL_array_20.append(np.mean(cat_RECALL_array))
    cat_f1_array_20.append(np.mean(cat_f1_array))
    cat_Pre_array_20.append(np.mean(cat_Pre_array))

    cf_ACC_array_20.append(np.mean(cf_ACC_array))
    cf_AUC_array_20.append(np.mean(cf_AUC_array))
    cf_AUPR_array_20.append(np.mean(cf_AUPR_array))
    cf_RECALL_array_20.append(np.mean(cf_RECALL_array))
    cf_f1_array_20.append(np.mean(cf_f1_array))
    cf_Pre_array_20.append(np.mean(cf_Pre_array))

    ACC_array_20.append(np.mean(ACC_array))
    AUC_array_20.append(np.mean(AUC_array))
    AUPR_array_20.append(np.mean(AUPR_array))
    RECALL_array_20.append(np.mean(RECALL_array))
    f1_array_20.append(np.mean(f1_array))
    Pre_array_20.append(np.mean(Pre_array))










ngb_ACC_sum_20 = np.mean(ngb_ACC_array_20)
ngb_AUC_sum_20 = np.mean(ngb_AUC_array_20)
ngb_AUPR_sum_20 = np.mean(ngb_AUPR_array_20)
ngb_RECALL_sum_20 = np.mean(ngb_RECALL_array_20)
ngb_f1_sum_20 = np.mean(ngb_f1_array_20)
ngb_Pre_sum_20 = np.mean(ngb_Pre_array_20)

cat_ACC_sum_20 = np.mean(cat_ACC_array_20)
cat_AUC_sum_20 = np.mean(cat_AUC_array_20)
cat_AUPR_sum_20 = np.mean(cat_AUPR_array_20)
cat_RECALL_sum_20 = np.mean(cat_RECALL_array_20)
cat_f1_sum_20 = np.mean(cat_f1_array_20)
cat_Pre_sum_20 = np.mean(cat_Pre_array_20)



cf_ACC_sum_20 = np.mean(cf_ACC_array_20)
cf_AUC_sum_20 = np.mean(cf_AUC_array_20)
cf_AUPR_sum_20 = np.mean(cf_AUPR_array_20)
cf_RECALL_sum_20 = np.mean(cf_RECALL_array_20)
cf_f1_sum_20 = np.mean(cf_f1_array_20)
cf_Pre_sum_20 = np.mean(cf_Pre_array_20)

ACC_sum_20 = np.mean(ACC_array_20)
AUC_sum_20 = np.mean(AUC_array_20)
AUPR_sum_20 = np.mean(AUPR_array_20)
RECALL_sum_20 = np.mean(RECALL_array_20)
f1_sum_20 = np.mean(f1_array_20)
Pre_sum_20 = np.mean(Pre_array_20)



print("###############################NGBoost-20-Five-fold cross validation—Result############################################")
print("20-Five-fold cross validation—Result--ACC:", '%.4f' % ngb_ACC_sum_20, "±", '%.4f' % np.std(ngb_ACC_array_20))
print("20-Five-fold cross validation—Result--AUC:", '%.4f' % ngb_AUC_sum_20, "±", '%.4f' % np.std(ngb_AUC_array_20))
print("20-Five-fold cross validation—Result--AUPR:", '%.4f' % ngb_AUPR_sum_20, "±", '%.4f' % np.std(ngb_AUPR_array_20))
print("20-Five-fold cross validation—Result--RECALL:", '%.4f' % ngb_RECALL_sum_20, "±", '%.4f' % np.std(ngb_RECALL_array_20))
print("20-Five-fold cross validation—Result--f1：", '%.4f' % ngb_f1_sum_20, "±", '%.4f' % np.std(ngb_f1_array_20))
print("20-Five-fold cross validation—Result--Pre", '%.4f' % ngb_Pre_sum_20, "±", '%.4f' % np.std(ngb_Pre_array_20))





print("###############################CatBoost-20-Five-fold cross validation—Result############################################")
print("20-Five-fold cross validation—Result--ACC:", '%.4f' % cat_ACC_sum_20, "±", '%.4f' % np.std(cat_ACC_array_20))
print("20-Five-fold cross validation—Result--AUC:", '%.4f' % cat_AUC_sum_20, "±", '%.4f' % np.std(cat_AUC_array_20))
print("20-Five-fold cross validation—Result--AUPR:", '%.4f' % cat_AUPR_sum_20, "±", '%.4f' % np.std(cat_AUPR_array_20))
print("20-Five-fold cross validation—Result--RECALL:", '%.4f' % cat_RECALL_sum_20, "±", '%.4f' % np.std(cat_RECALL_array_20))
print("20-Five-fold cross validation—Result--f1：", '%.4f' % cat_f1_sum_20, "±", '%.4f' % np.std(cat_f1_array_20))
print("20-Five-fold cross validation—Result--Pre", '%.4f' % cat_Pre_sum_20, "±", '%.4f' % np.std(cat_Pre_array_20))




print("###############################deepforest-f20-Five-fold cross validation—Result############################################")
print("20-Five-fold cross validation—Result--ACC:", '%.4f' % cf_ACC_sum_20, "±", '%.4f' % np.std(cf_ACC_array_20))
print("20-Five-fold cross validation—Result--AUC:", '%.4f' % cf_AUC_sum_20, "±", '%.4f' % np.std(cf_AUC_array_20))
print("20-Five-fold cross validation—Result--AUPR:", '%.4f' % cf_AUPR_sum_20, "±", '%.4f' % np.std(cf_AUPR_array_20))
print("20-Five-fold cross validation—Result--RECALL:", '%.4f' % cf_RECALL_sum_20, "±", '%.4f' % np.std(cf_RECALL_array_20))
print("20-Five-fold cross validation—Result--f1：", '%.4f' % cf_f1_sum_20, "±", '%.4f' % np.std(cf_f1_array_20))
print("20-Five-fold cross validation—Result--Pre", '%.4f' % cf_Pre_sum_20, "±", '%.4f' % np.std(cf_Pre_array_20))


print("###############################20-Five-fold cross validation—Result############################################")
print("20-Five-fold cross validation—Result--ACC:", '%.4f' % ACC_sum_20, "±", '%.4f' % np.std(ACC_array_20))
print("20-Five-fold cross validation—Result--AUC:", '%.4f' % AUC_sum_20, "±", '%.4f' % np.std(AUC_array_20))
print("20-Five-fold cross validation—Result--AUPR:", '%.4f' % AUPR_sum_20, "±", '%.4f' % np.std(AUPR_array_20))
print("20-Five-fold cross validation—Result--RECALL:", '%.4f' % RECALL_sum_20, "±", '%.4f' % np.std(RECALL_array_20))
print("20-Five-fold cross validation—Result--f1：", '%.4f' % f1_sum_20, "±", '%.4f' % np.std(f1_array_20))
print("20-Five-fold cross validation—Result--Pre", '%.4f' % Pre_sum_20, "±", '%.4f' % np.std(Pre_array_20))














