# ------------------------------------------------
# Macine Learnig     : neural network
# Target Competition : horse racing
# Score              : 
# 
# ------------------------------------------------

# ------------------------------------------------
# import
# ------------------------------------------------
import pandas as pd
import numpy as np
import glob
from tqdm import tqdm

# neural_network, random forest
from sklearn import neural_network
from sklearn.ensemble import RandomForestClassifier

# lightGBM
import lightgbm as lgb

# ------------------------------------------------
# read
# ------------------------------------------------
def read_files():

    fin = "re_data/*/*/*"

    # read data
    fff_info   = glob.glob("re_data/2008_data/01/*_info")
    fff_result = glob.glob("re_data/2008_data/01/*_result")

    train_info = []
    train_result = []
    for l in range(len(fff_info)):
        temp_info   = np.loadtxt(fff_info[l],   delimiter=",", skiprows=0, usecols=None, unpack=True, ndmin=0)
        temp_result = np.loadtxt(fff_result[l], delimiter=",", skiprows=0, usecols=None, unpack=True, ndmin=0)
        if temp_info.shape[1] == 13:
            temp_info   = temp_info.tolist()
            train_info.append(temp_info)
            temp_result = temp_result.tolist()
            train_result.append(temp_result)
        #end
    #end
    train_info = np.array(train_info)
    train_result = np.array(train_result)



    # read data
    fff_info   = glob.glob("re_data/2009_data/01/*_info")
    fff_result = glob.glob("re_data/2009_data/01/*_result")

    test_info = []
    test_result = []
    for l in range(len(fff_info)):
        temp_info   = np.loadtxt(fff_info[l],   delimiter=",", skiprows=0, usecols=None, unpack=True, ndmin=0)
        temp_result = np.loadtxt(fff_result[l], delimiter=",", skiprows=0, usecols=None, unpack=True, ndmin=0)
        if temp_info.shape[1] == 13:
            temp_info   = temp_info.tolist()
            test_info.append(temp_info)
            temp_result = temp_result.tolist()
            test_result.append(temp_result)
        #end
    #end
    test_info = np.array(test_info)
    test_result = np.array(test_result)

    return train_info, train_result, test_info, test_result
#end

# ------------------------------------------------
# read
# ------------------------------------------------
def read_files_temp():

    fin  = "re_data/*/*/*"
    maxd = 18 

    # read data
    fff_info   = glob.glob("re_data/200*_data/01/*_info")
    temp = glob.glob("re_data/201*_data/01/*_info")
    fff_info   = fff_info + temp
    temp = glob.glob("re_data/2020_data/01/*_info")
    fff_info   = fff_info + temp
    fff_result = glob.glob("re_data/200*_data/01/*_result")
    temp = glob.glob("re_data/201*_data/01/*_result")
    fff_result   = fff_result + temp
    temp = glob.glob("re_data/2020_data/01/*_result")
    fff_result   = fff_result + temp

    train_info = []
    train_result = []
    for l in tqdm(range(len(fff_info))):
        temp_info   = np.loadtxt(fff_info[l],   delimiter=",", skiprows=0, usecols=None, unpack=True, ndmin=0)
        temp_result = np.loadtxt(fff_result[l], delimiter=",", skiprows=0, usecols=None, unpack=True, ndmin=0)
        
        temp_info   = temp_info.tolist()
        train_info.append(temp_info)
        temp_result = temp_result.tolist()
        train_result.append(temp_result)
    #end
    train_info = np.array(train_info)
    train_result = np.array(train_result)



    # read data
    fff_info   = glob.glob("re_data/2021_data/01/*_info")
    fff_result = glob.glob("re_data/2021_data/01/*_result")

    test_info = []
    test_result = []
    for l in range(len(fff_info)):
        temp_info   = np.loadtxt(fff_info[l],   delimiter=",", skiprows=0, usecols=None, unpack=True, ndmin=0)
        temp_result = np.loadtxt(fff_result[l], delimiter=",", skiprows=0, usecols=None, unpack=True, ndmin=0)

        temp_info   = temp_info.tolist()
        test_info.append(temp_info)
        temp_result = temp_result.tolist()
        test_result.append(temp_result)
    #end
    test_info = np.array(test_info)
    test_result = np.array(test_result)

    return train_info, train_result, test_info, test_result
#end

# ------------------------------------------------
# main
# ------------------------------------------------
def main():   
    print(" ------------------------------ ")
    print(" start read data")
    
    train_info, train_result, test_info, test_result = read_files_temp() 

    print(" end read data")
    print(" ------------------------------ ")

    # confirm size of array
    train_shape = train_info.shape
    test_shape = test_info.shape
    print(" ------------------------------ ")
    print(" size of train data")
    print(train_shape)
    print(" size of train data result")
    print(train_result.shape)
    print(" size of test data")
    print(test_shape)
    print(" size of test data result")
    print(test_result.shape)
    print(" ------------------------------ ")
    
    # reshape
    train_nd = train_shape[1] * train_shape[2]
    test_nd  = test_shape[1]  * test_shape[2]
    train_info = np.reshape(train_info, [train_shape[0], train_nd])
    test_info  = np.reshape(test_info,  [test_shape[0],  test_nd])

    # get the objective and explanatory variables for "train"
    objective   = train_result
    explanatory = train_info

    # get the explanatory variables for "test"
    explanatory_test = test_info

    # データセットを登録
    lgb_train = lgb.Dataset(train_info, train_result)
    lgb_test  = lgb.Dataset(test_info, test_result, reference=lgb_train)
    
    # LightGBMのハイパーパラメータを設定
    params = {'task': 'train',            # タスクを訓練に設定
            'boosting_type': 'gbdt',      # GBDTを指定
            'objective': 'multiclass',    # 多クラス分類を指定
            'metric': 'multi_logloss',    # 多クラス分類の損失（誤差）
            'num_class': 18,              # クラスの数（irisデータセットが3個のクラスなので）
            'learning_rate': 0.1,         # 学習率
            'num_leaves': 21,             # ノードの数
            'min_data_in_leaf': 10,        # 決定木ノードの最小データ数
            'num_iteration': 100}         # 予測器(決定木)の数:イタレーション
        
    lgb_results = {}                                    # 学習の履歴を入れる入物
    model = lgb.train(params=params,                    # ハイパーパラメータをセット
                    train_set=lgb_train,              # 訓練データを訓練用にセット
                    valid_sets=[lgb_train, lgb_test], # 訓練データとテストデータをセット
                    valid_names=['Train', 'Test'],    # データセットの名前をそれぞれ設定
                    num_boost_round=100,              # 計算回数
                    early_stopping_rounds=10,         # アーリーストッピング設定
                    evals_result=lgb_results)         # 履歴を保存する

    loss_train = lgb_results['Train']['multi_logloss']  # 訓練誤差
    loss_test  = lgb_results['Test']['multi_logloss']   # 汎化誤差
    best_iteration = model.best_iteration               # 最良の予測器が得られたイタレーション数
    
    #print(best_iteration)
    
    y_pred = model.predict(test_info, num_iteration=model.best_iteration)
    
    """
    # accuracy
    print(" ------------------------------ ")
    print(" accuracy of train data")
    print("  " + str(clf.score(explanatory, objective)) )
    print(" accuracy of test data")
    print("  " + str(clf.score(explanatory_test, test_result)) )
    print(" ------------------------------ ")

    #print("Predicted probabilities:\n{}".format(clf.predict_proba(explanatory_test)))

    """

    # analysis
    #print(explanatory_test[0][18*12])
    #print(clf.predict_proba([explanatory_test[0]]))
    #print(test_result[0])

    maxn = 16
    rieki = 0
    
    ite_acc = 0
    ite_all = 0
    for i in range(len(explanatory_test)):
        ite_recover = 0
        jlist = []

        for j in range(maxn):
            recover_rate = explanatory_test[i][18*12 + j] * y_pred[i][j]
            #print(recover_rate)
            #print(temp[0])

            if recover_rate > 1.0:
                ite_recover += 1
                jlist.append(j)
            #end
        #end

        if ite_recover > 0:
            for j in jlist:
                ite_all += 1
                if test_result[i] == j+1:
                    rieki += 100 * explanatory_test[i][18*12 + j]
                    if explanatory_test[i][18*12 + j] / len(jlist) > 1.0:
                        ite_acc +=1
                    #end
                #end
            #end
        #end
    #end
    
    print("----------------------------")
    print(" 回収率 = オッズ*機械学習による勝率 > 1.0 を超えた馬全てに 100円 ずつかける場合")
    print(" 投資金：" + str(-ite_all*100))
    print(" 回収金：" + str(rieki))
    print(" 回収率：" + str(rieki / (ite_all*100)))
    print("----------------------------")



    # ana2
    maxn = 16
    rieki = 0
    st_big = 0

    censoring_max_odds = 5
    
    ite_all = 0
    for i in range(len(explanatory_test)):
        ite_all += 1
        
        be_rieki = rieki

        cor_num = (y_pred[i].tolist()).index(max(y_pred[i]))

        if test_result[i] == cor_num:
            rieki += 100 * explanatory_test[i][18*12 + int(test_result[i])-1]
        #end

        af_rieki = rieki
        
        diff = af_rieki - be_rieki
        if diff > 100*censoring_max_odds:
            #print(i)
            #print(diff)
            st_big += diff
        #end
    #end

    print("----------------------------")
    print(" 回収率 によらず全試合の一番確率の高いものに 100円 投資した場合")
    print(" 投資金：" + str(-ite_all*100))
    print(" 回収金：" + str(rieki))
    print(" 回収率：" + str(rieki / (ite_all*100)))
    print(" 回収金(" + str(censoring_max_odds) + "倍を除く)：" + str(rieki - st_big))
    print(" 回収率(" + str(censoring_max_odds) + "倍を除く)：" + str((rieki - st_big) / (ite_all*100)))
    print("----------------------------")

    # ana3
    maxn = 16
    ite_cor = 0
    
    ite_all = 0
    for i in range(len(explanatory_test)):
        ite_all += 1

        if test_result[i] == (y_pred[i].tolist()).index(max(y_pred[i])):
            ite_cor += 1
        #end    
    #end

    print("----------------------------")
    print(" 正答率：" + str(ite_cor / ite_all))
    print("----------------------------")
#end

# ------------------------------------------------
# ------------------------------------------------
if __name__ == "__main__":
    main()