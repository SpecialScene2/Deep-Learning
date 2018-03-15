# Feel free to add any functions, import statements, and variables.

def predict(file):
    import pandas as pd
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    import random
    from sklearn.model_selection import train_test_split
    from sklearn.feature_selection import RFE
    from sklearn.linear_model import LogisticRegression
    random.seed(3)

    # 데이터 불러오기

    train_df = pd.read_csv('train.csv')
    val_df = pd.read_csv('valid.csv')
    test_df = pd.read_csv(file)
    result_df = pd.read_csv('result.csv')

    X = train_df.drop('Class', axis=1)
    y = train_df['Class']

    val_X = val_df.drop('Class', axis=1)
    val_y = val_df['Class']

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=3)

    # 1. LogisticRegression()
    class_weight = 'balanced'
    lg_model = LogisticRegression()

    selector = RFE(lg_model)
    selector.fit(X_train, y_train)

    print(selector.n_features_)
    print(selector.ranking_)
    print(selector.support_)

    # 변수 위치 저장
    selected_cols = selector.support_
    X_RFE_train = X_train.loc[:, selected_cols]
    X_RFE_test = X_test.loc[:, selected_cols]

    lg_model.fit(X=X_RFE_train, y=y_train)
    lg_model.score(X=X_RFE_test, y=y_test)

    selected_cols = selector.support_
    val_X_RFE_train = val_X.loc[:, selected_cols]

    lg_model.score(X=val_X_RFE_train, y=val_y)

    val_df['logisticClass'] = lg_model.predict(X=val_X_RFE_train)

    notEqualCount = 0
    for i in range(0, 10000):
        if val_df.loc[i, 'Class'] != val_df.loc[i, 'logisticClass']:
            notEqualCount += 1
    print('Before NotEqualCount :', notEqualCount)

    # validation Set에 적용
    val_df = val_df.reset_index().drop('index', axis=1)

    totalPredicted = len(val_df[val_df['logisticClass'] == 1])

    # precision(정밀도)
    logisticTP = 0
    for i in range(0, 10000):
        if val_df.loc[i, 'Class'] == 1 and val_df.loc[i, 'logisticClass'] == 1:
            logisticTP += 1
    print(logisticTP)

    precision = logisticTP / totalPredicted
    print('Precision :', precision * 100)

    # TP + FN
    actualPositive = 100

    # recall(재현율)
    recall = logisticTP / actualPositive
    print('Recall :', recall * 100)
    print('F-Score : ', 2 * (precision * recall) / (precision + recall))

    # 2. Random Forest
    from sklearn.model_selection import GridSearchCV

    # 모든 변수를 다 집어 넣었을 때 hyperparameter 결정
    # parameters2 = {'max_depth': range(2, 10), 'n_estimators': [2000, 1000, 500, 100]}
    # forest_model = GridSearchCV(RandomForestClassifier(random_state=3), parameters2, n_jobs=1)
    # forest_model.fit(X=X_train, y=y_train)
    # best_forest_model = forest_model.best_estimator_
    # print('Best Model :\n', best_forest_model, '\n', '\nBest Score :', forest_model.best_score_, '\n',
    #       '\nBest parameters :', forest_model.best_params_)

    # 위에서 찾아낸 모델 생성
    class_weight = 'balanced'
    rf_model = RandomForestClassifier(bootstrap=True, class_weight=class_weight, criterion='gini',
                                      max_depth=6, max_features='auto', max_leaf_nodes=None,
                                      min_samples_leaf=1,
                                      min_samples_split=2, min_weight_fraction_leaf=0.0,
                                      n_estimators=2000, n_jobs=1, oob_score=False, random_state=3,
                                      verbose=1, warm_start=False)

    rf_model.fit(X_train, y_train)

    # feture선택을 위한 피쳐중요도(feature_importances) 확인
    # 확인 결과 12순위 피쳐부터는 중요도가 1%이하로 떨어져서 피쳐갯수를 1순위로 정함.
    importances = rf_model.feature_importances_
    indices = np.argsort(importances)[::-1]
    for f in range(X.shape[1]):
        print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

    # 위의 결과를 바탕으로 10개의 피쳐를 RFE로 통해서 생성해봄
    # selector = RFE(rf_model, n_features_to_select=11)
    # selector.fit(X_train, y_train)
    #
    # print(selector.n_features_)
    # print(selector.ranking_)
    # print(selector.support_)
    #
    # # 변수 위치 저장
    # selected_cols = selector.support_
    selected_cols = [False, False, False,True, True, False, False, True, False, False, True, True, True, False, True,
                     False, True, True, False, True, False, False, False, False, False, False, False, False, False,
                     True]
    X_RFE_train = X_train.loc[:, selected_cols]
    X_RFE_test = X_test.loc[:, selected_cols]

    rf_model.fit(X=X_RFE_train, y=y_train)
    rf_model.score(X=X_RFE_test, y=y_test)

    val_X_RFE_train = val_X.loc[:, selected_cols]

    rf_model.score(X=val_X_RFE_train, y=val_y)

    val_df['ForestClass'] = rf_model.predict(X=val_X_RFE_train)

    probResult = rf_model.predict_proba(X=val_X_RFE_train)
    len(probResult)

    notEqualCount = 0
    for i in range(0, 10000):
        if val_df.loc[i, 'Class'] != val_df.loc[i, 'ForestClass']:
            notEqualCount += 1
    print('Before NotEqualCount :', notEqualCount)

    val_df = val_df.reset_index().drop('index', axis=1)

    totalPredicted = len(val_df[val_df['ForestClass'] == 1])
    print(totalPredicted)

    # precision(정밀도)
    ForestTP = 0
    for i in range(0, 10000):
        if val_df.loc[i, 'Class'] == 1 and val_df.loc[i, 'ForestClass'] == 1:
            ForestTP += 1
    print(ForestTP)

    precision = ForestTP / totalPredicted
    print('Precision :', precision * 100)

    # TP + FN
    actualPositive = 100

    # recall(재현율) 및 F-Score
    recall = ForestTP / actualPositive
    print('Recall :', recall * 100)
    print('F-Score : ', 2 * (precision * recall) / (precision + recall))

    test_X = test_df

    # selected_cols = selector.support_
    X_RFE_test = test_X.loc[:, selected_cols]

    result_df['ForestClass'] = rf_model.predict(X=X_RFE_test)
    test_df['ForestClass'] = rf_model.predict(X=X_RFE_test)
    return list(result_df['ForestClass'])

def write_result(classes):
    # You don't need to modify this function.
    with open('result.csv', 'w') as f:
        f.write('Index,Class\n')
        for idx, l in enumerate(classes):
            f.write('{0},{1}\n'.format(idx, l))

def main():
    # You don't need to modify this function.
    classes = predict('test.csv')
    write_result(classes)


if __name__ == '__main__':
    # You don't need to modify this part.
    main()