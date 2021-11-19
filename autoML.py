import numpy as np
from sklearn import metrics
import time
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, train_test_split

# clustering scoring
def purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    # return purity
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix) 



# dataset : scaling & encoding datasets
# model : model
# param_name : using parameter name
# parameter : check parameter
# type : 0 - classification / 1 - clustering
def autoML(type, feature, target, model, parameter, model_op = None):

    if type == 0:  # classifiction
        notEnough = True

        param_list = parameter
        best_score = 0
        best_datatype = 0
        best_param = None
        model_score = []

        current_score = 0
        pre_score = 0
        pre_pre_score = 0

        while notEnough:
            param_name = parameter.keys()
            param_name = list(param_name)[0]
            for k in parameter[param_name]:
                start_time = time.time()

                sub_best_score = 0
                sub_best_datatype = None
                
                for key, value in feature.items():

                    #onehot 고려 안함
                    if "onehot" in key:
                        break

                    X_train, X_test, y_train, y_test = train_test_split(value, target,
                                                                        test_size=0.2)  # 이부분 틀렸으면 고쳐두길 바람

                    # model 만들기
                    if model_op != None:
                        models = GridSearchCV(model, param_grid={param_name : [k]})
                    else:
                         models = model(k)

                    models.fit(X_train, y_train)

                    if model_op != None:
                        bestModel = models.best_estimator_
                        current_score = bestModel.score(X_test, y_test)
                    else:
                        predicted = models.predict(X_test)
                        current_score = models.score(predicted, y_test)

                    if sub_best_score < current_score:
                        sub_best_score = current_score
                        sub_best_datatype = key

                # 일단 저장하기
                model_score.append(sub_best_score)

                if best_score < sub_best_score:
                    best_score = sub_best_score
                    best_datatype = sub_best_datatype
                    best_param = k

                # early stop 여부 결정 (값이 감소중이라면 정지)
                # 값이 계속 감소하면 미리 종료
                if pre_pre_score >= pre_score:
                    if pre_score >= current_score:
                        if current_score >= sub_best_score:
                            ck = 0
                            break

                pre_pre_score = pre_score
                pre_score = current_score
                current_score = sub_best_score

            print(time.time() - start_time)

            if not notEnough:
                break

            ck = 0
            #param을 다 돌았을 때, 
            # 마지막 값이 최상의 score인지 확인하고, 
            # 그렇다면 이전 값에서 얼마나 증가했는지 보고, 기준치 이하면 안함
            if best_score == current_score:
                if (current_score - pre_score) / pre_score > 0.05:
                    adder = parameter[-1] - parameter[-2]
                    parameter = [parameter[-1] + adder, parameter[-1] + adder*2]
                    param_list = param_list.extend(parameter)
                    ck = 1

        # Return값으로는 각 파라미터마다의 최상의 값 + 결론으로 얻은 최상의 값
        print("best score = ", best_score, " | best datatype = ", best_datatype, "| best param = ", best_param)

        #for elary stop
        if len(model_score) < len(param_list):
            param_list = param_list[0:4]
        return model_score, param_list


    else: #clusteing
        print("clusteinrg start")
        param_list = parameter
        best_score = 0
        best_datatype = 0
        best_param = None
        model_score = []

        current_score = 0
        pre_score = 0
        pre_pre_score = 0

        ck = 1
        
        while(ck):
            for k in parameter:
                start_time = time.time()

                sub_best_score = 0
                sub_best_datatype = None

                for key, value in feature.items():
                    models = model(k)
                    labels = models.fit_predict(value)

                    score = purity_score(target, labels)

                    if sub_best_score < score:
                        sub_best_score = score
                        sub_best_datatype = key

                # 일단 저장하기
                model_score.append(sub_best_score)

                if best_score < sub_best_score:
                    best_score = sub_best_score
                    best_datatype = sub_best_datatype
                    best_param = k

                # early stop 여부 결정 (값이 감소중이라면 정지)
                # 값이 계속 감소하면 미리 종료
                if pre_pre_score >= pre_score:
                    if pre_score >= current_score:
                        if current_score >= sub_best_score:
                            ck = 0
                            break

                pre_pre_score = pre_score
                pre_score = current_score
                current_score = sub_best_score

            print(time.time() - start_time)

            if ck == 0:
                break

            ck = 0
            #param을 다 돌았을 때, 
            # 마지막 값이 최상의 score인지 확인하고, 
            # 그렇다면 이전 값에서 얼마나 증가했는지 보고, 기준치 이하면 안함
            if best_score == current_score:
                if (current_score - pre_score) / pre_score > 0.05:
                    adder = parameter[-1] - parameter[-2]
                    parameter = [parameter[-1] + adder, parameter[-1] + adder*2]
                    param_list = param_list.extend(parameter)
                    ck = 1
                


        # Return값으로는 각 파라미터마다의 최상의 값 + 결론으로 얻은 최상의 값
        print("best score = ", best_score, " | best datatype = ", best_datatype, "| best param = ", best_param)



        #for elary stop
        if len(model_score) < len(param_list):
            param_list = param_list[0:4]
        return model_score, param_list