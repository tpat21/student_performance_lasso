import pandas as pd
import sklearn
from sklearn.linear_model import Lasso
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model


def run():
    data = pd.read_csv("student-mat.csv",sep=";")

    cleanup_data = {
               "school": {"GP":0, "MS":1},
               "sex": {"F":0, "M":1},
               "address": {"U":0, "R":1},
               "famsize": {"LE3": 0, "GT3": 1},
               "Pstatus": {"A": 0, "T": 1},
               "address": {"U": 0, "R": 1},
               "Mjob": {"teacher": 0, "health": 1, "services": 2, "at_home":3, "other":4},
               "Fjob": {"teacher": 0, "health": 1, "services": 2, "at_home":3, "other":4},
               "reason": {"home": 0, "reputation": 1, "course":2, "other": 3},
               "guardian": {"mother": 0, "father": 1, "other": 2},
               "schoolsup": {"no": 0, "yes": 1},
               "famsup": {"no": 0, "yes": 1},
               "paid": {"no": 0, "yes": 1},
               "activities": {"no": 0, "yes": 1},
               "nursery": {"no": 0, "yes": 1},
               "higher": {"no": 0, "yes": 1},
               "internet": {"no": 0, "yes": 1},
               "romantic": {"no": 0, "yes": 1},
               }
    data.replace(cleanup_data,inplace=True)
    predict = "G3"


    X = np.array(data.drop([predict], 1))
    y = np.array(data[predict])


    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=.25)


    linear = linear_model.LinearRegression(fit_intercept=True)
    linear.fit(x_train,y_train)
    acc = linear.score(x_test,y_test)
    print("Before Lasso: {:.4f}".format(acc))


    lasso = Lasso(alpha=0.10)
    lasso_coef = lasso.fit(X,y).coef_

    # clf = linear_model.Lasso(alpha=0.1)
    # clf.fit(X, y)
    # print(clf.coef_)
    # print(clf.intercept_)



    names = data.drop("G3", axis=1).columns
    plt.plot(range(len(names)),lasso_coef)
    plt.xticks(range(len(names)),names,rotation=60)
    plt.show()



    new_data = data[["reason","Walc","famrel", "G1", "G2"]]
    new_X = np.array(new_data)

    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(new_X, y, test_size=.25)


    linear.fit(x_train,y_train)
    acc = linear.score(x_test,y_test)
    print("After Lasso: {:.4f}".format(acc))





if __name__ == '__main__':
    run()