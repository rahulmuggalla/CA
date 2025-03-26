from model.randomforest import RandomForest


def model_predict(data, df, name):
    print("RandomForest with ClassifierChain")
    model = RandomForest("RandomForest", data.X, data.y)
    model.train(data)
    model.predict(data.X_test)
    model.print_chained_accuracy(data.y_test, model.predictions)
    model.print_stagewise_accuracy(data.y_test, model.predictions)