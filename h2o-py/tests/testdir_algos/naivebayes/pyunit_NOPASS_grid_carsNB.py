from __future__ import print_function
import sys
sys.path.insert(1,"../../../")
import h2o
from tests import pyunit_utils
import random
import copy
from h2o.estimators.naive_bayes import H2ONaiveBayesEstimator
from h2o.grid.grid_search import H2OGridSearch

def grid_cars_NB():

    cars = h2o.import_file(path=pyunit_utils.locate("smalldata/junit/cars_20mpg.csv"))
    r = cars[0].runif(seed=42)
    train = cars[r > .2]

    validation_scheme = random.randint(1,3) # 1:none, 2:cross-validation, 3:validation set
    print("Validation scheme: {0}".format(validation_scheme))
    if validation_scheme == 2:
        nfolds = 2
        print("Nfolds: 2")
    if validation_scheme == 3:
        valid = cars[r <= .2]

    grid_space = pyunit_utils.make_random_grid_space(algo="naiveBayes")
    print("Grid space: {0}".format(grid_space))

    problem = random.sample(["binomial","multinomial"],1)
    predictors = ["displacement","power","weight","acceleration","year"]
    if problem == "binomial":
        response_col = "economy_20mpg"
    else:
        response_col = "cylinders"

    print("Predictors: {0}".format(predictors))
    print("Response: {0}".format(response_col))

    print("Converting the response column to a factor...")
    train[response_col] = train[response_col].asfactor()
    if validation_scheme == 3:
        valid[response_col] = valid[response_col].asfactor()

    print("Grid space: {0}".format(grid_space))
    print("Constructing the grid of nb models...")

    grid_space["compute_metrics"] = [False]

    cars_nb_grid = H2OGridSearch(H2ONaiveBayesEstimator, hyper_params=grid_space)
    if validation_scheme == 1:
        cars_nb_grid.train(x=predictors,y=response_col,training_frame=train)
    elif validation_scheme == 2:
        cars_nb_grid.train(x=predictors,y=response_col,training_frame=train,nfolds=nfolds)
    else:
        cars_nb_grid.train(x=predictors,y=response_col,training_frame=train,validation_frame=valid)

if __name__ == "__main__":
    pyunit_utils.standalone_test(grid_cars_NB)
else:
    grid_cars_NB()