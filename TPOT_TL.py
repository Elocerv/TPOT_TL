# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
a=2
print(a)
conda install numpy scipy scikit-learn pandas
pip install deap update_checker tqdm stopit
conda install pywin32
pip install xgboost
pip install dask[delayed] dask-ml
pip install scikit-mdr skrebate

pip install tpot





from tpot import TPOTClassifier

pipeline_optimizer = TPOTRegressor(generations=100, population_size=271,
                         offspring_size=None, mutation_rate=0.9,
                         crossover_rate=0.1,
                         scoring='neg_mean_squared_error', cv=5,
                         subsample=1.0, n_jobs=1,
                         max_time_mins=None, max_eval_time_mins=5,
                         random_state=None, config_dict=None,
                         template="RandomTree",
                         warm_start=False,
                         memory=None,
                         use_dask=False,
                         periodic_checkpoint_folder=None,
                         early_stop=None,
                         verbosity=2,
                         disable_update_check=False)

pipeline_optimizer.fit(X_train, y_train)
print(pipeline_optimizer.score(X_test, y_test))

pipeline_optimizer.export('tpot_exported_pipeline.py')

from tpot import TPOTRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

digits = load_boston()
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target,
                                                    train_size=0.75, test_size=0.25)

tpot = TPOTRegressor(generations=5, population_size=50, verbosity=2)
tpot.fit(X_train, y_train)
print(tpot.score(X_test, y_test))
tpot.export('tpot_boston_pipeline.py')





from tpot import TPOTClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data.astype(np.float64),
    iris.target.astype(np.float64), train_size=0.75, test_size=0.25)

tpot = TPOTClassifier(generations=5, population_size=50, verbosity=2)
tpot.fit(X_train, y_train)
print(tpot.score(X_test, y_test))
tpot.export('tpot_iris_pipeline.py')


from tpot import TPOTRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

housing = load_boston()
X_train, X_test, y_train, y_test = train_test_split(housing.data, housing.target,
                                                    train_size=0.75, test_size=0.25)

tpot = TPOTRegressor(generations=5, population_size=50, verbosity=2)
tpot.fit(X_train, y_train)
print(tpot.score(X_test, y_test))
tpot.export('tpot_boston_pipeline.py')


import numpy as np

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split

# NOTE: Make sure that the class is labeled 'class' in the data file
tpot_data = np.recfromcsv('C:/Users/ecervera/.spyder-py3/tpot_boston_pipeline.py', delimiter=';', dtype=np.float64)
features = np.delete(tpot_data.view(np.float64).reshape(tpot_data.size, -1),
                     tpot_data.dtype.names.index('class'), axis=1)
training_features, testing_features, training_target, testing_target = \
    train_test_split(features, tpot_data['class'], random_state=None)

exported_pipeline = GradientBoostingRegressor(alpha=0.85, learning_rate=0.1, loss="ls",
                                              max_features=0.9, min_samples_leaf=5,
                                              min_samples_split=6)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
































from tpot import TPOTClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np
from deap import creator
from sklearn.model_selection import cross_val_score

# Iris flower classification
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data.astype(np.float64),
    iris.target.astype(np.float64), train_size=0.75, test_size=0.25)

tpot = TPOTClassifier(generations=5, population_size=50, verbosity=2)
tpot.fit(X_train, y_train)
print(tpot.score(X_test, y_test))
# print part of pipeline dictionary
print(dict(list(tpot_obj.evaluated_individuals_.items())[0:2]))
# print a pipeline and its values
pipeline_str = list(tpot.evaluated_individuals_.keys())[0]
print(pipeline_str)
print(tpot.evaluated_individuals_[pipeline_str])
# convert pipeline string to scikit-learn pipeline object
optimized_pipeline = creator.Individual.from_string(pipeline_str, tpot._pset) # deap object
fitted_pipeline = tpot._toolbox.compile(expr=optimized_pipeline ) # scikit-learn pipeline object
# print scikit-learn pipeline object
print(fitted_pipeline)
# Fix random state when the operator allows  (optional) just for get consistent CV score 
tpot._set_param_recursive(fitted_pipeline.steps, 'random_state', 42)
# CV scores from scikit-learn
scores = cross_val_score(fitted_pipeline, X_train, y_train, cv=5, scoring='accuracy', verbose=0)
print(np.mean(scores))
print(tpot.evaluated_individuals_[pipeline_str][1])








###################################################


import pandas as pd
import numpy as np


telescope_data=pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/magic/magic04.data',header=None)
telescope_data.head()
telescope_data.columns = ['fLength', 'fWidth','fSize','fConc','fConcl','fAsym','fM3Long','fM3Trans','fAlpha','fDist','class']
telescope_data.head()
telescope_data.info()
telescope_data.describe()
telescope_data['class'].value_counts()

telescope_shuffle=telescope_data.iloc[np.random.permutation(len(telescope_data))]
tele=telescope_shuffle.reset_index(drop=True)
tele.head()

tele['class']=tele['class'].map({'g':0,'h':1})

tele.head()
tele_class = tele['class'].values
pd.isnull(tele).any()

tele = tele.fillna(-999)

from sklearn.cross_validation import train_test_split
training_indices, validation_indices = training_indices, testing_indices = train_test_split(tele.index, stratify = tele_class, train_size=0.75, test_size=0.25)
training_indices.size, validation_indices.size

from tpot import TPOTClassifier
from tpot import TPOTRegressor

tpot = TPOTClassifier(generations=5,verbosity=2)

tpot.fit(tele.drop('class',axis=1).loc[training_indices].values,tele.loc[training_indices,'class'].values)
tpot.score(tele.drop('class',axis=1).loc[validation_indices].values,tele.loc[validation_indices, 'class'].values)

tpot.export('tpot_MAGIC_Gamma_Telescope_pipeline.py')





############################################################################


import pandas as pd
import numpy as np


TL_data=pd.read_csv('file:///C:/Users/ecervera/Documents/Placenta_preeclampsia/Macro/Lanadata/NewP2.csv',header=None)
TL_data.head()
TL_data[0][0]="num"
TL_data.loc[0]
TL_data.columns =TL_data.loc[0]
TL_data=TL_data.drop('num',axis=1)
TL_data=TL_data.drop(0)

TL_data.head()
TL_data.info()
TL_data.describe()
TL_data['Ethgroup.Others'].value_counts()

TL_shuffle=TL_data.iloc[np.random.permutation(len(TL_data))]
TL=TL_shuffle.reset_index(drop=True)
TL.head()

#tele['class']=tele['class'].map({'g':0,'h':1})

TL.head()
#TL_class = TL['Sample_Group.Cases'].values
TL_class = TL['Telo.Length'].values
pd.isnull(TL).any()

TL = TL.fillna(-999)

TL=TL.astype(np.float64)

from sklearn.model_selection import train_test_split
#training_indices, validation_indices = training_indices, testing_indices = train_test_split(TL.index, stratify = TL_class, train_size=0.75, test_size=0.25)
training_indices, validation_indices = training_indices, testing_indices = train_test_split(TL.index, train_size=0.75, test_size=0.25)
training_indices.size, validation_indices.size

from tpot import TPOTClassifier
from tpot import TPOTRegressor

#tpot = TPOTClassifier(generations=5,verbosity=2)
tpot = TPOTRegressor(generations=5,verbosity=2,scoring = 'r2')

#tpot.fit(TL.drop('Sample_Group.Cases',axis=1).loc[training_indices].values,TL.loc[training_indices,'Sample_Group.Cases'].values)
#tpot.score(TL.drop('Sample_Group.Cases',axis=1).loc[validation_indices].values,TL.loc[validation_indices, 'Sample_Group.Cases'].values)
A=tpot.fit(TL.drop('Telo.Length',axis=1).loc[training_indices].values,TL.loc[training_indices,'Telo.Length'].values)
tpot.score(TL.drop('Telo.Length',axis=1).loc[validation_indices].values,TL.loc[validation_indices, 'Telo.Length'].values)



tpot.export('C:\\Users\\ecervera\\tpot_MAGIC_Gamma_Telescope_pipeline.py')
tpot.export('tpot_MAGIC_Gamma_Telescope_pipeline.py')



extratriseregressor()

#TL_data[3]
#TL_data[1][0]
model = tpot.fit(TL.drop('Telo.Length',axis=1).loc[training_indices], TL.loc[training_indices,'Telo.Length'])
perm = PermutationImportance(model, n_iter=100).fit(TL.drop('Telo.Length',axis=1).loc[training_indices], TL.loc[training_indices,'Telo.Length'])
feat_imp = perm.feature_importances_
 










import eli5
from eli5.sklearn import PermutationImportance
from sklearn.svm import SVC
from sklearn.feature_selection import SelectFromModel

# ... load data

perm = PermutationImportance(SVC(), cv=5)
perm.fit(TL.drop('Telo.Length',axis=1).loc[training_indices], TL.loc[training_indices,'Telo.Length'])

# perm.feature_importances_ attribute is now available, it can be used
# for feature selection - let's e.g. select features which increase
# accuracy by at least 0.05:
sel = SelectFromModel(perm, threshold=0.05, prefit=True)
X_trans = sel.transform(X)

# It is possible to combine SelectFromModel and
# PermutationImportance directly, without fitting
# PermutationImportance first:
sel = SelectFromModel(
    PermutationImportance(SVC(), cv=5),
    threshold=0.05,
).fit(X, y)
X_trans = sel.transform(X)









