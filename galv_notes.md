

# <span style="color:red">Math/Stats</span>

* [Galvanize Short Course for Stats](https://galvanizeopensource.github.io/stats-shortcourse/)
* [PDF Book: Computer Age Statistical Inference Book](https://web.stanford.edu/~hastie/CASI_files/PDF/casi.pdf)

## Calculus

* [Derivative Rules](https://en.wikipedia.org/wiki/Differentiation_rules)

## Linear Algebra

* [Wiki](https://en.wikipedia.org/wiki/Linear_algebra)
* [MIT Course](https://ocw.mit.edu/courses/mathematics/18-02-multivariable-calculus-fall-2007/)



## Probability & Distributions

* [Distribution Applets](https://homepage.divms.uiowa.edu/~mbognar/)
* [Cheatsheet](https://static1.squarespace.com/static/54bf3241e4b0f0d81bf7ff36/t/55e9494fe4b011aed10e48e5/1441352015658/probability_cheatsheet.pdf)
* [Scipy Stats Functions](https://docs.scipy.org/doc/scipy/reference/stats.html)
* [Scipy Stats](https://stackoverflow.com/questions/37559470/what-do-all-the-distributions-available-in-scipy-stats-look-like#37559471)
* [Scipy Cheatsheet](https://onedrive.live.com/?authkey=%21APW6U3IpfyxCXmY&cid=233807F4EE406C1F&id=233807F4EE406C1F%215688&parId=233807F4EE406C1F%215605&o=OneUp)


thing | Discrete | Continuous | R
------|----------|----------| ---
x to density | pmf | pdf | d
x -> area to left | cdf  | cdf | p
area to left -> x | ppf | ppf | q

Plot Continuous Distribution
```python
def plot_continuous(dist):
    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(4, 5))
      # Plot hist
    rvs = dist.rvs(size=1000)
    ax[0].hist(rvs, normed=True, alpha=0.2, histtype='stepfilled')
    x=np.linspace(dist.ppf(0.01), dist.ppf(0.99), 50)
    ax[0].plot(x, dist.pdf(x), '-', lw=2)
    ax[0].set_title( dist.dist.name.title() + ' PDF')
    ax[0].set_ylabel('p(X=x)')
      # Plot cdf.
    ax[1].plot(x, dist.cdf(x), '-', lw=2)
    ax[1].set_title( dist.dist.name.title() + ' CDF')
    ax[1].set_ylabel('p(X<=x)')
    ax[1].set_xlabel('x')
    return (fig, ax)

## And then call it like
lam = 1  # lambda
exponential = st.expon(scale=1/lam)
plot_continuous(exponential);
```

Plot Discrete Distribution
```python
def plot_discrete(dist):
    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(4, 5))
     # Plot hist
    rvs = dist.rvs(size=1000)
    w = np.ones_like(rvs)/ float(len(rvs))
    ax[0].hist(rvs, weights=w, alpha=0.2, histtype='stepfilled')
      # Plot pmf.
    k = np.arange(dist.ppf(0.01), dist.ppf(0.99)+1)
    ax[0].plot(k, dist.pmf(k), 'bo', lw=2);
    ax[0].set_title( dist.dist.name.title() + ' PMF')
    ax[0].set_ylabel('p(X=k)')
     # Plot cdf.
    ax[1].plot(k, dist.cdf(k), 'bo', lw=2);
    ax[1].set_title( dist.dist.name.title() + ' CDF')
    ax[1].set_ylabel('p(X<=k)')
    ax[1].set_xlabel('k');
    return (fig, ax)

## And then call it with
binomial=st.binom(n=10,p=0.6)
plot_discrete(binomial);
```
## Bootstrap

The bootstrap takes 200-2000 samples of length equal to sample, and calculates the statistic of interest.  This process creates a distribution for the statistic and can be used to create confidence intervals. It is computationally expensive, but is more versatile that MLE.

* use `np.percentile(array, [2.5,97.5])`
* bootstrap
```python
# INPUT - np array OUTPUT - bootstrap confidence intervals
def bootstrap_ci(lst, bootstraps=1000, ci=95):
    n = len(lst)
    bootstraps_list = ([np.mean([lst[np.random.randint(n)] for i in np.arange(n)]) for i in np.arange(bootstraps)])
    conf_int = np.percentile(bootstraps_list, [(100-ci)/2,100-((100-ci)/2)])
    return print('The {} conf_int for the sample is {}.'.format(ci, conf_int))
```

## Maximum Likelihood Estimation

* [Maximum Liklihood Estimation](https://ocw.mit.edu/courses/mathematics/18-05-introduction-to-probability-and-statistics-spring-2014/readings/MIT18_05S14_Reading10b.pdf)
* [Maximum Likelihood Estimation (1)](http://statweb.stanford.edu/~susan/courses/s200/lectures/lect11.pdf)

## Experimental Design & Hypothesis Tests

* [Sampling](https://en.wikipedia.org/wiki/Sampling_(statistics))
* [Power](https://en.wikipedia.org/wiki/Statistical_power#Factors_influencing_power) - `Pr(Reject H0 | H1 is true)`
[Power](http://my.ilstu.edu/~wjschne/138/Psychology138Lab14.html)
* [Quick-R Power](https://www.statmethods.net/stats/power.html)
* [ANOVA vs T-test](https://keydifferences.com/difference-between-t-test-and-anova.html)

A/B Test of two sample proportions (e.g. sign up rate on website)
```python
from z_test import z_test
z_test.z_test(mean1, mean2, n1, n2, effect_size=.01,two_tailed=False)
```

Kolmogorov-Smirnov - Null: Two array are from same distribution

```python
import scipy.stats as st
st.ks_2samp(randpois, count_by_month)
```
## Bayesian Methods

[PyMC3](http://docs.pymc.io/notebooks/getting_started.html)
```Python
from pymc3 import Normal, Model, DensityDist, sample
from pymc3.math import log, exp
from pymc3 import df_summary

with Model() as disaster_model:
    switchpoint = DiscreteUniform('switchpoint', lower=0, upper=n_years)
```

See PyMC3 Distributions
```python
with Model() as disaster_model:

    switchpoint = DiscreteUniform('switchpoint', lower=0, upper=n_years)
```

PyMC3 switchpoint
```python
from pymc3.math import switch

with disaster_model:

    rate = switch(switchpoint >= np.arange(n_years), early_mean, late_mean)
```

Changepoint, Switchpoint analysis
```python
import scipy.stats as st
import numpy as np
import pymc3 as pm
import theano.tensor as tt
import matplotlib.pyplot as plt
%matplotlib inline

## Draw two random variables from Poisson
pois1 = st.poisson.rvs(5, size=100)
pois2 = st.poisson.rvs(6.5, size=100)
newlist = np.append(pois1, pois2)
n_newlist = len(newlist)

## assign lambdas and tau to stochastic variables
with pm.Model() as model:
    alpha = 1.0/newlist.mean()  # Recall count_data is the
                                   # variable that holds our txt counts
    lambda_1 = pm.Exponential("lambda_1", alpha)
    lambda_2 = pm.Exponential("lambda_2", alpha)
    tau = pm.DiscreteUniform("tau", lower=0, upper=n_newlist)

## create a combined function for lambda (it is still a RV)    
with model:
    idx = np.arange(n_newlist) # Index
    lambda_ = pm.math.switch(tau >= idx, lambda_1, lambda_2)

## combine the data with our proposed data generation scheme    
with model:
    observation = pm.Poisson("obs", lambda_, observed=newlist)

## inference
with model:
    step = pm.Metropolis()
    trace = pm.sample(1000, tune=5000,step=step)


    fig = plt.figure(figsize=(12.5,5))
    ax = fig.add_subplot(111)

    N = tau_samples.shape[0]
    expected_texts_per_day = np.zeros(n_newlist)
    for day in range(0, n_newlist):
        ix = day < tau_samples
        expected_texts_per_day[day] = (lambda_1_samples[ix].sum()
                                       + lambda_2_samples[~ix].sum()) / N

    ax.plot(range(n_newlist), expected_texts_per_day, lw=4, color="#E24A33",
             label="expected number of referrals")
    ax.set_xlim(0, n_newlist)
    ax.set_xlabel("Day")
    ax.set_ylabel("Expected # referrals")
    ax.set_title("Expected number of referrals")
    ax.bar(np.arange(len(newlist)), newlist, color="#348ABD", alpha=0.65,label="observed referrals per day")
    ax.legend(loc="upper left");
```

[MIT Bayesian Inference with Discrete Priors](https://ocw.mit.edu/courses/mathematics/18-05-introduction-to-probability-and-statistics-spring-2014/readings/MIT18_05S14_Reading11.pdf)


#### Math_Stats Miscellaneous

* [Book: Probabilistic Programming](http://nbviewer.jupyter.org/github/CamDavidsonPilon/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers/tree/master/)
* [Maximum A Posteriori](https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/ppt/22-MAP.pdf)
___

# <span style="color:red">Coding & Environment</span>

## GitHub

* [Interactive Tool](http://ndpsoftware.com/git-cheatsheet/previous/git-cheatsheet.html)
* [Adding a Github repo to house local project](https://help.github.com/articles/adding-an-existing-project-to-github-using-the-command-line/)
* Check origin: `git remote -v`

### Pair Workflow
* A:
  * needs to add B as a collaborator for that repo on his/her Github.
  * adds a branch, e.g. `$ git checkout -b pair_morning`
  * starts coding (B navigating and helping)
  * Add, commit, push the branch to Github when it’s B’s turn to code.
    * e.g. `$ git push origin pair_morning`
* B:
  * After A adds B as collaborator, adds A’s repo as a remote:
    * `$ git remote add <partner-name> <partner-remote-url>`
  * Help A!
  * When B’s turn to code comes:
    * `$ git fetch <partner-name>`
    * `$ git checkout --track <partner-name>/<branch-name>`
  * Starts coding (A navigating and helping)
  * When A’s turn comes again:
    * `$ git push <partner-name> <branch-name>`

**Switching**
* A:
  * `$ git checkout <branch-name>`
  * `$ git pull <remote-name> <branch-name>`
  * A works on code (B collaborating). ABC!
  *  `$ git push <remote-name> <branch-name>`
* B:
  * `$ git checkout <branch-name>`
  * `$ git pull <remote-name> <branch-name>`
  * B works on code (A collaborating). ABC!
  * `$ git push <remote-name> <branch-name>`



## SQL

* [SQL for Data Scientists](http://downloads.bensresearch.com/SQL.pdf)
* [ModeAnalytics: SQL School](http://sqlschool.modeanalytics.com/)
* [7 Handy SQL features for Data Scientists](http://blog.yhathq.com/posts/sql-for-data-scientists.html)
* [Postgres Docs](https://www.postgresql.org/docs/)
* [Postgres Guide](http://postgresguide.com/)
* [Statistics in SQL](https://github.com/tlevine/sql-statistics)
* [Intro to SQL](http://bensresearch.com/downloads/SQL.pdf)

Create a database `CREATE DATABASE readychef;`
load data into empty database `$ psql readychef < readychef.sql`
Navigate to a db `psql db`

Order of Execution
`SELECT FROM JOIN WHERE GROUPBY HAVING ORDERBY LIMIT`

## Python

* [Style Guide](https://www.python.org/dev/peps/pep-0008/)
* [Pythonic Code](http://docs.python-guide.org/en/latest/writing/style/)
* Create a Python 2 environment `conda create -n py2 python=2 anaconda`
* [Classes and Objects Youtube Videos](https://www.youtube.com/watch?v=ZDa-Z5JzLYM&list=PL-osiE80TeTt2d9bfVyTiXJA-UTHn6WwU&index=37)

```Python
## stuff you want to run no matter what
## whether directly or from an import
## typically functions

if __name__ == '__main__':
  ## stuff you want to run only if running the file directly
```
[read](https://stackoverflow.com/questions/419163/what-does-if-name-main-do)

Categorical Var from Continuous Var
```python
df['new_categ_var'] = pd.cut(df['continuous_var'], [3, 6, 10], labels=['<3','4-6','7-10'])
```

python starter
```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
pd.set_option("display.max_columns", 100)
%matplotlib inline
```
python starter uncommon
```python
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.pipeline import Pipeline
from regression_tools.dftransformers import (
    ColumnSelector, Identity, FeatureUnion, MapFeature, Intercept)
from sklearn.linear_model import LogisticRegression
```



### psycopg (Python to PostgreSQL)

Cursor operations typically goes like the following:
* execute a query
* fetch rows from query result if it is a SELECT query because it is iterative, previously fetched rows can only be fetched again by rerunning the query
* close cursor through .close()

* [Browse Psycopg documentation](http://initd.org/psycopg/docs/)

```python
import psycopg2
from datetime import datetime
conn = psycopg2.connect(dbname='socialmedia', user='postgres', host='/tmp')
c = conn.cursor()
today = '2014-08-14'
# This is not strictly necessary but demonstrates how you can convert a date
# to another format
ts = datetime.strptime(today, '%Y-%m-%d').strftime("%Y%m%d")
c.execute(
    '''CREATE TABLE logins_7d AS
    SELECT userid, COUNT(*) AS cnt, timestamp %(ts)s AS date_7d
    FROM logins
    WHERE logins.tmstmp > timestamp %(ts)s - interval '7 days'
    GROUP BY userid;''', {'ts': ts}
)
conn.commit()
conn.close()
```

## Mongodb

* [SQL to Mongodb translator](https://docs.mongodb.com/manual/reference/sql-aggregation-comparison/)


### Pymongo

* [Cheat sheet](https://gist.github.com/stevemclaugh/530979cddbc458d25e37c9d4703c13f6)


### Unit Testing

`unittest` is a package that you use when you're doing unit testing.  You write the code, then the test code in a separate file. In the test code, one of the files that you load is the actual code that you're testing

`$ python -m unittest test.unit_test_sample`

run a test `make test`

```python
import unittest
from primes import is_prime

class PrimesTestCase(unittest.TestCase):
    """Tests for `primes.py`."""

    def test_is_five_prime(self):
        """Is five successfully determined to be prime?"""
        self.assertTrue(is_prime(5))

if __name__ == '__main__':
```

* [Tutorial](https://jeffknupp.com/blog/2013/12/09/improve-your-python-understanding-unit-testing/)



### Python to R, R to Python

In a python script, you can write ".R" files and then run them, all while still in Python.  Use code:
```Python
%%writefile getmtcars.R
a = mtcars
write.csv(a, 'cars.csv')
```
```python
 !Rscript getmtcars.R
```

 RPy2


___

# <span style="color:red">Machine learning</span>

* [An Introduction to Statistical Learning - ISLR](http://www-bcf.usc.edu/~gareth/ISL/ISLR%20Seventh%20Printing.pdf)
* [SK Learn](http://scikit-learn.org/stable/)
* [Model Smoothers](http://madrury.github.io/smoothers/)

**Parametric Model** - makes assumption about underlying form of the data (i.e. linear)

**Non-parametric Models** -

## Data Cleaning

Split data into x, y for training and testing
```python
from sklearn.model_selection import train_test_split
## make a train test split
X_train, X_test, y_train, y_test = train_test_split(X, y)
```

Scaling - normalizes values of each feature, mean = 0, sd = 1
```python
## scale using pandas
df['var'] = ((df['var']-(df['var'].mean())))/(df['var'].std())
```

Dummies
```Python
## in pandas
df = pd.DataFrame({'country': ['usa', 'canada', 'australia','japan','germany']})
pd.get_dummies(df,prefix=['country'])
```
Get Dummies
```python
Dummies = pd.get_dummies(df['var_with_multiple_cat_levels'])
df.drop('var_with_multiple_cat_levels', inplace=True, axis=1)
```


Binarizing
```Python
lb = preprocessing.LabelBinarizer()
lb.fit([1, 2, 6, 4, 2])
print(lb.transform((1,4)))
print(lb.classes_)
```

Imputation
```Python
## See number of nulls
test_scores.isnull().sum(0)

## Strategy could be 'mean', 'most_frequent'
from sklearn.preprocessing import Imputer
imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
imp.fit([[1, 2], [np.nan, 3], [7, 6]])
X = [[np.nan, 1], [6, np.nan], [3, 6]]
imp.transform(X)

## outputs:
array([[ 4.        ,  1.        ],
       [ 6.        ,  3.66666667],
       [ 3.        ,  6.        ]])
```

Pipelines
```python
## Create an object that pipelines three transformations
wells_pipeline = Pipeline([
    ('select_best_3', SelectKBest(chi2, k=3)),
    ('standardize', StandardScaler()),
    ('regression', LogisticRegression())
])

## Fit that object to data.
wells_pipeline.fit(X_wells, y_wells)
```

Pipeline General Form
```python
pipeline_fit_object = Pipeline([
    ('name_first_pipeline_piece', ColumnSelector(args)),
    ('name_secon_pipeline_piece', NaturalCubicSpline(args))
])

Pipeline([('name1', Thing1(args)), ('name2', Thing2(args))]
```

Pipeline of pipelines
```python
feature_pipeline = FeatureUnion([
    ('pipeline_name_fit', pipeline_name_fit),
    ('pipeline2_name_fit', cement_fit)
])
```

Fitting it altogther
```python
feature_pipeline.fit(dataframe)
features = feature_pipeline.transform(dataframe)
```

## General Linear Models

* [Multicollinearity](https://en.wikipedia.org/wiki/Multicollinearity)


### Linear Regression

Cost Function  

Dropping Variables:
```python
best_3_selector = SelectKBest(chi2, k=3)
best_3_selector.fit(X_wells, y_wells)
```

Sk learn model fitting
```python
model = LinearRegression(fit_intercept=False)
model.fit(features.values, y=concrete['compressive_strength'])
## display param coefficients
display_coef(model, features.columns)
```
Bootstrap estimates of coefficients
```python
models = bootstrap_train(
    LinearRegression,
    features.values,
    concrete['compressive_strength'],
    fit_intercept=False
)
fig, axs = plot_bootstrap_coefs(models, features.columns, n_col=4)
fig.tight_layout()
```


Hypothesis Tests
* Breusch-Pagan - tests for Heteroscedasticity
* Shaprio-Wilk - tests for normality of residuals


### Logistic Regression


Two variables, plotting y's
```Python
fig1 = plt.figure(figsize=(12,10))
ax1 = fig.add_subplot(111)
ax1.scatter(X_dogs[:,0], X_dogs[:,1], color='b', label='dogs')
ax1.scatter(X_horses[:,0], X_horses[:,1], color='r', label='horses')
ax1.legend(shadow=True, fontsize='xx-large')
ax1.set_xlabel('Weight (lb)',fontsize=font_size)
ax1.set_ylabel('Height (in)',fontsize=font_size)
ax1.set_title('Horse or dog?',fontsize=font_size)
plt.show()
```

3D Plotting
```python
fig = plt.figure(figsize=(12,10))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:,0], X[:,1], y, c=y,s=35)
ax.set_xlabel('Weight (lb)',fontsize=font_size,labelpad=25.0)
ax.set_ylabel('Height (in)',fontsize=font_size,labelpad=25.0)
ax.set_zlabel('Horseness',fontsize=font_size,labelpad=25.0)
plt.tight_layout()
plt.show()
```

Assessing Fit, Model Accuracy, Cross Validation
```python
roc_auc = cross_val_score(ML_instance, X_pandas, y_pandas, scoring='roc_auc', cv=8)
```

Test Logistic Lasso Params
```python
def try_lasso_hyperparam(X_train, y_train, X_test, y_test, params_to_try,):
    aucs = []
    for param in params_to_try:
        mod = LogisticRegression(penalty='l1', C=param)
        mod.fit(X_train, y_train)
        y_test_preds = mod.predict_proba(X_test)[:,1]
        aucs.append(roc_auc_score(y_test, y_test_preds))
    return aucs
```

Bootstrap coefficients
```python
def bootstrap_ci_coefficients(X_train, y_train, num_bootstraps):
    X_train = X_train.values
    y_train = y_train.values
    bootstrap_estimates = []
    for i in np.arange(num_bootstraps):
        sample_index = np.random.choice(range(0, len(y_train)), len(y_train))
        X_samples = X_train[sample_index]
        y_samples = y_train[sample_index]
        lm = LogisticRegression()
        lm.fit(X_samples, y_samples)
        bootstrap_estimates.append(lm.coef_[0])
    bootstrap_estimates = np.asarray(bootstrap_estimates)
    return bootstrap_estimates
```


### Regularization

Sampling Density, Curse of dimensionality

$N^\frac{1}{D}$  
Where:
* N = number of data points
* D = number of dimensions

### K-folds Cross-Validation

Takes a training set of data and breaks it into 5 folds.  Through 5 iterations, fits a linear model on the *other* folds, then scores how will the model fits on the fold at hand `foldrmse = rmse(y_test_f, test_f_predicted)`. Collects this scores and returns them
```python
def crossVal(X_train, y_train):
    kf = KFold(n_splits=5)
    RMSES = []
    for train_index, test_index in kf.split(X_train):
        X_train_f, X_test_f = X_train[train_index], X_train[test_index]
        y_train_f, y_test_f = y_train[train_index], y_train[test_index]
        linear = LinearRegression()
        linear.fit(X_train_f, y_train_f)
        test_f_predicted = linear.predict(X_test_f)
        foldrmse = rmse(y_test_f, test_f_predicted)
        RMSES.append(float(foldrmse))
    print('The rsme of each fold is {}'.format(RMSES))
    print('The average rmse of each fold is {}'.format(np.mean(RMSES)))
```

### Model selection

Recursive Feature Eliminate RFE.   
At each iteration, select one feature to remove until there are n feature left

```python
from sklearn.feature_selection import RFE
linear_fri = LinearRegression()
selector.fit(X_fri, y_fri)
def gen_modselect_score(n):
    scores = []
    for i in np.arange(n):
        selector = RFE(linear_fri, (21-i), step=1)
        selector.fit(X_fri, y_fri)
        scores.append(selector.score(X_fri, y_fri))
    return scores
```

ROC Curve
```python
from sklearn.metrics import roc_curve, auc
TPR, FPR, thresholds = roc_curve(y_test, y_test_preds, pos_label=None, sample_weight=None, drop_intermediate=True)

def plotroc(TPR, FPR):
    roc_auc = auc(TPR, FPR)
    plt.figure()
    lw = 2
    plt.plot(TPR, FPR, color='darkorange',
             lw=lw, label="ROC curve area = {0:0.4f}".format(roc_auc))
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()
```

### Decision Trees

[Visual Explanation](http://www.r2d3.us/visual-intro-to-machine-learning-part-1/)

How to graphically display
```python
dot_data = tree.export_graphviz(clf, out_file=None,
                         feature_names=iris.feature_names,  
                         class_names=iris.target_names,  
                         filled=True, rounded=True,  
                         special_characters=True)  
graph = graphviz.Source(dot_data)  
graph
```

**Bagging for Regression Trees** --
To apply bagging to regression
trees, we simply construct B regression trees using B bootstrapped training
sets, and average the resulting predictions. These trees are grown deep,
and are not pruned. Hence each individual tree has high variance, but
low bias. Using
B = 100 is sufficient to achieve good performance in this example.

**Bagging for Classification Trees** -- For a given test
observation, we can record the class predicted by each of the B trees, and
take a majority vote: the overall prediction is the most commonly occurring majority
class among the B predictions. Using
B = 100 is sufficient to achieve good performance in this example.

**Out of Bag Error Estimation (OOB)** --

### Random Forest

Random forests provide an improvement over bagged trees by way of a random
small tweak that decorrelates the trees. As in bagging, we build a number forest
of decision trees on bootstrapped training samples. But when building these
decision trees, each time a split in a tree is considered, a random sample of
m predictors is chosen as split candidates from the full set of p predictors.
The split is allowed to use only one of those m predictors.

Ways to interpret feature impact:
* Partial Dependency Plot
* Permute a single feature
* Keep track of information gains due to each features
* Keep track of traffic that passes by each value.

### K-Nearest Neighbors (kNN)

How to choose k?  Could start with sqrt(n)
- better way - use cross validation to find the best value

Point weighting - consider points closer more important to determining

[Plot decision boundary](http://scikit-learn.org/stable/auto_examples/neighbors/plot_classification.html#example-neighbors-plot-classification-py)
___
# <span style="color:red">Visualization</span>

* [flowingdata](http://flowingdata.com/)

## matplotlibx

* Plot in style xkcd: add  `plt.xkcd()` in header
* Plot "fivethirtyeight: `plt.style.use('fivethirtyeight')`
* [Gallery](https://matplotlib.org/gallery.html)
* [Pyplot examples (scroll down)](https://matplotlib.org/gallery/index.html#pyplots-examples)

```python
fig, axes = plt.subplots(3,3, figsize=(10,10))
col_names = ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']

for m, ax in zip(col_names, axes.flatten()):
    ax.scatter(Xpd[m], ypd)
    ax.set_title(m)
```

Plot Violin Plot - see difference in continuous var across levels of categorical variable
```python
def violin_plot_binary(categorical_var, continuous_var, df):
    # Draw a nested violinplot and split the violins for easier comparison
    sns.violinplot(x=categorical_var, y=continuous_var, data=df, split=True,
                   inner="quart")
    sns.despine(left=True)
```

Plotting Two Histograms with Alpha = 0.5
```python
figpois = plt.figure(figsize=(12,6))
acax = figpois.add_subplot(111)
acax.set_title('Histogram, Accidents in a Month')
acax.hist(count_by_month, bins = 15, alpha = 0.5, normed=1, label='Actual')
acax.hist(randpois, bins = 57, alpha = 0.5, color='g', normed=1, label='Poisson')
acax.set_ylabel('Frequency')
acax.legend();
```


Plot Corr heat map
```python
sns.set(style="white")
# Compute the correlation matrix
corr = train_train.corr()
# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))
# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)
# Draw the heatmap with the mask and correct aspect ratio
sns_plot = sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
```


## GGPlot

* [Cheat sheet](https://www.rstudio.com/wp-content/uploads/2015/03/ggplot2-cheatsheet.pdf)

## Seaborn

* [Gallery](https://seaborn.pydata.org/examples/index.html)
___
# <span style="color:red">Data Products</span>





### Markdown

* [Cheatsheet](https://github.com/adam-p/markdown-here/wiki/Markdown-Here-Cheatsheet)
* [Math Symbols](https://reu.dimacs.rutgers.edu/Symbols.pdf)
* [Markdown Table Generator](http://www.tablesgenerator.com/markdown_tables)




### TODOS

* clean up repositories
* Study maximum a posteriori (MAP)
* Networking for jobs
* Switch to sublime text
* Get in touch with Clouse (emailed)
* objects and classes practice
* finish diabetes git hub and blog


### RESOURCES WE SKIMMED THAT I SHOULD COME BACK TO

* [Bayesian Inference for Hackers](http://nbviewer.jupyter.org/github/CamDavidsonPilon/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers/tree/master/)
* Model selection 4_machine_learning/glms - notebook
*
