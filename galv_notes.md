

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


thing | Discrete | Continuous | R
------|----------|----------| ---
x to density | pmf | pdf | d
x -> area to left | cdf  | cdf | p
area to left -> x | ppf | ppf | q


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
* `if __name__ == '__main__':`

Categorical Var from Continuous Var
```python
df['new_categ_var'] = pd.cut(df['continuous_var'], [3, 6, 10], labels=['<3','4-6','7-10'])
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

[An Introduction to Statistical Learning - ISLR](http://www-bcf.usc.edu/~gareth/ISL/ISLR%20Seventh%20Printing.pdf)

___
# <span style="color:red">Visualization</span>

* [flowingdata](http://flowingdata.com/)

## matplotlibx

* Plot in style xkcd: add  `plt.xkcd()` in header
* Plot "fivethirtyeight: `plt.style.use('fivethirtyeight')`
* [Gallery](https://matplotlib.org/gallery.html)
* [Pyplot examples (scroll down)](https://matplotlib.org/gallery/index.html#pyplots-examples)

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

## GGPlot

* [Cheat sheet](https://www.rstudio.com/wp-content/uploads/2015/03/ggplot2-cheatsheet.pdf)

## Seaborn

* [Gallery](https://seaborn.pydata.org/examples/index.html)
___
# <span style="color:red">Data Products</span>

## Data Cleaning

Scaling
```python
from sklearn.model_selection import train_test_split
## make a train test split
X_train, X_test, y_train, y_test = train_test_split(X, y)

## scale using sklearn
scaler = preprocessing.StandardScaler().fit(X_train)
X_train_1 = scaler.transform(X_train)
X_test_1 = scaler.transform(X_test)
```

Dummies
```Python
## in pandas
df = pd.DataFrame({'country': ['usa', 'canada', 'australia','japan','germany']})
pd.get_dummies(df,prefix=['country'])
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



### Markdown

* [Cheatsheet](https://github.com/adam-p/markdown-here/wiki/Markdown-Here-Cheatsheet)
* [Math Symbols](https://reu.dimacs.rutgers.edu/Symbols.pdf)




### TODOS

* [15] clean up repositories
* Study maximum a posteriori (MAP)
* Find a Kaggle data set to play with.
* Try to find a few data scientists working on operations
* Networking for jobs
* Switch to sublime text
* Get in touch with Clouse (emailed)
* try iterm2
* look at amelia's notes

### RESOURCES WE SKIMMED THAT I SHOULD COME BACK TO

* [Bayesian Inference for Hackers](http://nbviewer.jupyter.org/github/CamDavidsonPilon/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers/tree/master/)
