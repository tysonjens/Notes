
---
## Course
___
## Math/Stats

* [Galvanize Short Course for Stats](https://galvanizeopensource.github.io/stats-shortcourse/)
* [PDF Book: Computer Age Statistical Inference Book](https://web.stanford.edu/~hastie/CASI_files/PDF/casi.pdf)



#### Distributions

* [Distribution Applets](https://homepage.divms.uiowa.edu/~mbognar/)
* [Cheatsheet](https://static1.squarespace.com/static/54bf3241e4b0f0d81bf7ff36/t/55e9494fe4b011aed10e48e5/1441352015658/probability_cheatsheet.pdf)
* [Scipy Stats Functions](https://docs.scipy.org/doc/scipy/reference/stats.html)
* [Scipy Stats](https://stackoverflow.com/questions/37559470/what-do-all-the-distributions-available-in-scipy-stats-look-like#37559471)


thing | Discrete | Continuous | R
------|----------|----------| ---
x to density | pmf | pdf | d
x -> area to left | cdf  | cdf | p
area to left -> x | ppf | ppf | q

###### Continuous

* Normal
  * [Standard Normal Table](https://github.com/gSchool/dsi-probability/blob/master/standard_normal_table.pdf)
* Gamma
* Beta
* Chi-Squared
* Exponential
* F Distribution
* Log-Normal
* t

###### Discrete

* Binomial
* Geometric
* Hypergeometric
* Negative Binomial
* Poisson

#### Bootstrap

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

#### Maximum Likelihood Estimation

#### Experimental Design

#### Hypothesis Testing

* Power - `Pr(Reject H0 | H1 is true)`
  * [Wiki](https://en.wikipedia.org/wiki/Statistical_power#Factors_influencing_power)

A/B Test of two sample proportions (e.g. sign up rate on website)
```python
from z_test import z_test
z_test.z_test(mean1, mean2, n1, n2, effect_size=.01,two_tailed=False)
```

###### Categorical Hypothesis Testing

#### Math_Stats Miscellaneous

* [Log Rules](http://tutorial.math.lamar.edu/Classes/CalcI/DiffExpLogFcns.aspx)
* [Logarithm and Log rules](http://web.mit.edu/kayla/tcom/tcom_handout_logs.pdf)
*

$$
\frac{d}{dx} ln(x) = \frac{1}{x}
$$

* [Maximum Liklihood Estimation](https://ocw.mit.edu/courses/mathematics/18-05-introduction-to-probability-and-statistics-spring-2014/readings/MIT18_05S14_Reading10b.pdf)
* [Maximum Likelihood Estimation (1)](http://statweb.stanford.edu/~susan/courses/s200/lectures/lect11.pdf)
* [Maximum A Posteriori](https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/ppt/22-MAP.pdf)
___

## Coding & Environment

#### GitHub

* [Interactive Tool](http://ndpsoftware.com/git-cheatsheet/previous/git-cheatsheet.html)
* [Adding a Github repo to house local project](https://help.github.com/articles/adding-an-existing-project-to-github-using-the-command-line/)
* Check origin: `git remote -v`

###### Pair Workflow
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



#### SQL

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



#### Python

* [Style Guide](https://www.python.org/dev/peps/pep-0008/)
* [Pythonic Code](http://docs.python-guide.org/en/latest/writing/style/)
* Create a Python 2 environment `conda create -n py2 python=2 anaconda`
* [Classes and Objects Youtube Videos](https://www.youtube.com/watch?v=ZDa-Z5JzLYM&list=PL-osiE80TeTt2d9bfVyTiXJA-UTHn6WwU&index=37)
* `if __name__ == '__main__':`

Categorical Var from Continuous Var
```python
df['new_categ_var'] = pd.cut(df['continuous_var'], [3, 6, 10], labels=['<3','4-6','7-10'])
```


###### psycopg

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






#### Mongodb

#### Unit Testing

`unittest` is a package that you use when you're doing unit testing.  You write the code, then the test code in a separate file. In the test code, one of the files that you load is the actual code that you're testing

`$ python -m unittest test.unit_test_sample`

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



####
___

## Machine learning

___
## Visualization

#### matplotlib

* Plot in style xkcd: add  `plt.xkcd()` in header
* Plot "fivethirtyeight: `plt.style.use('fivethirtyeight')`
___
## Data Products

#### Markdown

* [Cheatsheet](https://github.com/adam-p/markdown-here/wiki/Markdown-Here-Cheatsheet)
* [Math Symbols](https://reu.dimacs.rutgers.edu/Symbols.pdf)


#### Matplotlib

* [Gallery](https://matplotlib.org/gallery.html)
* [Pyplot examples (scroll down)](https://matplotlib.org/gallery/index.html#pyplots-examples)

#### Seaborn

* [Gallery](https://seaborn.pydata.org/examples/index.html)








     For postgres, we use psycopg2
    for mySQL: Connector/Python
    Connections must be established using an existing database, username, database IP/URL, and maybe passwords
    If you have no created databases, you can connect to Postgres using the dbname 'postgres' to initialize db commands
    Data changes are not actually stored until you choose to commit. This can be done either through conn.commit() or setting autocommit = True. Until committed, all transactions is only temporary stored.
    Autocommit = True is necessary to do database commands like CREATE DATABASE. This is because Postgres does not have temporary transactions at the database level.
    If you ever need to build similar pipelines for other forms of database, there are libraries such PyODBC which operate very similarly.
    SQL connection databases utilizes cursors for data traversal and retrieval. This is kind of like an iterator in Python.
    Cursor operations typically goes like the following:
    execute a query
    fetch rows from query result if it is a SELECT query
    because it is iterative, previously fetched rows can only be fetched again by rerunning the query
    close cursor through .close()
    Cursors and Connections must be closed using .close() or else Postgres will lock certain operation on the database/tables to connection is severed.
DAY 4 - Mongo DB
  MORNING EXERCISE
    db.log.find({'t': {$exists: 1}}).forEach(function(entry) { entry.t = new Date(entry.t * 1000); db.log.save(entry); })
  resources
    Aggregation:  https://docs.mongodb.com/manual/reference/sql-aggregation-comparison/
    db.articles.aggregate([{$group:{_id:null, average: {$avg: "$word_count"}}}])

DAY 5 -- pandas
  STUDY
    numpy for numbers, pandas for heterogenous data
    two workhorses of pandas - Series and dataframes
    Wes Mckinney - Time series https://www.youtube.com/watch?v=0unf-C-pBYE
    MORNING - did numpy and pandas practice with individual assignments
    AFTERNOON -
DAY 6 -- matplotlib
  Three main types of variables
    Quantitative
    Categorical
    Ordinal
  Scatter Plot
    def scatterplot(x_data, y_data, x_label, y_label, title):

    # Create the plot object
    , ax = plt.subplots()

    # Plot the data, set the size (s), color and transparency (alpha)
    # of the points
    ax.scatter(x_data, y_data, s = 30, color = '#539caf', alpha = 0.75)

    # Label the axes and provide a title
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
  Line Plot with Confidence Intervals
    # Define a function for a plot with two y axes
    def lineplot2y(x_data, x_label, y1_data, y1_color, y1_label, y2_data, y2_color, y2_label, title):
    # Each variable will actually have its own plot object but they
    # will be displayed in just one plot
    # Create the first plot object and draw the line
    , ax1 = plt.subplots()
    ax1.plot(x_data, y1_data, color = y1_color)
    # Label axes
    ax1.set_ylabel(y1_label, color = y1_color)
    ax1.set_xlabel(x_label)
    ax1.set_title(title)

    # Create the second plot object, telling matplotlib that the two
    # objects have the same x-axis
    ax2 = ax1.twinx()
    ax2.plot(x_data, y2_data, color = y2_color)
    ax2.set_ylabel(y2_label, color = y2_color)
    # Show right frame line
    ax2.spines['right'].set_visible(True)
  Histogram
    # Define a function for a histogram
    def histogram(data, x_label, y_label, title):
      _, ax = plt.subplots()
      ax.hist(data, color = '#539caf')
      ax.set_ylabel(y_label)
      ax.set_xlabel(x_label)
      ax.set_title(title)

    # Call the function to create plot
    histogram(data = daily_data['registered']
             , x_label = 'Check outs'
             , y_label = 'Frequency'
             , title = 'Distribution of Registered Check Outs')




To dos
  Apply to the the UM Gupta hacks thing
  Make plans for:
    Project - getting data -- read Ruan's Stuff
    Networking for jobs
  Study maximum a posteriori (MAP)
  Practice enumerate and a few list comprehension examples
  Get a few matplotlib examples from exercises
