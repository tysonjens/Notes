
---
## Course
___
## Math/Stats

* [Galvanize Short Course for Stats](https://galvanizeopensource.github.io/stats-shortcourse/)
* [Computer Age Statistical Inference Book](https://web.stanford.edu/~hastie/CASI_files/PDF/casi.pdf)



#### Distributions

* [Cheatsheet](https://static1.squarespace.com/static/54bf3241e4b0f0d81bf7ff36/t/55e9494fe4b011aed10e48e5/1441352015658/probability_cheatsheet.pdf)
* [Scipy Stats Functions](https://docs.scipy.org/doc/scipy/reference/stats.html)

###### Normal

* [Standard Normal Table](https://github.com/gSchool/dsi-probability/blob/master/standard_normal_table.pdf)

###### Rayleigh

###### Binomial

#### Bootstrap

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

#### Hypothesis Testing

#### Math_Stats Miscellaneous

* [Log Rules](http://tutorial.math.lamar.edu/Classes/CalcI/DiffExpLogFcns.aspx)

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


#### Python

* [Style Guide](https://www.python.org/dev/peps/pep-0008/)
* [Pythonic Code](http://docs.python-guide.org/en/latest/writing/style/)
* Create a Python 2 environment `conda create -n py2 python=2 anaconda`
* [Classes and Objects Youtube Videos](https://www.youtube.com/watch?v=ZDa-Z5JzLYM&list=PL-osiE80TeTt2d9bfVyTiXJA-UTHn6WwU&index=37)
* `if __name__ == '__main__':`


###### psycopg

* [Browse Psycopg documentation](http://initd.org/psycopg/docs/)





#### Mongodb

#### Unit Testing

* [Tutorial](https://jeffknupp.com/blog/2013/12/09/improve-your-python-understanding-unit-testing/)


Text goes underneath

###### A Table
name | link
-----|------
thing1 | thing 2

####
___

## Machine learning

___
## Visualization
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




DAY 1 - Command Line, Github, Unit testing.

  COMMAND LINE DICTIONARY
    grep -i string file -- takes string and returns rows from file
    sort
    python python_script -- runs a python python_script
    rm -- removes a file
    rmdir -- removes an empty directory
  NOTE -
    Git and github - play around and don't be afraid of breaking things - it's a version control tool, so it's meant to protect from screwing things up too bad.
    Keep branches for a specific period of time, then merge it in to master and then delete.
  GIT COMMANDS
    git remote -v
    Interactive tool
      http://ndpsoftware.com/git-cheatsheet/previous/git-cheatsheet.html
  NOTE
    so far so good, keeping up with all my assignments so far.  Doing a little awk study at lunch
  UNIT TESTING - writing code (separate from your actual application code) that invokes the code it tests to help determine if there are any errors.
    You write a function to test the code you're writing.  This is a key concept for DS.  The code below opens python, specifies that it wants to open module, and runs unittest using test.unit_test_sample

    $ python -m unittest test.unit_test_sample

    EXAMPLE CODE
      import unittest
      from primes import is_prime

      class PrimesTestCase(unittest.TestCase):
          """Tests for `primes.py`."""

          def test_is_five_prime(self):
              """Is five successfully determined to be prime?"""
              self.assertTrue(is_prime(5))

      if __name__ == '__main__':
          unittest.main()
    RESOURCES
      https://jeffknupp.com/blog/2013/12/09/improve-your-python-understanding-unit-testing/
      See example files is ds-day-1
    NOTE
      unittest is a package that you use when you're doing unittest.  You write the code, then the test code in a separate file. In the test code, one of the files that you load is the actual code that you're testing
  NOTE -
    had a happy hour in the afternoon.  Met John, Doster, Marshall (walking back to the trains), Chris Fuller (Nik's Friend), Elliot, Damien, and few guys from the BP program (Anu)
DAY 2 - Pair Programming, OOP
  PRE-READING
    GIT -
    PAIR PROGRAMMING
    WRITING PYTHONIC CODE
      http://docs.python-guide.org/en/latest/writing/style/
    PYTHON STYLE GUIDE
      https://www.python.org/dev/peps/pep-0008/
  PRE-CLASS NOTE
    came in this morning and practiced some of the code for the unit test. Used the two files from yesterday to craete my own addition and exponential test, then successfully ran the unittest module in command line. Boom.
  LECTURE
    How to create a Python 2 environment: conda create -n py2 python=2 anaconda
    Goal is to be able to write a script, and run from command LINE
      $ python script.py datafile.csv
  LECTURE PM
    Learned about OOP, classes, objects (instances of classes), functions.  Learned that classes can have attributes and methods.  Attributes are like nouns, or data that is stored about the object, and methods are verbs that the object can do.  Spent the afternoon in a pair programming assignment updating a game of war, and tonight I'll need to start in on a game of blackjack.  I'm learning all the things!!
  LECTURE LATE PM
    Tried really hard to get the blackjack game to work.  I think I was fairly close but ultimately didn't finish. checking out solutions.
  RESOURCES
    Videos from Chris
      https://www.youtube.com/watch?v=ZDa-Z5JzLYM&list=PL-osiE80TeTt2d9bfVyTiXJA-UTHn6WwU&index=37
DAY 3 - SQL, PostgreSQL,  all in folder dsi-sql
  PRE-READING
  LECTURE AM
    learning about HAVING.  Can be used on aggregated rows after ORDER BY
    order of operations - SELECT FROM JOIN WHERE GROUPBY HAVING ORDERBY LIMIT;
  NOTE -
    felt really good about the morning.  Was working well ahead of others and learn quite a bit about SQL
  LECTURE PM -
    how to build a pipeline.  There's a living, breathing database and we want to create a snapshot of it each day to run programs etc.  This is called
    jupyter-notebook
    how to connect python to RMDBS.  For postgres, we use psycopg2
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
WEEKEND STUDY PLAN
  [15] Review Unit Testing - find a few resources and store above
  [15] Find a pair programming workflow - store above
  [30] finish the sqlzoo tutorial
  [15] watch the next OOP video from
  [30] write and practice with a class
  45 try to write a python script that opens colo-dwrpt01 and gets data
  [30] Mongodb thing - do the examples.md file in ds-mongo
  30 study pandas
  30 update blog and send out to team
  30 make a networking plan for break WEEK
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
    _, ax = plt.subplots()

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
    _, ax1 = plt.subplots()
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





TOPICS TO STUDY
  Apply to the the UM Gupta hacks thing
  Make plans for:
    Project - getting data -- read Ruan's Stuff
    Networking for jobs
  Study maximum a posteriori (MAP)
  Study exponential distribution parameter
  Bootstrap notes



MATH
  Logarithm and Log rules
    http://web.mit.edu/kayla/tcom/tcom_handout_logs.pdf

WRITING GOOD CODE
  Unit TEST
    http://pythontesting.net/framework/unittest/unittest-introduction/

DATA VISUALIZATION
  https://www.datascience.com/blog/learn-data-science-intro-to-data-visualization-in-matplotlib
  https://seaborn.pydata.org/examples/index.html
