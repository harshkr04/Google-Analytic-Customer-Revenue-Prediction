# Google-Analytic-Customer-Revenue-Prediction
### Abstraction :
We design and examine an analytics solution to make probability predictions of earning revenue per customer visit.  The motivation for this study is to help marketing teams make better use of their budgets. Specifically, we wish to aid a firm to use their data as a guiding- tool for decision-making. Often the 80/20 rule has been proven right for many firms- A significant portion of their revenue comes from a relatively small percentage of customers. 
Therefore, marketing teams are challenged to make appropriate investments in promotional strategies. We use customer data from Google’s Merchandise Store (G-Store) available on Kaggle along with SAS® University Edition to derive essential insights and generate models to predict the probability of earning revenue per visit.  Our analysis shows that the G-store's revenue earning potential is the highest amongst the customers who visited the store 100-500 times since the mean revenues from this segment are the highest even though the number of such transactions was limited. The features highlighted in our model can be used by Google to increase the revenues from its existing customer base rather than expand its resources to acquire new customers, who may or may not make a substantial purchase per visit on its G-Store.
* DATASET : https://www.kaggle.com/competitions/ga-customer-revenue-prediction/data
### Installation And Requirement
If you do not have Python installed yet, it is highly recommended that you install the Anaconda distribution of Python, which already has the above packages and more included. This project requires the following Python libraries installed:
* sci-kit learn
```bash
conda install scikit-learn
```
* lightgbm
```bash
conda install lightgbm
```
* xgboost
```bash
conda install xgboost
```
* catboost
```bash
conda install catboost
```
* Bayes Optimization
```bash
pip install bayesian-optimization
```
# Code
### Preprocessing
 * Some columns namely `device`,`geoNetworkDomain`,`traficSource`,`totals` in the dataset are in JSON form. We need to convert this JSON columns to tabular format. This is done using a`json_normalize` module.
* Created subcolumns for each JSON columns.
* check number of unique values in each column and `drop` constant columns.
This is done using
```python
pd.nunique(dropna=False) 
```
* Explore the target variable i.e, `totals.transactionRevenue`
### Feature Engineering
* Created new columns (`_day`,`_weekday`,`_month`,`_year`) from `date` feature.
* created `_visitHour` feature from the `visitStartTime` which is given in timestamp format.
### Exploratory Data Analysis

_Date_ and _Time_ v/s_transactionRevenue_
* visalised the change of `transactionRevenue` against `_day`,`_weekday`,`_month`,`_year`.
* visualised the change of `transactionRevenue` against `_visitHour`.

_device_ columns v/s_transactionRevenue_
* visualised the change of `transactionRevenue` against `device.browser`, `device.deviceCategory`, `device.operatingSystem`. 

_geoNetwork_ v/s _transactionRevenue_
* visualised the change of `transactionRevenue` against `geoNetwork.continent`, `geoNetwork.country`, `geoNetwork.subContinent` , `geoNetwork.networkDomain`.

_trafficSource_ v/s _transactionRevenue_
* visualised  the change of`transactionRevenue` against `trafficSource.source`,`trafficSource.referralPath', `trafficSource.medium`.

_totals_ v/s _transactionRevenue_
* visualised  the change of`transactionRevenue` against `totals.pageViews`, `totals.hits`.
## Missing value treatment
```python
pd.fillna(value)
```
* Numerical columns
In numerical columns list only `totals.hits`,`totals.bounces` and `totals.pageViews` were missing, so filled it with appropriate value.

* Categorical columns
some columns had more than 60% of missing value so not able to fill it with existing category. Hence filled it with a new category *'unknown'*.

## Label Encoding
All categorical columns  are encoded using `LabelEncoder()` class which is imported from `sklearn.preprocessing` module.

## Train and Validation split
* train.csv dataset has covered the data from 1st August 2016 to 30th April 2018.
* Last 4 months data i.e, 1st Jan 2018 to 30th April 2018 as validation set.

## Training

* I have used the `LightGBM` model.
* It supports parallel and `GPU` learning. 
* It is highly efficeint in handling large size data.
* To enable GPU access we need to install some drivers
* for more information check [here](https://lightgbm.readthedocs.io/en/latest/GPU-Tutorial.html)




