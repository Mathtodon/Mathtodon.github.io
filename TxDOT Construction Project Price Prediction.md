
# TxDOT Project


```python
import os
import math
import numpy as np
import pandas as pd
%matplotlib inline
import matplotlib.pyplot as plt

import csv
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn import feature_selection, linear_model
```

## Load in the Data


```python
df = pd.read_csv('data/Bid_Data.csv')
len(df)
```

    32447



## Formatting & Identifying the Data


```python
df = df[df['Rank'] == 1]
df = df[df['Type'] == 'Construction']

df = df.drop(' From', 1)
df = df.drop('To', 1)
df = df.drop('Contract Description', 1)
df = df.drop('Contractor', 1)
df = df.drop('Contract Category', 1)
df = df.drop('LNG MON', 1)
df = df.drop('MONTH', 1)

df['Award Amount'] = df['Award Amount'].str.lstrip('$')
df['Engineers Estimate'] = df['Engineers Estimate'].str.lstrip('$')
df['Award Amount'] = df['Award Amount'].str.replace(',','').astype(float)
df['Engineers Estimate'] = df['Engineers Estimate'].str.replace(',','').astype(float)

#Renaming Variables
df['EngEst'] = df['Engineers Estimate']
df['NBidders'] = df['Number of Bidders']
df['Date'] = pd.to_datetime(df['Letting Date'])
df.set_index('Date' , inplace=True)
df['Year'] = df.index.year
df['Month'] = df.index.month
df['WinBid'] = df['Award Amount']

# Creating New Varialbes
df['Diff'] = df['EngEst'] - df['WinBid']
df['lnWinBid'] = np.log(df['WinBid'])
df['lnEngEst'] = np.log(df['EngEst'])
df['DiffLn'] = df['lnWinBid'] - df['lnEngEst']
df['Within10Percent'] = 1 
df['PercentOff'] = df['Diff'] / df['EngEst']
df['MoreOrLessThan10'] = 0
df['LessThan10'] = 0
df['MoreThan10'] = 0

df.loc[(df.PercentOff > .10) , 'Within10Percent'] = 0
df.loc[(df.PercentOff < -.10) , 'Within10Percent'] = 0
df.loc[(df.PercentOff > .10) , 'MoreOrLessThan10'] = 1
df.loc[(df.PercentOff < -.10) , 'MoreOrLessThan10'] = 2
df.loc[(df.PercentOff > .10) , 'MoreThan10'] = 1
df.loc[(df.PercentOff < -.10) , 'LessThan10'] = 1

print(len(df))
```

    5177
    


```python
df
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Type</th>
      <th>CCSJ</th>
      <th>Letting Call Number</th>
      <th>Project</th>
      <th>Length</th>
      <th>Highway</th>
      <th>Letting Date</th>
      <th>District</th>
      <th>County</th>
      <th>Number of Bidders</th>
      <th>...</th>
      <th>WinBid</th>
      <th>Diff</th>
      <th>lnWinBid</th>
      <th>lnEngEst</th>
      <th>DiffLn</th>
      <th>Within10Percent</th>
      <th>PercentOff</th>
      <th>MoreOrLessThan10</th>
      <th>LessThan10</th>
      <th>MoreThan10</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2010-10-21</th>
      <td>Construction</td>
      <td>0001-05-016</td>
      <td>10103012</td>
      <td>STP 2009(816)ES</td>
      <td>0.200</td>
      <td>FM 259</td>
      <td>10/21/2010</td>
      <td>El Paso</td>
      <td>El Paso</td>
      <td>1</td>
      <td>...</td>
      <td>116285.82</td>
      <td>7730.08</td>
      <td>11.663806</td>
      <td>11.728165</td>
      <td>-0.064359</td>
      <td>1</td>
      <td>0.062331</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2010-08-11</th>
      <td>Construction</td>
      <td>0002-01-083</td>
      <td>8103285</td>
      <td>STP 2011(321)</td>
      <td>9.050</td>
      <td>SH 20</td>
      <td>8/11/2010</td>
      <td>El Paso</td>
      <td>El Paso</td>
      <td>6</td>
      <td>...</td>
      <td>2477913.10</td>
      <td>1365614.90</td>
      <td>14.722927</td>
      <td>15.161901</td>
      <td>-0.438974</td>
      <td>0</td>
      <td>0.355302</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2010-04-07</th>
      <td>Construction</td>
      <td>0002-08-050</td>
      <td>4103999</td>
      <td>C 2-8-50</td>
      <td>1.250</td>
      <td>IH 10</td>
      <td>4/7/2010</td>
      <td>El Paso</td>
      <td>Hudspeth</td>
      <td>1</td>
      <td>...</td>
      <td>228662.50</td>
      <td>37724.88</td>
      <td>12.340002</td>
      <td>12.492707</td>
      <td>-0.152704</td>
      <td>0</td>
      <td>0.141617</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2010-08-11</th>
      <td>Construction</td>
      <td>0002-11-056</td>
      <td>8103286</td>
      <td>IM 0101(256)</td>
      <td>4.316</td>
      <td>IH 10</td>
      <td>8/11/2010</td>
      <td>El Paso</td>
      <td>Culberson</td>
      <td>2</td>
      <td>...</td>
      <td>1094961.15</td>
      <td>492577.51</td>
      <td>13.906229</td>
      <td>14.277695</td>
      <td>-0.371466</td>
      <td>0</td>
      <td>0.310277</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2010-05-12</th>
      <td>Construction</td>
      <td>0002-14-033</td>
      <td>5103213</td>
      <td>STP 2010(756)ES</td>
      <td>0.100</td>
      <td>FM 258</td>
      <td>5/12/2010</td>
      <td>El Paso</td>
      <td>El Paso</td>
      <td>4</td>
      <td>...</td>
      <td>149069.50</td>
      <td>109807.00</td>
      <td>11.912168</td>
      <td>12.464106</td>
      <td>-0.551938</td>
      <td>0</td>
      <td>0.424168</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2010-02-09</th>
      <td>Construction</td>
      <td>0003-01-051</td>
      <td>2103035</td>
      <td>IM 0101(254)</td>
      <td>1.069</td>
      <td>IH 10</td>
      <td>2/9/2010</td>
      <td>El Paso</td>
      <td>Culberson</td>
      <td>2</td>
      <td>...</td>
      <td>409447.31</td>
      <td>-66821.41</td>
      <td>12.922564</td>
      <td>12.744394</td>
      <td>0.178169</td>
      <td>0</td>
      <td>-0.195027</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2010-05-12</th>
      <td>Construction</td>
      <td>0003-05-046</td>
      <td>5103208</td>
      <td>IM 0201(176)</td>
      <td>3.806</td>
      <td>IH 20</td>
      <td>5/12/2010</td>
      <td>Odessa</td>
      <td>Reeves</td>
      <td>4</td>
      <td>...</td>
      <td>2284464.17</td>
      <td>550755.83</td>
      <td>14.641642</td>
      <td>14.857630</td>
      <td>-0.215988</td>
      <td>0</td>
      <td>0.194255</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2010-02-10</th>
      <td>Construction</td>
      <td>0003-05-048</td>
      <td>2103209</td>
      <td>IM 0201(175)</td>
      <td>57.050</td>
      <td>IH 20</td>
      <td>2/10/2010</td>
      <td>Odessa</td>
      <td>Reeves</td>
      <td>3</td>
      <td>...</td>
      <td>3460027.14</td>
      <td>193621.95</td>
      <td>15.056787</td>
      <td>15.111237</td>
      <td>-0.054450</td>
      <td>1</td>
      <td>0.052994</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2010-08-11</th>
      <td>Construction</td>
      <td>0004-04-076</td>
      <td>8103305</td>
      <td>STP 2000(400)TE</td>
      <td>1.181</td>
      <td>IH 20</td>
      <td>8/11/2010</td>
      <td>Odessa</td>
      <td>Ward</td>
      <td>8</td>
      <td>...</td>
      <td>14473728.60</td>
      <td>-1989505.68</td>
      <td>16.487846</td>
      <td>16.339976</td>
      <td>0.147870</td>
      <td>0</td>
      <td>-0.159362</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2010-12-08</th>
      <td>Construction</td>
      <td>0005-05-105</td>
      <td>12103231</td>
      <td>IM 0202(231)</td>
      <td>78.025</td>
      <td>IH 20</td>
      <td>12/8/2010</td>
      <td>Abilene</td>
      <td>Howard</td>
      <td>5</td>
      <td>...</td>
      <td>2555217.75</td>
      <td>66828.32</td>
      <td>14.753648</td>
      <td>14.779466</td>
      <td>-0.025818</td>
      <td>1</td>
      <td>0.025487</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2010-10-22</th>
      <td>Construction</td>
      <td>0005-07-049</td>
      <td>10103227</td>
      <td>STP 2011(623)ES</td>
      <td>18.838</td>
      <td>IH 20</td>
      <td>10/22/2010</td>
      <td>Abilene</td>
      <td>Mitchell</td>
      <td>6</td>
      <td>...</td>
      <td>7827281.73</td>
      <td>1083350.06</td>
      <td>15.873126</td>
      <td>16.002756</td>
      <td>-0.129630</td>
      <td>0</td>
      <td>0.121579</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2010-03-09</th>
      <td>Construction</td>
      <td>0005-08-097</td>
      <td>3103012</td>
      <td>STP 2010(601)HES</td>
      <td>0.001</td>
      <td>IH 20</td>
      <td>3/9/2010</td>
      <td>Abilene</td>
      <td>Mitchell</td>
      <td>4</td>
      <td>...</td>
      <td>85238.00</td>
      <td>51205.42</td>
      <td>11.353203</td>
      <td>11.823665</td>
      <td>-0.470463</td>
      <td>0</td>
      <td>0.375287</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2010-08-10</th>
      <td>Construction</td>
      <td>0005-13-053</td>
      <td>8103084</td>
      <td>IM 0201(177)</td>
      <td>9.164</td>
      <td>IH 20</td>
      <td>8/10/2010</td>
      <td>Odessa</td>
      <td>Ector</td>
      <td>1</td>
      <td>...</td>
      <td>6438932.93</td>
      <td>-639101.05</td>
      <td>15.677873</td>
      <td>15.573339</td>
      <td>0.104534</td>
      <td>0</td>
      <td>-0.110193</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2010-04-06</th>
      <td>Construction</td>
      <td>0006-01-091</td>
      <td>4103008</td>
      <td>IM 0202(230)</td>
      <td>9.040</td>
      <td>IH 20</td>
      <td>4/6/2010</td>
      <td>Abilene</td>
      <td>Mitchell</td>
      <td>6</td>
      <td>...</td>
      <td>3272602.41</td>
      <td>-254919.33</td>
      <td>15.001096</td>
      <td>14.920000</td>
      <td>0.081096</td>
      <td>1</td>
      <td>-0.084475</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2010-08-11</th>
      <td>Construction</td>
      <td>0006-02-104</td>
      <td>8103239</td>
      <td>C 6-2-104</td>
      <td>8.284</td>
      <td>IH 20</td>
      <td>8/11/2010</td>
      <td>Abilene</td>
      <td>Nolan</td>
      <td>6</td>
      <td>...</td>
      <td>1038393.49</td>
      <td>82349.54</td>
      <td>13.853185</td>
      <td>13.929502</td>
      <td>-0.076317</td>
      <td>1</td>
      <td>0.073478</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2010-08-10</th>
      <td>Construction</td>
      <td>0006-02-106</td>
      <td>8103085</td>
      <td>STP 2011(292)HES</td>
      <td>9.889</td>
      <td>IH 20</td>
      <td>8/10/2010</td>
      <td>Abilene</td>
      <td>Nolan</td>
      <td>8</td>
      <td>...</td>
      <td>589049.95</td>
      <td>92165.21</td>
      <td>13.286266</td>
      <td>13.431633</td>
      <td>-0.145367</td>
      <td>0</td>
      <td>0.135295</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2010-01-05</th>
      <td>Construction</td>
      <td>0006-04-068</td>
      <td>1103023</td>
      <td>STP 2010(233)SB</td>
      <td>17.488</td>
      <td>IH 20</td>
      <td>1/5/2010</td>
      <td>Abilene</td>
      <td>Taylor</td>
      <td>14</td>
      <td>...</td>
      <td>1191750.29</td>
      <td>126077.35</td>
      <td>13.990934</td>
      <td>14.091495</td>
      <td>-0.100562</td>
      <td>1</td>
      <td>0.095671</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2010-04-07</th>
      <td>Construction</td>
      <td>0006-05-108</td>
      <td>4103213</td>
      <td>STP 2010(559)ES</td>
      <td>2.986</td>
      <td>IH 20</td>
      <td>4/7/2010</td>
      <td>Abilene</td>
      <td>Taylor</td>
      <td>3</td>
      <td>...</td>
      <td>548674.30</td>
      <td>36426.05</td>
      <td>13.215260</td>
      <td>13.279539</td>
      <td>-0.064278</td>
      <td>1</td>
      <td>0.062256</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2010-05-12</th>
      <td>Construction</td>
      <td>0006-18-052</td>
      <td>5103221</td>
      <td>STP 2010(845)</td>
      <td>3.283</td>
      <td>BI 20-R</td>
      <td>5/12/2010</td>
      <td>Abilene</td>
      <td>Taylor</td>
      <td>5</td>
      <td>...</td>
      <td>595110.27</td>
      <td>84426.13</td>
      <td>13.296502</td>
      <td>13.429166</td>
      <td>-0.132664</td>
      <td>0</td>
      <td>0.124241</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2010-03-10</th>
      <td>Construction</td>
      <td>0007-02-047</td>
      <td>3103218</td>
      <td>STP 2010(602)ES</td>
      <td>8.914</td>
      <td>IH 20</td>
      <td>3/10/2010</td>
      <td>Abilene</td>
      <td>Callahan</td>
      <td>7</td>
      <td>...</td>
      <td>2294673.03</td>
      <td>301437.12</td>
      <td>14.646101</td>
      <td>14.769525</td>
      <td>-0.123424</td>
      <td>0</td>
      <td>0.116111</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2010-01-05</th>
      <td>Construction</td>
      <td>0007-03-083</td>
      <td>1103006</td>
      <td>IM 0203(077)</td>
      <td>308.483</td>
      <td>IH 20</td>
      <td>1/5/2010</td>
      <td>Brownwood</td>
      <td>Eastland</td>
      <td>4</td>
      <td>...</td>
      <td>6349325.82</td>
      <td>408080.75</td>
      <td>15.663859</td>
      <td>15.726150</td>
      <td>-0.062291</td>
      <td>1</td>
      <td>0.060390</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2010-02-09</th>
      <td>Construction</td>
      <td>0007-04-109</td>
      <td>2103024</td>
      <td>STP 2010(322)HES</td>
      <td>7.724</td>
      <td>IH 20</td>
      <td>2/9/2010</td>
      <td>Brownwood</td>
      <td>Eastland</td>
      <td>13</td>
      <td>...</td>
      <td>515542.26</td>
      <td>393293.65</td>
      <td>13.152975</td>
      <td>13.719920</td>
      <td>-0.566945</td>
      <td>0</td>
      <td>0.432744</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2010-03-10</th>
      <td>Construction</td>
      <td>0007-06-076</td>
      <td>3103217</td>
      <td>IM 0203(078)</td>
      <td>5.594</td>
      <td>IH 20</td>
      <td>3/10/2010</td>
      <td>Brownwood</td>
      <td>Eastland</td>
      <td>10</td>
      <td>...</td>
      <td>2803289.11</td>
      <td>120462.21</td>
      <td>14.846304</td>
      <td>14.888378</td>
      <td>-0.042074</td>
      <td>1</td>
      <td>0.041201</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2010-06-03</th>
      <td>Construction</td>
      <td>0007-06-081</td>
      <td>6103009</td>
      <td>IM 0203(079)</td>
      <td>2.244</td>
      <td>IH 20</td>
      <td>6/3/2010</td>
      <td>Brownwood</td>
      <td>Eastland</td>
      <td>4</td>
      <td>...</td>
      <td>1554812.14</td>
      <td>-141855.95</td>
      <td>14.256865</td>
      <td>14.161195</td>
      <td>0.095671</td>
      <td>0</td>
      <td>-0.100397</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2010-08-10</th>
      <td>Construction</td>
      <td>0007-10-054</td>
      <td>8103032</td>
      <td>STP 2011(203)</td>
      <td>7.380</td>
      <td>US 180</td>
      <td>8/10/2010</td>
      <td>Fort Worth</td>
      <td>Palo Pinto</td>
      <td>9</td>
      <td>...</td>
      <td>1517352.16</td>
      <td>296081.24</td>
      <td>14.232477</td>
      <td>14.410733</td>
      <td>-0.178255</td>
      <td>0</td>
      <td>0.163271</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2010-10-21</th>
      <td>Construction</td>
      <td>0008-02-070</td>
      <td>10103013</td>
      <td>STP 2011(472)</td>
      <td>179.385</td>
      <td>US 180</td>
      <td>10/21/2010</td>
      <td>Fort Worth</td>
      <td>Parker</td>
      <td>5</td>
      <td>...</td>
      <td>7357575.84</td>
      <td>1156674.43</td>
      <td>15.811241</td>
      <td>15.957252</td>
      <td>-0.146011</td>
      <td>0</td>
      <td>0.135852</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2010-04-06</th>
      <td>Construction</td>
      <td>0008-03-098</td>
      <td>4103032</td>
      <td>IM 0204(271)</td>
      <td>5.170</td>
      <td>IH 20</td>
      <td>4/6/2010</td>
      <td>Fort Worth</td>
      <td>Parker</td>
      <td>6</td>
      <td>...</td>
      <td>1730019.68</td>
      <td>140955.22</td>
      <td>14.363643</td>
      <td>14.441970</td>
      <td>-0.078327</td>
      <td>1</td>
      <td>0.075338</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2010-08-10</th>
      <td>Construction</td>
      <td>0008-06-048</td>
      <td>8103035</td>
      <td>BR 2011(098)</td>
      <td>0.169</td>
      <td>SH 180</td>
      <td>8/10/2010</td>
      <td>Fort Worth</td>
      <td>Tarrant</td>
      <td>5</td>
      <td>...</td>
      <td>1447722.05</td>
      <td>48917.86</td>
      <td>14.185502</td>
      <td>14.218733</td>
      <td>-0.033231</td>
      <td>1</td>
      <td>0.032685</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2010-12-07</th>
      <td>Construction</td>
      <td>0008-13-223</td>
      <td>12103027</td>
      <td>IM 8204(276)</td>
      <td>2.341</td>
      <td>IH 820</td>
      <td>12/7/2010</td>
      <td>Fort Worth</td>
      <td>Tarrant</td>
      <td>4</td>
      <td>...</td>
      <td>1243621.79</td>
      <td>317799.96</td>
      <td>14.033538</td>
      <td>14.261107</td>
      <td>-0.227569</td>
      <td>0</td>
      <td>0.203532</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2010-09-08</th>
      <td>Construction</td>
      <td>0008-15-044</td>
      <td>9103021</td>
      <td>IM 8204(273)</td>
      <td>0.100</td>
      <td>IH 820</td>
      <td>9/8/2010</td>
      <td>Fort Worth</td>
      <td>Tarrant</td>
      <td>5</td>
      <td>...</td>
      <td>96000.00</td>
      <td>31793.18</td>
      <td>11.472103</td>
      <td>11.758168</td>
      <td>-0.286065</td>
      <td>0</td>
      <td>0.248786</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2016-03-03</th>
      <td>Construction</td>
      <td>1186-01-089</td>
      <td>3163028</td>
      <td>STP 2016(636)</td>
      <td>4.103</td>
      <td>FM 969</td>
      <td>3/3/2016</td>
      <td>Austin</td>
      <td>Travis</td>
      <td>6</td>
      <td>...</td>
      <td>2067551.68</td>
      <td>284564.36</td>
      <td>14.541876</td>
      <td>14.670826</td>
      <td>-0.128950</td>
      <td>0</td>
      <td>0.120982</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2016-03-04</th>
      <td>Construction</td>
      <td>1259-01-041</td>
      <td>3163228</td>
      <td>STP 2016(812)HES</td>
      <td>1.336</td>
      <td>FM 1097</td>
      <td>3/4/2016</td>
      <td>Houston</td>
      <td>Montgomery</td>
      <td>3</td>
      <td>...</td>
      <td>238826.40</td>
      <td>-41172.80</td>
      <td>12.383492</td>
      <td>12.194271</td>
      <td>0.189221</td>
      <td>0</td>
      <td>-0.208308</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2016-02-10</th>
      <td>Construction</td>
      <td>1277-01-015</td>
      <td>2163228</td>
      <td>BR 2016(411)</td>
      <td>0.318</td>
      <td>FM 1012</td>
      <td>2/10/2016</td>
      <td>Beaumont</td>
      <td>Newton</td>
      <td>3</td>
      <td>...</td>
      <td>1777740.74</td>
      <td>-130944.23</td>
      <td>14.390854</td>
      <td>14.314342</td>
      <td>0.076511</td>
      <td>1</td>
      <td>-0.079515</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2016-04-06</th>
      <td>Construction</td>
      <td>1289-01-029</td>
      <td>4163237</td>
      <td>BR 1602(046)</td>
      <td>3.551</td>
      <td>FM 1126</td>
      <td>4/6/2016</td>
      <td>Dallas</td>
      <td>Navarro</td>
      <td>8</td>
      <td>...</td>
      <td>6714487.74</td>
      <td>268564.38</td>
      <td>15.719778</td>
      <td>15.758997</td>
      <td>-0.039219</td>
      <td>1</td>
      <td>0.038459</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2016-01-06</th>
      <td>Construction</td>
      <td>1397-01-031</td>
      <td>1163240</td>
      <td>C 1397-1-31</td>
      <td>3.094</td>
      <td>FM 1836</td>
      <td>1/6/2016</td>
      <td>Dallas</td>
      <td>Kaufman</td>
      <td>5</td>
      <td>...</td>
      <td>5943147.30</td>
      <td>-906869.64</td>
      <td>15.597749</td>
      <td>15.432178</td>
      <td>0.165572</td>
      <td>0</td>
      <td>-0.180067</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2016-01-06</th>
      <td>Construction</td>
      <td>1402-03-011</td>
      <td>1163242</td>
      <td>STP 2016(732)</td>
      <td>20.298</td>
      <td>FM 1375</td>
      <td>1/6/2016</td>
      <td>Houston</td>
      <td>Montgomery</td>
      <td>4</td>
      <td>...</td>
      <td>1297509.45</td>
      <td>352076.70</td>
      <td>14.075957</td>
      <td>14.316035</td>
      <td>-0.240078</td>
      <td>0</td>
      <td>0.213433</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2016-04-06</th>
      <td>Construction</td>
      <td>1494-03-001</td>
      <td>4163222</td>
      <td>C 1494-3-1</td>
      <td>0.018</td>
      <td>FM 3486</td>
      <td>4/6/2016</td>
      <td>Dallas</td>
      <td>Kaufman</td>
      <td>5</td>
      <td>...</td>
      <td>4849821.37</td>
      <td>-215531.22</td>
      <td>15.394452</td>
      <td>15.348994</td>
      <td>0.045459</td>
      <td>1</td>
      <td>-0.046508</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2016-02-10</th>
      <td>Construction</td>
      <td>1526-03-015</td>
      <td>2163233</td>
      <td>STP 2016(678)</td>
      <td>1.865</td>
      <td>FM 1606</td>
      <td>2/10/2016</td>
      <td>Abilene</td>
      <td>Scurry</td>
      <td>5</td>
      <td>...</td>
      <td>6876410.33</td>
      <td>1435554.66</td>
      <td>15.743607</td>
      <td>15.933207</td>
      <td>-0.189599</td>
      <td>0</td>
      <td>0.172709</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2016-02-09</th>
      <td>Construction</td>
      <td>1531-02-012</td>
      <td>2163022</td>
      <td>STP 2016(679)</td>
      <td>0.001</td>
      <td>FM 1610</td>
      <td>2/9/2016</td>
      <td>Abilene</td>
      <td>Scurry</td>
      <td>7</td>
      <td>...</td>
      <td>5397845.79</td>
      <td>369148.18</td>
      <td>15.501511</td>
      <td>15.567662</td>
      <td>-0.066151</td>
      <td>1</td>
      <td>0.064011</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2016-02-09</th>
      <td>Construction</td>
      <td>1606-02-019</td>
      <td>2163020</td>
      <td>STP 2014(884)</td>
      <td>0.062</td>
      <td>FM 2123</td>
      <td>2/9/2016</td>
      <td>Fort Worth</td>
      <td>Wise</td>
      <td>6</td>
      <td>...</td>
      <td>2779919.35</td>
      <td>676795.35</td>
      <td>14.837932</td>
      <td>15.055829</td>
      <td>-0.217897</td>
      <td>0</td>
      <td>0.195791</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2016-02-10</th>
      <td>Construction</td>
      <td>1685-03-088</td>
      <td>2163237</td>
      <td>STP 2016(731)</td>
      <td>1.762</td>
      <td>FM 1960</td>
      <td>2/10/2016</td>
      <td>Houston</td>
      <td>Harris</td>
      <td>3</td>
      <td>...</td>
      <td>6636000.51</td>
      <td>113980.17</td>
      <td>15.708020</td>
      <td>15.725050</td>
      <td>-0.017030</td>
      <td>1</td>
      <td>0.016886</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2016-01-05</th>
      <td>Construction</td>
      <td>1685-03-096</td>
      <td>1163024</td>
      <td>STP 2016(474)</td>
      <td>2.088</td>
      <td>FM 1960</td>
      <td>1/5/2016</td>
      <td>Houston</td>
      <td>Harris</td>
      <td>3</td>
      <td>...</td>
      <td>191207.77</td>
      <td>81159.98</td>
      <td>12.161116</td>
      <td>12.514908</td>
      <td>-0.353793</td>
      <td>0</td>
      <td>0.297979</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2016-03-04</th>
      <td>Construction</td>
      <td>2003-01-008</td>
      <td>3163217</td>
      <td>STP 2016(839)HES</td>
      <td>3.594</td>
      <td>FM 2122</td>
      <td>3/4/2016</td>
      <td>Paris</td>
      <td>Lamar</td>
      <td>6</td>
      <td>...</td>
      <td>763854.27</td>
      <td>-9884.53</td>
      <td>13.546132</td>
      <td>13.533108</td>
      <td>0.013025</td>
      <td>1</td>
      <td>-0.013110</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2016-01-05</th>
      <td>Construction</td>
      <td>2079-01-043</td>
      <td>1163037</td>
      <td>STP 2013(901)</td>
      <td>2.879</td>
      <td>FM 1220</td>
      <td>1/5/2016</td>
      <td>Fort Worth</td>
      <td>Tarrant</td>
      <td>2</td>
      <td>...</td>
      <td>8350775.03</td>
      <td>-2311318.34</td>
      <td>15.937865</td>
      <td>15.613825</td>
      <td>0.324040</td>
      <td>0</td>
      <td>-0.382703</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2016-03-04</th>
      <td>Construction</td>
      <td>2263-03-041</td>
      <td>3163212</td>
      <td>C 2263-3-41</td>
      <td>5.830</td>
      <td>SH 361</td>
      <td>3/4/2016</td>
      <td>Corpus Christi</td>
      <td>Nueces</td>
      <td>4</td>
      <td>...</td>
      <td>5552811.60</td>
      <td>302405.80</td>
      <td>15.529815</td>
      <td>15.582844</td>
      <td>-0.053029</td>
      <td>1</td>
      <td>0.051647</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2016-01-06</th>
      <td>Construction</td>
      <td>2266-02-138</td>
      <td>1163205</td>
      <td>NH 2016(594)</td>
      <td>0.056</td>
      <td>SH 360</td>
      <td>1/6/2016</td>
      <td>Fort Worth</td>
      <td>Tarrant</td>
      <td>2</td>
      <td>...</td>
      <td>2126639.56</td>
      <td>-165737.61</td>
      <td>14.570054</td>
      <td>14.488915</td>
      <td>0.081139</td>
      <td>1</td>
      <td>-0.084521</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2016-01-05</th>
      <td>Construction</td>
      <td>2296-01-051</td>
      <td>1163035</td>
      <td>NH 2016(565)</td>
      <td>7.055</td>
      <td>SH 191</td>
      <td>1/5/2016</td>
      <td>Odessa</td>
      <td>Ector</td>
      <td>3</td>
      <td>...</td>
      <td>4387311.96</td>
      <td>-338356.71</td>
      <td>15.294227</td>
      <td>15.213969</td>
      <td>0.080258</td>
      <td>1</td>
      <td>-0.083566</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2016-01-05</th>
      <td>Construction</td>
      <td>2560-01-076</td>
      <td>1163033</td>
      <td>STP 2016(425)</td>
      <td>3.901</td>
      <td>SL 224</td>
      <td>1/5/2016</td>
      <td>Lufkin</td>
      <td>Nacogdoches</td>
      <td>3</td>
      <td>...</td>
      <td>789787.24</td>
      <td>12523.33</td>
      <td>13.579519</td>
      <td>13.595251</td>
      <td>-0.015732</td>
      <td>1</td>
      <td>0.015609</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2016-04-06</th>
      <td>Construction</td>
      <td>2607-01-011</td>
      <td>4163205</td>
      <td>STP 2016(962)</td>
      <td>1.492</td>
      <td>SH 188</td>
      <td>4/6/2016</td>
      <td>Corpus Christi</td>
      <td>Aransas</td>
      <td>2</td>
      <td>...</td>
      <td>1142011.55</td>
      <td>-117358.92</td>
      <td>13.948302</td>
      <td>13.839864</td>
      <td>0.108438</td>
      <td>0</td>
      <td>-0.114535</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2016-04-05</th>
      <td>Construction</td>
      <td>2754-01-008</td>
      <td>4163012</td>
      <td>STP 2016(370)</td>
      <td>3.626</td>
      <td>FM 2705</td>
      <td>4/5/2016</td>
      <td>Waco</td>
      <td>Limestone</td>
      <td>3</td>
      <td>...</td>
      <td>4733942.79</td>
      <td>-79712.79</td>
      <td>15.370269</td>
      <td>15.353287</td>
      <td>0.016982</td>
      <td>1</td>
      <td>-0.017127</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2016-04-06</th>
      <td>Construction</td>
      <td>2887-01-012</td>
      <td>4163206</td>
      <td>STP 2016(958)HES</td>
      <td>4.942</td>
      <td>FM 2830</td>
      <td>4/6/2016</td>
      <td>Beaumont</td>
      <td>Liberty</td>
      <td>5</td>
      <td>...</td>
      <td>999523.59</td>
      <td>168362.80</td>
      <td>13.815034</td>
      <td>13.970706</td>
      <td>-0.155672</td>
      <td>0</td>
      <td>0.144160</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2016-02-10</th>
      <td>Construction</td>
      <td>3050-02-024</td>
      <td>2163202</td>
      <td>STP 2016(119)MM</td>
      <td>0.001</td>
      <td>FM 2978</td>
      <td>2/10/2016</td>
      <td>Houston</td>
      <td>Montgomery</td>
      <td>5</td>
      <td>...</td>
      <td>15788778.28</td>
      <td>612127.00</td>
      <td>16.574810</td>
      <td>16.612847</td>
      <td>-0.038037</td>
      <td>1</td>
      <td>0.037323</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2016-02-10</th>
      <td>Construction</td>
      <td>3136-01-177</td>
      <td>2163239</td>
      <td>STP 2016(637)</td>
      <td>0.054</td>
      <td>SL 1</td>
      <td>2/10/2016</td>
      <td>Austin</td>
      <td>Travis</td>
      <td>4</td>
      <td>...</td>
      <td>787593.55</td>
      <td>94266.36</td>
      <td>13.576737</td>
      <td>13.689788</td>
      <td>-0.113051</td>
      <td>0</td>
      <td>0.106895</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2016-02-09</th>
      <td>Construction</td>
      <td>3256-02-091</td>
      <td>2163032</td>
      <td>NH 2016(728)</td>
      <td>0.026</td>
      <td>SL 8</td>
      <td>2/9/2016</td>
      <td>Houston</td>
      <td>Harris</td>
      <td>3</td>
      <td>...</td>
      <td>596804.70</td>
      <td>-98636.20</td>
      <td>13.299345</td>
      <td>13.118694</td>
      <td>0.180652</td>
      <td>0</td>
      <td>-0.197998</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2016-04-05</th>
      <td>Construction</td>
      <td>3271-01-011</td>
      <td>4163009</td>
      <td>STP 2016(959)</td>
      <td>0.016</td>
      <td>FM 3180</td>
      <td>4/5/2016</td>
      <td>Beaumont</td>
      <td>Chambers</td>
      <td>4</td>
      <td>...</td>
      <td>6439873.51</td>
      <td>-1090631.57</td>
      <td>15.678019</td>
      <td>15.492465</td>
      <td>0.185554</td>
      <td>0</td>
      <td>-0.203885</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2016-03-03</th>
      <td>Construction</td>
      <td>3277-01-022</td>
      <td>3163005</td>
      <td>STP 2014(116)TE</td>
      <td>30.194</td>
      <td>FM 3177</td>
      <td>3/3/2016</td>
      <td>Austin</td>
      <td>Travis</td>
      <td>5</td>
      <td>...</td>
      <td>2045353.10</td>
      <td>67623.37</td>
      <td>14.531081</td>
      <td>14.563608</td>
      <td>-0.032527</td>
      <td>1</td>
      <td>0.032004</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2016-01-05</th>
      <td>Construction</td>
      <td>3348-01-019</td>
      <td>1163018</td>
      <td>STP 1502(549)HES</td>
      <td>0.001</td>
      <td>FM 3136</td>
      <td>1/5/2016</td>
      <td>Fort Worth</td>
      <td>Johnson</td>
      <td>2</td>
      <td>...</td>
      <td>6829235.17</td>
      <td>-2143649.77</td>
      <td>15.736723</td>
      <td>15.360001</td>
      <td>0.376722</td>
      <td>0</td>
      <td>-0.457499</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2016-02-10</th>
      <td>Construction</td>
      <td>3417-01-028</td>
      <td>2163210</td>
      <td>CC 3417-1-28</td>
      <td>3.834</td>
      <td>FM 734</td>
      <td>2/10/2016</td>
      <td>Austin</td>
      <td>Travis</td>
      <td>4</td>
      <td>...</td>
      <td>839340.91</td>
      <td>-60727.46</td>
      <td>13.640372</td>
      <td>13.565270</td>
      <td>0.075102</td>
      <td>1</td>
      <td>-0.077994</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2016-04-05</th>
      <td>Construction</td>
      <td>3538-01-041</td>
      <td>4163027</td>
      <td>STP 2016(963)</td>
      <td>0.030</td>
      <td>SH 242</td>
      <td>4/5/2016</td>
      <td>Houston</td>
      <td>Montgomery</td>
      <td>2</td>
      <td>...</td>
      <td>3635039.14</td>
      <td>841444.90</td>
      <td>15.106130</td>
      <td>15.314348</td>
      <td>-0.208218</td>
      <td>0</td>
      <td>0.187970</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2016-02-09</th>
      <td>Construction</td>
      <td>3538-01-044</td>
      <td>2163036</td>
      <td>STP 2016(722)HES</td>
      <td>3.529</td>
      <td>SH 242</td>
      <td>2/9/2016</td>
      <td>Houston</td>
      <td>Montgomery</td>
      <td>3</td>
      <td>...</td>
      <td>275772.88</td>
      <td>9539.89</td>
      <td>12.527333</td>
      <td>12.561341</td>
      <td>-0.034008</td>
      <td>1</td>
      <td>0.033437</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5177 rows × 43 columns</p>
</div>



## Exploring the Data

## We see that Bids are Log Normally Distributed


```python
sns.jointplot(x="EngEst", y="WinBid", data=df, kind="reg"); sns.jointplot(x="lnEngEst", y="lnWinBid", data=df, kind="reg");
```

    C:\Users\Collin\Anaconda3\lib\site-packages\statsmodels\nonparametric\kdetools.py:20: VisibleDeprecationWarning: using a non-integer number instead of an integer will result in an error in the future
      y = X[:m/2+1] + np.r_[0,X[m/2+1:],0]*1j
    


![png](output_8_1.png)



![png](output_8_2.png)



```python
cmap = {'0': 'g', '1': 'r', '2': 'b' }
df['cMoreOrLessThan10'] = df.MoreOrLessThan10.apply(lambda x: cmap[str(x)])
print (df.plot('lnEngEst', 'lnWinBid', kind='scatter', c=df.cMoreOrLessThan10))
```

    Axes(0.125,0.125;0.775x0.755)
    


![png](output_9_1.png)



```python
df_test = df[(df.Year == 2016) & (df.Month == 4)]
print(len(df_test) , 'projects in April 2016')

df_train = df[(df.Year != 2016) | (df.Month != 4)]
print(len(df_train) ,'projects from Jan 2010 to April 2016')

#df_train[['Year','Month']].tail()
```

    67 projects in April 2016
    5110 projects from Jan 2010 to April 2016
    


```python
#Using ALL the Data

Percent = float(df.Within10Percent.sum()) / len(df)
print(round((Percent)*100,2) , '% of All the TxDOT estimates were within 10% of actual bid')

Percent_April_2016 = float(df[(df.Year == 2016) & (df.Month == 4)].Within10Percent.sum()) / len(df_test)
print (round((Percent_April_2016)*100,2) , '% of the April 2016 TxDOT estimates were within 10% of actual bid')
```

    50.34 % of All the TxDOT estimates were within 10% of actual bid
    46.27 % of the April 2016 TxDOT estimates were within 10% of actual bid
    


```python
names_X = ['Length','NBidders','Year','Month','lnEngEst','Time']

def X_y(df):
    X = df[ names_X ]
    y_more = df['MoreThan10']
    y_less =df['LessThan10']
    return X, y_more, y_less

train_X, train_y_more, train_y_less = X_y(df_train)
test_X, test_y_more, test_y_less = X_y(df_test)

print(len(train_y_more))
print(len(train_y_less))
print(len(test_y_more))
print(len(test_y_less))
```

    5110
    5110
    67
    67
    


```python
test_X.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Length</th>
      <th>NBidders</th>
      <th>Year</th>
      <th>Month</th>
      <th>lnEngEst</th>
      <th>Time</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2016-04-05</th>
      <td>15.055</td>
      <td>2</td>
      <td>2016</td>
      <td>4</td>
      <td>14.979827</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2016-04-06</th>
      <td>0.016</td>
      <td>5</td>
      <td>2016</td>
      <td>4</td>
      <td>14.722602</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2016-04-06</th>
      <td>2.024</td>
      <td>4</td>
      <td>2016</td>
      <td>4</td>
      <td>16.836501</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2016-04-06</th>
      <td>6.294</td>
      <td>3</td>
      <td>2016</td>
      <td>4</td>
      <td>14.893941</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2016-04-06</th>
      <td>0.455</td>
      <td>3</td>
      <td>2016</td>
      <td>4</td>
      <td>13.837735</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
def RFC_model(X, y):
    """ Performs grid search over the 'n_estimators' parameter for a 
        random forest regressor trained on the input data [X, y]. """
    
    # Create cross-validation sets from the training data
    cv_sets = KFold(n_splits = 30)
    cv_sets.split(X,y)

    # Create a decision tree regressor object
    clf = RandomForestClassifier()

    # Create a dictionary for the parameter 'max_depth' with a range from 1 to 100
    params = {'n_estimators':range(1,len(X.columns))}

    # Transform 'performance_metric' into a scoring function using 'make_scorer' 
    #scoring_fnc = make_scorer(performance_metric)

    # Create the grid search object
    grid = GridSearchCV(clf, params, cv=cv_sets)

    # Fit the grid search object to the data to compute the optimal model
    grid = grid.fit(X, y)

    # Return the optimal model after fitting the data
    return grid.best_estimator_
```


```python
model_1 = RFC_model(train_X,train_y_more)
```


```python
print('correct training classification = ', model_1.score(train_X, train_y_more))
print ('correct testing classification = ', model_1.score(test_X, test_y_more))
```

    correct training classification =  0.929549902153
    correct testing classification =  0.641791044776
    


```python

```
