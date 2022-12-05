# **Predicting Housing Price Using Neural Network** [Tensorflow  and Keras]


<br>

## _The Data:_

<br>

Real DB from Kaggle: 
(https://www.kaggle.com/datasets/harlfoxem/housesalesprediction)

<br>

2014-15 Home Sales in King County, WA, one of three Washington counties that are included in the Seattle metropolitan area.

<br>

DB Dictionary (https://geodacenter.github.io/data-and-lab/KingCounty-HouseSales2015/):

<br>

- id: Identification
- date:	Date sold
- price: Sale price
- bedrooms:	Number of bedrooms
- bathrooms: Number of bathrooms
- sqft_liv:	Size of living area in square feet
- sqft_lot:	Size of the lot in square feet
- floors: Number of floors
- waterfront: ‘1’ if the property has a waterfront, ‘0’ if not.
- view	An index from 0 to 4 of how good the view of the property was
- condition	Condition of the house, ranked from 1 to 5
- grade	Classification by construction quality which refers to the types of materials used and the quality of workmanship. Buildings of better quality (higher grade) cost more to build per unit of measure and command higher value. Additional information in: KingCounty
- sqft_above: Square feet above ground
- sqft_basmt: Square feet below ground
- yr_built: Year built
- yr_renov: Year renovated. ‘0’ if never renovated
- zipcode: 5 digit zip code
- lat: Latitude
- long: Longitude
- squft_liv15: Average size of interior housing living space for the closest 15 houses, in square feet
- squft_lot15: Average size of land lots for the closest 15 houses, in square feet
- Shape_leng: Polygon length in meters
- Shape_Area: Polygon area in meters

<br>

## _Goal / Target:_

<br>

Predict house prices.

<br>

## _Tools:_

<br>

- Python
- Numpy, Pandas
- Matplotlib, Seaborn
- Scikitlearn
- Keras, Tensorflow 

<br>

---

<br>

## **Step 1: Preparing the Data**

<br>

There isn't missing data (null) or duplicate values.

```python
df.isnull().sum()

id               0
date             0
price            0
bedrooms         0
bathrooms        0
sqft_living      0
sqft_lot         0
floors           0
waterfront       0
view             0
condition        0
grade            0
sqft_above       0
sqft_basement    0
yr_built         0
yr_renovated     0
zipcode          0
lat              0
long             0
sqft_living15    0
sqft_lot15       0
dtype: int64

```

<br>

---

<br>

## **Step 2: EDA - Exploratory Data Analysis**

<br>

Probably we can predict very well the house prices between 0 and 2 million dolars. So, we can exclude the outliers.

```python
sns.histplot(df['price'],kde=True)
```

![img histplot](https://github.com/TSLSouth/House-Price-Predictions-with-Artificial-Neural-Network/blob/main/img/histplot.png?raw=true)

<br>

Exploring posible correlations.

Seems that the price is very correlated with sqft_living feature:
<br>

```python
df.corr()['price'].sort_values()

zipcode         -0.053402
id              -0.016772
long             0.022036
condition        0.036056
yr_built         0.053953
sqft_lot15       0.082845
sqft_lot         0.089876
yr_renovated     0.126424
floors           0.256804
waterfront       0.266398
lat              0.306692
bedrooms         0.308787
sqft_basement    0.323799
view             0.397370
bathrooms        0.525906
sqft_living15    0.585241
sqft_above       0.605368
grade            0.667951
sqft_living      0.701917
price            1.000000
Name: price, dtype: float64
```

<br>

We can see the same correlation on the pairplot and with others faetures. 
#But price and sqft_living is really correlated.

Not showing it here beacause its not a good way to see in pairplot with a lot of attributes, the plots got very small.

<br> 

Let´s see that correlation closely:

```python
sns.scatterplot(data=df,x='price',y='sqft_living')
```

![img scatterplot](https://github.com/TSLSouth/House-Price-Predictions-with-Artificial-Neural-Network/blob/main/img/scatterplot.png?raw=true)

<br>

Other correlations:

```python
sns.scatterplot(data=df,x='price',y='bathrooms')
```

![img price vs bathrooms](https://github.com/TSLSouth/House-Price-Predictions-with-Artificial-Neural-Network/blob/main/img/price_bath.png?raw=true)

<br>

```python
sns.scatterplot(data=df,x='price',y='sqft_above')
```

![img price vs sqft above](https://github.com/TSLSouth/House-Price-Predictions-with-Artificial-Neural-Network/blob/main/img/price_sqft_above.png?raw=true)

<br>

Distributions of price per number of bedrooms:

```python
sns.boxplot(data=df,x='bedrooms',y='price')
```

![img price vs bedrooms](https://github.com/TSLSouth/House-Price-Predictions-with-Artificial-Neural-Network/blob/main/img/price_bedrooms.png?raw=true)

<br> 

Let's use lat (latitude) and long (longitude feature) to build a King County's map and check if that interferes on the house prices:

```python
sns.scatterplot(data=df,x='long',y='lat',hue='price')
```

![img kings county map1](https://github.com/TSLSouth/House-Price-Predictions-with-Artificial-Neural-Network/blob/main/img/kingscounty_map1.png?raw=true)

<br>

Let's drop 1% of the house/prices (the outliers):

```python
len(df)*0.01
215.97

df_99perc = df.sort_values('price',ascending=False).iloc[216:]

sns.scatterplot(data=df_99perc,x='long',y='lat',hue='price',
               edgecolor=None, alpha=0.2, palette='RdYlGn')

# Red less expensive
# Green more expensive
```

![img kings county map2](https://github.com/TSLSouth/House-Price-Predictions-with-Artificial-Neural-Network/blob/main/img/kingscounty_map2.png?raw=true)

<br>

Water front feature vs. prices:

```python
sns.boxplot(data=df,x='waterfront',y='price')
```

![img waterfront vs. price](https://github.com/TSLSouth/House-Price-Predictions-with-Artificial-Neural-Network/blob/main/img/waterfront_price.png?raw=true)

<br>

---

<br>

## **Step 3: Feature Engineering Process**

<br>

Transformming the most relevant variables from raw data to improve the performance of machine learning (ML) algorithms.

<br>

Lets drop features that are not important to us and use just 99% of the data (taking out those outliers):

```python
df_99perc = df.drop('id',axis=1)
```

<br>

Converting to datetime object:

```python
df_99perc['date'] = pd.to_datetime(df_99perc['date'])
```

Creating two new columns to observe the datetime better:

```python
df_99perc['year'] = df_99perc['date'].apply(lambda date: date.year)

df_99perc['month'] = df_99perc['date'].apply(lambda date: date.month)

df_99perc = df_99perc.drop('date',axis=1)
```
<br>

There is some influence in on sales price by month? A bit of influence, yes.

```python
df_99perc.groupby('month').mean()['price'].plot()
```
![img sales price by month](https://github.com/TSLSouth/House-Price-Predictions-with-Artificial-Neural-Network/blob/main/img/salesprice_bymonth.png?raw=true)

<br> 

The prices go up one year after other, what makes sense:

```python
df_99perc.groupby('year').mean()['price'].plot()
```

![img sales price by year](https://github.com/TSLSouth/House-Price-Predictions-with-Artificial-Neural-Network/blob/main/img/salesprice%20by%20year.png?raw=true)

<br>

Let´s drop the zipcode too because it's not logical distributted and we already have latitude and longitude.

![img postal code map](https://github.com/TSLSouth/House-Price-Predictions-with-Artificial-Neural-Network/blob/main/img/Brooklyn-Postal-Code-Map.jpg?raw=true)

```python
df_99perc = df_99perc.drop('zipcode',axis=1)
```
<br>

---

<br>

## **Step 4: Predictions**

<br>

## _Train test split:_

<br>

```python
X = df_99perc.drop('price',axis=1).values
y = df_99perc['price'].values

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
```

<br>

## _Scalling the data:_

<br>

```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)
```

<br>

## _Creating a deep learning model with Tensorflow and Keras:_

<br>

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

X_train.shape

(15117, 19)
```

<br>

Let´s use 19 neurons per layer:

```python
model = Sequential()

model.add(Dense(19,activation='relu'))
model.add(Dense(19,activation='relu'))
model.add(Dense(19,activation='relu'))
model.add(Dense(19,activation='relu')) 

model.add(Dense(1))

model.compile(optimizer='adam',loss='mse')
```

<br>

_Obs.: Adam optimization is a stochastic gradient descent method beacuse it 'combine' the benefits of AdaGrad (Adaptive Gradient Algorithm) and RMSProp (Root Mean Square Propagation)._

<br>

## _Trainning our model:_

<br>

```python
model.fit(x=X_train,y=y_train,
         validation_data=(X_test,y_test),
         batch_size=128,
         epochs=400,
         verbose=0) 
```

<br>

## _Evaluating the Model:_

<br>

History of the **calculated losses**:

```python
pd.DataFrame(model.history.history)

	loss	        val_loss
0	4.302356e+11	4.188866e+11
1	4.287633e+11	4.137970e+11
2	4.083790e+11	3.674646e+11
3	3.131799e+11	2.215630e+11
4	1.561175e+11	1.030742e+11
...       ...            ...
```

<br>

Plot:
```python
losses = pd.DataFrame(model.history.history)
losses.plot()
```

![img calculated losses] ()

<br>

## _Predicting:_

<br>

```python
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score

pred = model.predict(X_test)

pred

array([[432760.72],
       [586596.44],
       [575481.7 ],
       ...,
       [406132.4 ],
       [597030.94],
       [683578.3 ]], dtype=float32)
```
<br>

RMSE:

```python
np.sqrt(mean_squared_error(y_test,pred))

162532.95927484107
```

<br>

MAE:

```python
mean_absolute_error(y_test,pred)

100852.6434148341
```

<br>

Mean of the house prices:

```python
df_99perc['price'].describe()

count    2.159700e+04
mean     5.402966e+05
std      3.673681e+05
min      7.800000e+04
25%      3.220000e+05
50%      4.500000e+05
75%      6.450000e+05
max      7.700000e+06
Name: price, dtype: float64
```

<br>

Mean of the house prices = 540.000
<p>RMSE = 163.000
<p>MAE = 101.000
<br>

Our model is predicting with 18-20% of error, not good.

<br>

How much of the variance is explained by our model:

```python
explained_variance_score(y_test,pred)

0.8008035596774933
```

<br>

Plotting predictions vs. real values:

```python
plt.figure(figsize=(16,9))
plt.scatter(y_test,pred)
plt.plot(y_test,y_test,'r')
```

![img predictions vs. real values](https://github.com/TSLSouth/House-Price-Predictions-with-Artificial-Neural-Network/blob/main/img/predictions%20vs.%20real%20values.png?raw=true)

<br>

## _Conclusion:_

<br>

Looks like we should train our model again using less than 99% of the houses sale prices or try a multidimensional model.

<br>

## _Calculating a price for a new house:_

<br>

```python
new_house = df_99perc.drop('price',axis=1).iloc[0]

new_house = scaler.transform(new_house.values.reshape(-1,19))

model.predict(new_house)

array([[280802.2]], dtype=float32)
```
