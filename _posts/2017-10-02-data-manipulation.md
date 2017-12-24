---
layout:     post
title:      Data Manipulation
date:       2017-10-02 02:10:29
summary:    Using NumPy, Pandas & Matplotlib
categories: numpy, pandas, matplotlib, data visulization, data analytics, scipy
---
## NumPy

## What is NumPy
> NumPy is the fundamental package for scientific computing in Python. It is a Python library that provides a multidimensional array object, various derived objects (such as masked arrays and matrices), and an assortment of routines for fast operations on arrays, including mathematical, logical, shape manipulation, sorting, selecting, I/O, discrete Fourier transforms, basic linear algebra, basic statistical operations, random simulation and much more.

NumPy is used for scientific computations in Python. These computations are very useful while doing machine learning computations. It can be used to write a machine-learning and deep learning algorithms from the scratch. Out of the libraries we are going to cover in this blog, Numpy is the **most important** one. So you should have a good understanding of Numpy.By using NumPy, you can speed up your workflow, and interface with other packages in the Python ecosystem, like scikit-learn, that use NumPy under the hood.

Python lists have limitations to it. Like we cannot perform algebric computations using list. Thus Numpy has arrays, which cover all the limitations of lists.NumPy arrays are more like C arrays than Python lists. 


```python
import numpy as np
```

After this import statement, we will always use pd for accessing the in-built pandas functions and methods.

Creating a numpy array.


```python
s=np.array([1,2,3,4])
s
```




    array([1, 2, 3, 4])



#### Some basic computations


```python
s+3
```




    array([4, 5, 6, 7])



So we add 3 to all the elements. It is just simple scalar addition. If we do this operation on lists, then it won't add 3 to the numbers, instead it will add append 3 to the list.


```python
[1,2,3,4]+[3] #example
```




    [1, 2, 3, 4, 3]



#### Subtraction


```python
s-3 #we can do similar scalar computations
```




    array([-2, -1,  0,  1])



#### Checking the type and shape


```python
print(type(s))
print(s.shape)
print(s.dtype)
```

    <class 'numpy.ndarray'>
    (2, 4)
    int64


The output clearly shows that the type of s is of type ndarray.

The shape of the array is (4,). This means that it is a one-dimensional array, with 4 columns and 1 rows.

The datatype of the array elements is of type int.


```python
s=np.array([[1,2,3,4],[7,8,9,10]])
s.shape
```




    (2, 4)



The shape now here is (2,4) which means 2 rows and 4 columns. Also for creating a multidimensional array, we pass multiple lists as done above.

#### Converting to a different data type


```python
s.astype('float')
```




    array([[  1.,   2.,   3.,   4.],
           [  7.,   8.,   9.,  10.]])



#### Generating Random Numbers With Numpy


```python
np.random.normal()
```




    3.137676397424334



The above code gives a random number from normal distribution.

#### Generate Multiple Random Numbers From The Normal Distribution


```python
np.random.normal(size=4)
```




    array([-0.47623571,  1.7373749 ,  0.15226591,  0.38203617])



#### Generate Four Random Integers Between 1 and 100


```python
np.random.randint(low=1, high=100, size=4)
```




    array([47, 69, 27, 65])



#### Create an Array of Zeroes and Range Generator


```python
np.zeros(6)
```




    array([ 0.,  0.,  0.,  0.,  0.,  0.])



#### Create an array from 1 to 100


```python
np.arange(0, 100)
```




    array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
           17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
           34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
           51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67,
           68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84,
           85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99])



#### Create 100 equally space numbers between 0 and 1


```python
np.linspace(0, 1, 100)
```




    array([ 0.        ,  0.01010101,  0.02020202,  0.03030303,  0.04040404,
            0.05050505,  0.06060606,  0.07070707,  0.08080808,  0.09090909,
            0.1010101 ,  0.11111111,  0.12121212,  0.13131313,  0.14141414,
            0.15151515,  0.16161616,  0.17171717,  0.18181818,  0.19191919,
            0.2020202 ,  0.21212121,  0.22222222,  0.23232323,  0.24242424,
            0.25252525,  0.26262626,  0.27272727,  0.28282828,  0.29292929,
            0.3030303 ,  0.31313131,  0.32323232,  0.33333333,  0.34343434,
            0.35353535,  0.36363636,  0.37373737,  0.38383838,  0.39393939,
            0.4040404 ,  0.41414141,  0.42424242,  0.43434343,  0.44444444,
            0.45454545,  0.46464646,  0.47474747,  0.48484848,  0.49494949,
            0.50505051,  0.51515152,  0.52525253,  0.53535354,  0.54545455,
            0.55555556,  0.56565657,  0.57575758,  0.58585859,  0.5959596 ,
            0.60606061,  0.61616162,  0.62626263,  0.63636364,  0.64646465,
            0.65656566,  0.66666667,  0.67676768,  0.68686869,  0.6969697 ,
            0.70707071,  0.71717172,  0.72727273,  0.73737374,  0.74747475,
            0.75757576,  0.76767677,  0.77777778,  0.78787879,  0.7979798 ,
            0.80808081,  0.81818182,  0.82828283,  0.83838384,  0.84848485,
            0.85858586,  0.86868687,  0.87878788,  0.88888889,  0.8989899 ,
            0.90909091,  0.91919192,  0.92929293,  0.93939394,  0.94949495,
            0.95959596,  0.96969697,  0.97979798,  0.98989899,  1.        ])



#### Indexing and Slicing Numpy Arrays


```python
a = [[34, 25], [75, 45]]
b = np.array(a)
b
```




    array([[34, 25],
           [75, 45]])




```python
b[0, 1]# Select the top row, second item
```




    25




```python
b[:, 1]# Select the second column
```




    array([25, 45])




```python
b[1, :]# Select the second row
```




    array([75, 45])




```python
b[1,1] # select the intersection of 1st row and 1st column
```




    45



#### Aggregate functions


```python
print(b.mean())
print(b.min())
print(b.max())
```

    44.75
    25
    75


#### Change Value using Array Index


```python
b[0]=100
b
```




    array([[100, 100],
           [ 75,  45]])



The value of the entire row is set to 100. If you want to change only 1 value, use the following:


```python
b[0,1]=12
b
```




    array([[100,  12],
           [ 75,  45]])



Now just the 2nd element of 1st row changes.

#### Changing Shapes

Sometimes you'll need to change the shape of your data without actually changing its contents. For example, you may have a vector, which is one-dimensional, but need a matrix, which is two-dimensional. There are two ways you can do that.


```python
b.reshape(1,4)
```




    array([[100, 100,  75,  45]])




```python
b.reshape(4,1)
```




    array([[100],
           [ 12],
           [ 75],
           [ 45]])



#### More Examples on Reshaping


```python
a=np.arange(6).reshape((3, 2))
a
```




    array([[0, 1],
           [2, 3],
           [4, 5]])




```python
np.reshape(a, (2, 3))
```




    array([[0, 1, 2],
           [3, 4, 5]])



#### Transpose

Transpose is the same as matrix transpose. 

m(row,column)=m(column,row)


```python
a
```




    array([[0, 1],
           [2, 3],
           [4, 5]])




```python
a.T
```




    array([[0, 2, 4],
           [1, 3, 5]])




```python
a.transpose()
```




    array([[0, 2, 4],
           [1, 3, 5]])



### Matrix Multiplication


```python
a
```




    array([[0, 1],
           [2, 3],
           [4, 5]])



### Using scalar multiplication


```python
a*a
```




    array([[ 0,  1],
           [ 4,  9],
           [16, 25]])



The above result is not a matrix multiplication product. It simply multiplies the numbers element wise. That means (0,0) is multiplied with (0,0), (0,1) with (0,1) and so on. This is not matrix multiplication but a normal scalar multiplication.

### Using Matrix Multiplication


```python
a=np.array([[1,2],[3,4]])
```


```python
np.dot(a,a)
```




    array([[ 7, 10],
           [15, 22]])



The above result is a proper matrix multiplication

## Pandas

The real world datasets are not very clean, and contain a lot of irrelevant,dummy and empty values. We cannot pass the data as it is to the Machine Learning algorithms, as the algorithms can process the data in only some formats. Thus it is necessary to tranform the data into appropriate formats. For cleaning and transformation of data, we use the Pandas library.

**Pandas** is the most widely used library for data munging and transformation. It is Python equivalent of SQL. SQL, as you might know is used for directly working with databases, while Pandas is used to work with tables rather than databases. However Pandas has a better data cleaning and wrangling capacity. A basic knowledge of Pandas is sufficient enough for working with Machine Learning problems. In the following blog, I will try to explain all the important functions and methods in Pandas, and try to give its SQL equivalent code for better understanding.   


```python
#importing the library
import pandas as pd
```


```python
df=pd.read_csv('Pokemon.csv')
```

Now Pandas has tha capacity to read different types of data files. It can be a excel file, .csv file or even a json file. All of these files have their different method. Like read_csv() is for csv files, similarly read_excel() is for .xlsl file or excel files. Now df is the variable where the csv file is stored.

### Displaying the Data

Just to check what does the dataset looks like. We will display the first 5 rows of the dataset

**SQL:** select * from df LIMIT 5;


```python
df.head(5)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>#</th>
      <th>Name</th>
      <th>Type 1</th>
      <th>Type 2</th>
      <th>Total</th>
      <th>HP</th>
      <th>Attack</th>
      <th>Defense</th>
      <th>Sp. Atk</th>
      <th>Sp. Def</th>
      <th>Speed</th>
      <th>Generation</th>
      <th>Legendary</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Bulbasaur</td>
      <td>Grass</td>
      <td>Poison</td>
      <td>318</td>
      <td>45</td>
      <td>49</td>
      <td>49</td>
      <td>65</td>
      <td>65</td>
      <td>45</td>
      <td>1</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Ivysaur</td>
      <td>Grass</td>
      <td>Poison</td>
      <td>405</td>
      <td>60</td>
      <td>62</td>
      <td>63</td>
      <td>80</td>
      <td>80</td>
      <td>60</td>
      <td>1</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>Venusaur</td>
      <td>Grass</td>
      <td>Poison</td>
      <td>525</td>
      <td>80</td>
      <td>82</td>
      <td>83</td>
      <td>100</td>
      <td>100</td>
      <td>80</td>
      <td>1</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>VenusaurMega Venusaur</td>
      <td>Grass</td>
      <td>Poison</td>
      <td>625</td>
      <td>80</td>
      <td>100</td>
      <td>123</td>
      <td>122</td>
      <td>120</td>
      <td>80</td>
      <td>1</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>Charmander</td>
      <td>Fire</td>
      <td>NaN</td>
      <td>309</td>
      <td>39</td>
      <td>52</td>
      <td>43</td>
      <td>60</td>
      <td>50</td>
      <td>65</td>
      <td>1</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>



.head() with the number of rows as the argument shows the first 'n' rows of the dataset. If you just write df without .head(), the complete dataframe will be displayed.

Now the above table is known as a dataframe, and each column of the 'dataframe' is known as a 'Series'.

For displaying a single column of the dataframe, just specify the column name as follwing: **df['column-name']**


```python
df['HP'].head(5)
```




    0    45
    1    60
    2    80
    3    80
    4    39
    Name: HP, dtype: int64



#### Showing only certain columns

**SQL:** select HP, Defence, Total from df;

Thsi will only display name and salary from salary table.In pandas , we pass a list of column names to the variable as follows:

**df[['col-name1','col-name2']]**


```python
df[['HP','Defense','Total']].head(5)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>HP</th>
      <th>Defense</th>
      <th>Total</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>45</td>
      <td>49</td>
      <td>318</td>
    </tr>
    <tr>
      <th>1</th>
      <td>60</td>
      <td>63</td>
      <td>405</td>
    </tr>
    <tr>
      <th>2</th>
      <td>80</td>
      <td>83</td>
      <td>525</td>
    </tr>
    <tr>
      <th>3</th>
      <td>80</td>
      <td>123</td>
      <td>625</td>
    </tr>
    <tr>
      <th>4</th>
      <td>39</td>
      <td>43</td>
      <td>309</td>
    </tr>
  </tbody>
</table>
</div>



#### Slicing

Slicing dataframes is same as slicing lists. The only difference is that we can pass two dimensions of slicing, 1-row and 2-column.

The syntax is **df.iloc[row,column]**


```python
df.iloc[1:,:4].head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>#</th>
      <th>Type 1</th>
      <th>Type 2</th>
      <th>Total</th>
    </tr>
    <tr>
      <th>Name</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Ivysaur</th>
      <td>2</td>
      <td>Grass</td>
      <td>Poison</td>
      <td>405</td>
    </tr>
    <tr>
      <th>Venusaur</th>
      <td>3</td>
      <td>Grass</td>
      <td>Poison</td>
      <td>525</td>
    </tr>
    <tr>
      <th>VenusaurMega Venusaur</th>
      <td>3</td>
      <td>Grass</td>
      <td>Poison</td>
      <td>625</td>
    </tr>
    <tr>
      <th>Charmander</th>
      <td>4</td>
      <td>Fire</td>
      <td>unknown</td>
      <td>309</td>
    </tr>
    <tr>
      <th>Charmeleon</th>
      <td>5</td>
      <td>Fire</td>
      <td>unknown</td>
      <td>405</td>
    </tr>
  </tbody>
</table>
</div>



In the above statement, we slice from 2nd row till the last row, and 1st column to the 3rd column.

### Data Filtering

Data Filtering is done by using a condition in the **where** clause in SQL. For pandas we have boolean filtering, where we specify a condition on some column.

**SQL:** select * from df;

  where Total>100;
 
Lets filter Pokemons with Total > 100


```python
(df['Total']>100).head(5)
```




    0    True
    1    True
    2    True
    3    True
    4    True
    Name: Total, dtype: bool



Whats the problem in the above output?? The problem is that we just get *True or False* and not the actual data. For getting the actual data, write the above code in following way: **df[df['column-name'] condition]**


```python
df[df['Total']>100].head(5)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>#</th>
      <th>Name</th>
      <th>Type 1</th>
      <th>Type 2</th>
      <th>Total</th>
      <th>HP</th>
      <th>Attack</th>
      <th>Defense</th>
      <th>Sp. Atk</th>
      <th>Sp. Def</th>
      <th>Speed</th>
      <th>Generation</th>
      <th>Legendary</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Bulbasaur</td>
      <td>Grass</td>
      <td>Poison</td>
      <td>318</td>
      <td>45</td>
      <td>49</td>
      <td>49</td>
      <td>65</td>
      <td>65</td>
      <td>45</td>
      <td>1</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Ivysaur</td>
      <td>Grass</td>
      <td>Poison</td>
      <td>405</td>
      <td>60</td>
      <td>62</td>
      <td>63</td>
      <td>80</td>
      <td>80</td>
      <td>60</td>
      <td>1</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>Venusaur</td>
      <td>Grass</td>
      <td>Poison</td>
      <td>525</td>
      <td>80</td>
      <td>82</td>
      <td>83</td>
      <td>100</td>
      <td>100</td>
      <td>80</td>
      <td>1</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>VenusaurMega Venusaur</td>
      <td>Grass</td>
      <td>Poison</td>
      <td>625</td>
      <td>80</td>
      <td>100</td>
      <td>123</td>
      <td>122</td>
      <td>120</td>
      <td>80</td>
      <td>1</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>Charmander</td>
      <td>Fire</td>
      <td>NaN</td>
      <td>309</td>
      <td>39</td>
      <td>52</td>
      <td>43</td>
      <td>60</td>
      <td>50</td>
      <td>65</td>
      <td>1</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>



Now the above output contains the Pokemons with HP > 100

#### Multiple conditions


```python
df[(df['Total']>100)&(df['HP']>50)].head(5)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>#</th>
      <th>Name</th>
      <th>Type 1</th>
      <th>Type 2</th>
      <th>Total</th>
      <th>HP</th>
      <th>Attack</th>
      <th>Defense</th>
      <th>Sp. Atk</th>
      <th>Sp. Def</th>
      <th>Speed</th>
      <th>Generation</th>
      <th>Legendary</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Ivysaur</td>
      <td>Grass</td>
      <td>Poison</td>
      <td>405</td>
      <td>60</td>
      <td>62</td>
      <td>63</td>
      <td>80</td>
      <td>80</td>
      <td>60</td>
      <td>1</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>Venusaur</td>
      <td>Grass</td>
      <td>Poison</td>
      <td>525</td>
      <td>80</td>
      <td>82</td>
      <td>83</td>
      <td>100</td>
      <td>100</td>
      <td>80</td>
      <td>1</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>VenusaurMega Venusaur</td>
      <td>Grass</td>
      <td>Poison</td>
      <td>625</td>
      <td>80</td>
      <td>100</td>
      <td>123</td>
      <td>122</td>
      <td>120</td>
      <td>80</td>
      <td>1</td>
      <td>False</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5</td>
      <td>Charmeleon</td>
      <td>Fire</td>
      <td>NaN</td>
      <td>405</td>
      <td>58</td>
      <td>64</td>
      <td>58</td>
      <td>80</td>
      <td>65</td>
      <td>80</td>
      <td>1</td>
      <td>False</td>
    </tr>
    <tr>
      <th>6</th>
      <td>6</td>
      <td>Charizard</td>
      <td>Fire</td>
      <td>Flying</td>
      <td>534</td>
      <td>78</td>
      <td>84</td>
      <td>78</td>
      <td>109</td>
      <td>85</td>
      <td>100</td>
      <td>1</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>



### Changing the index


```python
df.index #checking the current index
```




    RangeIndex(start=0, stop=800, step=1)



For changing the index, use method .set_index('column-name'), where column-name is the column you want to set as the index.


```python
df.set_index('Name',inplace=True)
```

The **inplace=True** attribute is used to make the changes directly into df. If we dont use inplace , then the changes made in the dataframe will be temporary and will disappear in the next step


```python
df.index
```




    Index(['Bulbasaur', 'Ivysaur', 'Venusaur', 'VenusaurMega Venusaur',
           'Charmander', 'Charmeleon', 'Charizard', 'CharizardMega Charizard X',
           'CharizardMega Charizard Y', 'Squirtle',
           ...
           'Noibat', 'Noivern', 'Xerneas', 'Yveltal', 'Zygarde50% Forme',
           'Diancie', 'DiancieMega Diancie', 'HoopaHoopa Confined',
           'HoopaHoopa Unbound', 'Volcanion'],
          dtype='object', name='Name', length=800)



### Working with NULL values


```python
df.isnull().sum()
```




    #               0
    Type 1          0
    Type 2        386
    Total           0
    HP              0
    Attack          0
    Defense         0
    Sp. Atk         0
    Sp. Def         0
    Speed           0
    Generation      0
    Legendary       0
    dtype: int64



The above output shows that there are 386 null values in 'Type 2' column while other columns are free from null values.

#### Filling/Droping the NULL values


```python
df.fillna('unknown',inplace=True)
```

The above statement will fill all the NULL values in the dataframe with value as 'unknown'. We can also fill NULL values for only some columns. For that, specify the column name and use .fillna('value') as follows: **df['column-name'].fillna('value')**. For eg:


```python
df['Type 2'].fillna('unknown',inplace=True)
```


```python
df.isnull().sum()
```




    #             0
    Type 1        0
    Type 2        0
    Total         0
    HP            0
    Attack        0
    Defense       0
    Sp. Atk       0
    Sp. Def       0
    Speed         0
    Generation    0
    Legendary     0
    dtype: int64



So now we do not have any NULL values left in the dataframe.


```python
df.dropna(inplace=True)
```

The above statement will drop all NULL values from the dataframe. We can also drop NULL values from a single column also. The syntax will be same as in case of .fillna()

### Counting Frequency

This is basically used to get a count of a value in a column of a dataframe.

**SQL:** Select Type1,count(Type1) 

from df,

Group By Type1;
 
 For Pandas we have .value_counts(),which returns the frequency of each unique value in a dataframe column.


```python
df['Type 1'].value_counts()
```




    Water       112
    Normal       98
    Grass        70
    Bug          69
    Psychic      57
    Fire         52
    Electric     44
    Rock         44
    Ground       32
    Ghost        32
    Dragon       32
    Dark         31
    Poison       28
    Steel        27
    Fighting     27
    Ice          24
    Fairy        17
    Flying        4
    Name: Type 1, dtype: int64



The above output shows the frequency of each distinct Type from the 'Type 1' column.

### GroupBy

It has the same functioning as the GroupBy in SQL. .groupby() typically refers to a process where we’d like to split a dataset into groups, apply some aggregate function, and then combine the groups together.

**SQL:** Select Type1,count(Type1) from df
 ,Group By Type1;


```python
df.groupby('Type 1')['Type 1'].count()
```




    Type 1
    Bug          69
    Dark         31
    Dragon       32
    Electric     44
    Fairy        17
    Fighting     27
    Fire         52
    Flying        4
    Ghost        32
    Grass        70
    Ground       32
    Ice          24
    Normal       98
    Poison       28
    Psychic      57
    Rock         44
    Steel        27
    Water       112
    Name: Type 1, dtype: int64



Okay so the above output is same as the output we got with value_counts(). Lets try grouping by with multiple columns.

**SQL:** Select Type1,count(Type1),

from df,

Group By Type1,Type2;


```python
df.groupby(['Type 1','Type 2'])['Type 1'].count()
```




    Type 1    Type 2  
    Bug       Electric     2
              Fighting     2
              Fire         2
              Flying      14
              Ghost        1
              Grass        6
              Ground       2
              Poison      12
              Rock         3
              Steel        7
              Water        1
              unknown     17
    Dark      Dragon       3
              Fighting     2
              Fire         3
              Flying       5
              Ghost        2
              Ice          2
              Psychic      2
              Steel        2
              unknown     10
    Dragon    Electric     1
              Fairy        1
              Fire         1
              Flying       6
              Ground       5
              Ice          3
              Psychic      4
              unknown     11
    Electric  Dragon       1
                          ..
    Rock      Ground       6
              Ice          2
              Psychic      2
              Steel        3
              Water        6
              unknown      9
    Steel     Dragon       1
              Fairy        3
              Fighting     1
              Flying       1
              Ghost        4
              Ground       2
              Psychic      7
              Rock         3
              unknown      5
    Water     Dark         6
              Dragon       2
              Electric     2
              Fairy        2
              Fighting     3
              Flying       7
              Ghost        2
              Grass        3
              Ground      10
              Ice          3
              Poison       3
              Psychic      5
              Rock         4
              Steel        1
              unknown     59
    Name: Type 1, dtype: int64



So in the above example, we have grouped Type1 and Type2 columns, and have counted the number of Pokemons in each of the subgroups.

### Aggregations

Applying aggregate functions on the dataframe

**SQL:** select max(HP) from df

2)select min(HP) from df

3)select count(*) from df...etc


```python
print('Max HP:',df['HP'].max())
print('Min HP:',df['HP'].min())
print('No. of rows:',df.shape[0])
```

    Max HP: 255
    Min HP: 1
    No. of rows: 800


## Matplotlib

Matplotlib is the most prominent visualisation library in Python. It is used to generate graphs similar to some proprietory software like Matlab,SAS,etc. Some of the most important graphs that maybe useful in understanding the data better are : **Line Charts, Bar Charts, Pie Charts, Scatter Plots, Box-plots** and few others. Matplotlib is very easy to use. The only thing we need to take care is to pass data in the proper format to the Matplotlib functions, and the graphs are rendered automatically.


```python
import matplotlib.pyplot as plt
```


```python
l1=[1,2,3,4]
l2=[4,5,6,7]
plt.plot(l1,l2)
plt.title('Trial Graph')
plt.show()
```


![png](https://github.com/djinit-ai/djinit-ai.github.io/blob/master/images/output_114_0.png?raw=true)


To plot a graph, we just use the .plot() method. The arguments we pass to this method, is the data that we want to plot. The default graph is the line graph.
.title() is used for naming the title. After .plot(), we need to use .show() to render the graph on the screen.


```python
plt.scatter(l1,l2)
plt.show()
```


![png](https://github.com/djinit-ai/djinit-ai.github.io/blob/master/images/output_116_0.png?raw=true)


To plot a different graph, we use the desired graph type, such as .scatter or .bar.


```python
plt.bar(l1,l2,color='r')
plt.xlabel('X-axis')
plt.ylabel('Y-label')
plt.show()
```


![png](https://github.com/djinit-ai/djinit-ai.github.io/blob/master/images/output_118_0.png?raw=true)


.xlabel() and .ylabel() is used to set the name for x-axis and y-axis. All these methods are similar to those in Matlab.

## Plotting some real world data


```python
#Lets say we want to plot this
df['Type 1'].value_counts()
```




    Water       112
    Normal       98
    Grass        70
    Bug          69
    Psychic      57
    Fire         52
    Electric     44
    Rock         44
    Ground       32
    Ghost        32
    Dragon       32
    Dark         31
    Poison       28
    Steel        27
    Fighting     27
    Ice          24
    Fairy        17
    Flying        4
    Name: Type 1, dtype: int64



Now we can simply write .plot() method after the above statement to render the graph.


```python
df['Type 1'].value_counts().plot.bar()
plt.show()
```


![png](https://github.com/djinit-ai/djinit-ai.github.io/blob/master/images/output_123_0.png?raw=true)



```python
df['Type 1'].value_counts().plot.pie()
plt.show()
```


![png](https://github.com/djinit-ai/djinit-ai.github.io/blob/master/images/output_124_0.png?raw=true)


Now the graph type should be selected such that it represents the data very well. Like for a continous distribution, a histogram is suitable but for a time-series, a line plot is suitable. Lets see a continous distribution plot..


```python
df['Total'].plot.hist(color='g')
plt.title('Distribution of Total')
plt.show()
```


![png](https://github.com/djinit-ai/djinit-ai.github.io/blob/master/images/output_126_0.png?raw=true)


The above graph shows the distribution of Total for all the given pokemons.


```python
df['Total'].plot.box()
plt.show()
```


![png](https://github.com/djinit-ai/djinit-ai.github.io/blob/master/images/output_128_0.png?raw=true)


Above is a boxplot, showing the median, 25th quartile, 75th quartile, Max value and Min value.

The methods and functions covered in this blog are more than sufficient to get you started with Machine Learning and AI.

For further reference, prefer using offical docs.
1. [NumPy](http://www.numpy.org)
2. [Pandas](https://pandas.pydata.org)
3. [Matplotlib](https://matplotlib.org/api/pyplot_api.html)

We hope this post was helpful. Feel free to comment in case of doubts and do let us know your feedback. Stay tuned for more!
