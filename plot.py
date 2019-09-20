#import packages
import pandas as pd
import numpy as np

#to plot within notebook
import matplotlib.pyplot as plt
plt.show(block=True)
#get_ipython().magic(u'matplotlib inline')

#setting figure size
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 20,10

#for normalizing data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))

#read the file
df = pd.read_csv('temp_data.csv')

#print the head
df.head()

#setting index as date
df['Date'] = pd.to_datetime(df.Date,unit='s')
df.index = df['Date']

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

#plot
plt.figure(figsize=(16,8))
plt.plot(df['insideTemp'], label='Temperature')

#plt.legend(loc='best')
#plt.show()

# Let's load the required libs.
# We'll be using the Tensorflow backend (default).
from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, Activation, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.utils import shuffle

# Get the raw data values from the pandas data frame.
data_raw = df.values.astype("float32")

# We apply the MinMax scaler from sklearn
# to normalize data in the (0, 1) interval.
scaler = MinMaxScaler(feature_range = (0, 1))
dataset = scaler.fit_transform(data_raw)

# Print a few values.
dataset[1:5]