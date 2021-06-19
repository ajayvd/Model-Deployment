import numpy as np

x=np.arange(1,50,2)
len(x)
x_square=np.arange(1,100,4)
len(x_square)
y=np.arange(600,700,4)
len(y)
import pandas as pd
my_dict={'x':x,'x_square':x_square,'output':y}
df=pd.DataFrame(my_dict)

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

x=df.iloc[:,0:2]
y=df.iloc[:,-1]


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

#Fitting model with trainig data
regressor.fit(x, y)

# Saving model to disk
pickle.dump(regressor, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[2, 9]]))