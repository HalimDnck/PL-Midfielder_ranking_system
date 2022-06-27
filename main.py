import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb

from sklearn import preprocessing
from sklearn import metrics



data = pd.read_csv("file_data.csv")
import pandas as pd
import numpy as np

def clean_dataset(data):
    assert isinstance(data, pd.DataFrame), "df needs to be a pd.DataFrame"
    data.dropna(inplace=True)
    indices_to_keep = ~data.isin([np.nan, np.inf, -np.inf]).any(1)
    return data[indices_to_keep].astype(np.float64)

clean_dataset(data)

x = data.iloc[:,1:23].values
y = data.iloc[:,23:].values


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test= train_test_split(x,y,test_size=0.33, random_state=0)


from sklearn.preprocessing import StandardScaler

sc= StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)


from sklearn.linear_model import LogisticRegression

logr = LogisticRegression(random_state=0)
logr.fit(X_train,y_train)

y_pred = logr.predict(X_test)


from sklearn.metrics import accuracy_score
#print ("Accuracy : ", accuracy_score(y_test, y_pred))

z= ("2000,20,900,1000,90,18000,4092,382,424,90.1,460,500,92.4,98,100,98,12,16,65,215,15,4,88")
print(logr.predict(z))

