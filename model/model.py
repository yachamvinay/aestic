import pandas as pd
import pickle
data = pd.read_csv("C:\\Users\\HP\\Desktop\\Major project\\new.csv")
from sklearn.model_selection import train_test_split
x = data.drop('Class', axis=1)
y = data['Class']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=101)
x_train.head(10)
from sklearn.ensemble import RandomForestClassifier
RF = RandomForestClassifier(n_estimators=50)
RF = RF.fit(x_train,y_train)
predictions_rf = RF.predict(x_test)

# Loading model to compare the results
model = pickle.load(open('RF_model.pkl','rb'))
