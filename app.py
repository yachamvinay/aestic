from flask import Flask, render_template, request
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler
app = Flask(__name__)
data = pd.read_csv("F:\\vinay files\\HCL\\New folder\\projects\\aestic\\new.csv")
from sklearn.model_selection import train_test_split
x = data.drop('Class', axis=1)
y = data['Class']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=101)
x_train.head(10)
from sklearn.ensemble import RandomForestClassifier
RF = RandomForestClassifier(n_estimators=50)
RF = RF.fit(x_train,y_train)

standard_to = StandardScaler()
model = pickle.load(open('random forest.pkl', 'rb'))
@app.route('/',methods=['GET'])
def Home():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        A1_Score = request.form['A1_Score']
        if(A1_Score=='NO'):
            A1_Score=0
        else:
            A1_Score=1
        A2_Score = request.form['A2_Score']
        if(A2_Score=='NO'):
            A2_Score=0 
        else:
                 A2_Score=1
        A3_Score = request.form['A3_Score']
        if(A3_Score=='NO'):
                 A3_Score=0
        else:
                 A3_Score=1	
        A4_Score = request.form['A4_Score']
        if(A4_Score=='NO'):
                 A4_Score=0
        else:
                 A4_Score=1
        A5_Score = request.form['A5_Score']
        if(A5_Score=='NO'):
                 A5_Score=0
        else:
                 A5_Score=1	
        A6_Score = request.form['A6_Score']
        if(A6_Score=='NO'):
                 A6_Score=0
        else:
                 A6_Score=1
        A7_Score = request.form['A7_Score']
        if(A7_Score=='NO'):
                 A7_Score=0
        else:
                 A7_Score=1	
        A8_Score = request.form['A8_Score']
        if(A8_Score=='NO'):
                 A8_Score=0
        else:
                 A8_Score=1	
        A9_Score = request.form['A9_Score']
        if(A9_Score=='NO'):
                 A9_Score=0
        else:
                 A9_Score=1
        A10_Score = request.form['A10_Score']
        if(A10_Score=='NO'):
                 A10_Score=0
        else:
                 A10_Score=1	
        age = float(request.form['age'])
        gender= request.form['gender']
        if(gender=='Female'):
                 gender=1
        else:
                 gender=0	
        jaundice= request.form['jaundice']
        if(jaundice=='YES'):
                 jaundice=1
        else:
                 jaundice=0	
        autism= request.form['autism']
        if(autism=='YES'):
                 autism=1
        else:
                autism=0	
        relation= request.form['relation']
        if(relation=='Self'):
                  relation=0
        elif(relation=='Parent'):
                  relation=1
        elif(relation=='Others'):
                  relation=2
        elif(relation=='Health care professional'):
                  relation=3
        else:
                 relation=4
    
        output=model.predict([[A1_Score,A2_Score,A3_Score,A4_Score,A5_Score,A6_Score,A7_Score,A8_Score,A9_Score,
                               A10_Score,age,gender,jaundice,autism,relation]])
   
    if output ==1:
         return render_template('index.html',prediction_text="Person is Autistic")
    elif output ==0:
          return render_template('index.html',prediction_text="Person is Non-Autistic")

    else:
      return render_template('index.html')


if __name__=="__main__":
    app.run(debug=True)








