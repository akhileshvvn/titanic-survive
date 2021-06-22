import numpy as np


# To store dataset in a Pandas Dataframe
import io
import pandas as pd
import tensorflow as tf
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
gs = pd.read_csv('gender_submission.csv')
print("TRAIN",train.isna().sum())
print("TEST", test.isna().sum())
print("----GS----",gs.isna().sum())

train1 = pd.read_csv('train.csv')
train1['Sex'] = pd.Categorical(train1['Sex'])
train1['Sex'] = train1.Sex.cat.codes
train1['Embarked'] = pd.Categorical(train1['Embarked'])
train1['Embarked'] = train1.Embarked.cat.codes

test.pop('PassengerId')
test.pop('Name')
test.pop('Ticket')
test.pop('Cabin')
test['Sex'] = pd.Categorical(test['Sex'])
test['Sex'] = test.Sex.cat.codes
test['Embarked'] = pd.Categorical(test['Embarked'])
test['Embarked'] = test.Embarked.cat.codes
print(test.isna().sum())
test['Age'].fillna(value = test['Age'].mean(),inplace=True)
test['Fare'].fillna(value = test['Fare'].mean(),inplace=True)
print("After Correction",test.isna().sum())

gs=gs.pop('Survived')

train1.pop('PassengerId')
train1.pop('Name')
train1.pop('Ticket')
train1.pop('Cabin')
target = train1.pop('Survived')
train1['Age'].fillna(value = train1['Age'].mean(),inplace=True)
train1['Embarked'].fillna(value = train1['Embarked'].mean(),inplace=True)

faredata = list(train1['Fare'])
faredata = list(set(faredata)) 
faredata.sort()

test = test.to_numpy()
train1 = train1.to_numpy()

model = tf.keras.Sequential()
model.add(tf.keras.layers.Input(shape=(7,)))
model.add(tf.keras.layers.Dense(10,activation='relu'))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(14,activation='relu'))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(2,activation=tf.nn.softmax))
model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                metrics=['accuracy'])

epochs = 100
history  = model.fit(train1,target,epochs=epochs, validation_data=(test,gs))






#Start of flask
from flask import Flask, render_template, request

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("home.html",len=len(faredata),faredata = faredata)

@app.route("/test", methods =["GET","POST"])
def test():
    if request.method=="POST":
        age = float(request.form.get('age'))
        pclass = float(request.form.get("pclass"))
        gender = float(request.form.get("gender"))
        siblings = float(request.form.get("siblings"))
        family = float(request.form.get("family"))
        fare = float(request.form.get("fare"))
        port = float(request.form.get("port"))
        lis = [age, pclass, gender, siblings, family, fare, port]
        print(lis)
        lis = np.array(lis)
        predictions = model.predict(lis.reshape(1,7))
        print(predictions)
        return render_template("home2.html",survive=str((predictions.tolist()[0][1])*100)+"%")

if __name__ == "__main__":
    app.run(debug=False,host='0.0.0.0')











'''from flask import Flask, flash, redirect, render_template, \
     request, url_for

app = Flask(__name__)

@app.route('/')
def index():
    return render_template(
        'home2.html',
        data=[{'name':'red'}, {'name':'green'}, {'name':'blue'}])

@app.route("/test" , methods=['GET', 'POST'])
def test():
    select = request.form.get('comp_select')
    return(str(select)) # just to see what select is

if __name__=='__main__':
    app.run(debug=True)'''
