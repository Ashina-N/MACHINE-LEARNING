import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import streamlit as st

data = pd.read_csv("spam .csv", encoding="latin1")
#print(data.head())
#print(data.shape)
data.drop_duplicates(inplace=True)
#print(data.shape)
#print(data.isnull().sum())

data['Category']=data['Category'].replace(['ham','spam'],['NOTSPAM','SPAM'])
#print(data.head())


mess=data['Message']
cat=data['Category']

(mess_train,mess_test,cat_train,cat_test)=train_test_split(mess,cat,test_size=0.2)

cv=CountVectorizer(stop_words='english')
features=cv.fit_transform(mess_train)



# Train the model
model = MultinomialNB()
model.fit(features, cat_train)



#test our model

features_test=cv.transform(mess_test)
#print(model.score(features_test,cat_test))

#predict data

def predict(message):
    input_message = cv.transform([message]).toarray()  # Transform message into the same feature space
    result = model.predict(input_message)
    return result

st.header("SPAM DETECTION")
#output=predict('Congratulation,you won a lottery')
#print(output)

input_mess=st.text_input("ENTER YOUR MESSAGE HERE")
if st.button("VALIDATE"):
    output=predict(input_mess)
    st.markdown(output)







