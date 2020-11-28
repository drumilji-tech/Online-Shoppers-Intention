import streamlit as st
st.set_option('deprecation.showPyplotGlobalUse', False)
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC 
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_roc_curve,plot_confusion_matrix,plot_precision_recall_curve
from sklearn.metrics import precision_score,recall_score

def main():
    st.title("Online Shoppers Purchasing Intention")
    st.sidebar.title("Machine Learning and its specifications")
    
    st.markdown("Online shopping is a huge and growing form of purchasing and represents a huge portion of B2C (Business to Customer) revenue. 69% of Americans have shopped online at some point (1), with an average revenue of $1804 per online shopper(2). 36% of Americans shop online at least once per month! Learning how and when shoppers will research and purchase goods online is important to businesses as they can use customer behavior insights to target advertising, marketing, and deals to potential customers to further increase their sales and revenue..")
   
    
    st.markdown("So, Let's evaluate our model with different Evaluation metrices as the metrices provide us how effective our model is.")
    st.sidebar.markdown("Let\'s do it")
    data = pd.read_csv('https://raw.githubusercontent.com/drumilji-tech/Online-Shoppers-Intention/main/online_shoppers_intention.csv')
    
    @st.cache(persist=True)
    def split(data):
        data1 = pd.get_dummies(data)
        le = LabelEncoder()
        data['Revenue'] = le.fit_transform(data['Revenue'])
        x = data1
        # removing the target column revenue from x
        x = x.drop(['Revenue'], axis = 1)
        y = data[['Revenue']]
        x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.35,random_state=101)
        return x_train,x_test,y_train,y_test
     
    def plot_metrics(metrics_list):
        if 'Confusion Matrix' in metrics_list:
            st.subheader("Confusion Matrix")
            plot_confusion_matrix(Model,x_test,y_test,display_labels=class_names)
            st.pyplot()
        
        if 'ROC Curve' in metrics_list:
            st.subheader("ROC Curve")
            plot_roc_curve(Model,x_test,y_test)
            st.pyplot()
            
        if 'Precision Recall Curve' in metrics_list:
            st.subheader("Precision Recall Curve")
            plot_precision_recall_curve(Model,x_test,y_test)
            st.pyplot()
            
            
     
    x_train,x_test,y_train,y_test = split(data)
    class_names = ['Default','Not Default']
    
    st.sidebar.subheader('Choose Model')
    Model = st.sidebar.selectbox("Model",('Logistic Regression','Decision Tree','Random Forest'))
    
    
    if Model == "Logistic Regression":
        st.sidebar.subheader("Model Hyperparameters")
        C = st.sidebar.number_input("C(Regularization parameter)",0.01,10.0,step=0.01,key='C_LR')
        max_iter = st.sidebar.slider("Maximum Number of Iterations",100,500,key='max_iter')
        metrics = st.sidebar.selectbox("Which metrics to plot?",('ROC Curve','Precision Recall Curve','Confusion Matrix'))
        
        if st.sidebar.button("Classify",key='classify'):
            st.subheader("Logistic Regression Results")
            Model = LogisticRegression(C=C,max_iter=max_iter)
            Model.fit(x_train,y_train)
            accuracy = Model.score(x_test,y_test)
            y_pred = Model.predict(x_test)
            st.write("Accuracy:",accuracy.round(2))
            st.write("Precision:",precision_score(y_test,y_pred).round(2))
            st.write("Recall:",recall_score(y_test,y_pred).round(2))
            plot_metrics(metrics)
            
           
   
            
    if Model == "Random Forest":
        st.sidebar.subheader("Model Hyperparameters")
        n_estimators = st.sidebar.number_input("The number of trees in the forest",100,5000,step=10,key='n_est')
        max_depth = st.sidebar.number_input("The maximum depth of the tree",1,20,step=1,key='max_depth')
        bootstrap = st.sidebar.radio("Bootstrap samples when building trees",('True','False'),key='bootstrap')
        metrics = st.sidebar.selectbox("Which metrics to plot?",('ROC Curve','Precision Recall Curve','Confusion Matrix'),key='1')
        
        if st.sidebar.button("Classify",key='class'):
            st.subheader("Random Forest Result")
            Model = RandomForestClassifier(n_estimators=n_estimators,max_depth=max_depth,bootstrap=bootstrap)
            Model.fit(x_train,y_train)
            accuracy = Model.score(x_test,y_test)
            y_pred = Model.predict(x_test)
            st.write("Accuracy:",accuracy.round(2))
            st.write("Precision:",precision_score(y_test,y_pred).round(2))
            st.write("Recall:",recall_score(y_test,y_pred).round(2))
            plot_metrics(metrics)
    
  
            
    if Model == "Decision Tree":
        st.sidebar.subheader("Model Hyperparameters")
        criterion= st.sidebar.radio('Criterion(measures the quality of split)', ('gini', 'entropy'), key='criterion')
        splitter = st.sidebar.radio('Splitter (How to split at each node?)', ('best','random'), key='splitter')
        metrics = st.sidebar.selectbox("Which metrics to plot?",('ROC Curve','Precision Recall Curve','Confusion Matrix'))
        
        if st.sidebar.button("Classify",key='class'):
            st.subheader('Decision Tree Results')
            model = DecisionTreeClassifier()
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)
            
            st.write("Precision:",precision_score(y_test,y_pred).round(2))
            st.write("Recall:",recall_score(y_test,y_pred).round(2))
            plot_metrics(metrics)
            
           
    
     
    
    if st.sidebar.checkbox("Show Raw Data",False):
        st.subheader("Shoppers-Intention Data")
        st.write(data)




if __name__ == '__main__':
    main()
