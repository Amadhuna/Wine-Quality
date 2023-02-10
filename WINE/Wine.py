import streamlit as st
import pickle 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_option('deprecation.showPyplotGlobalUse', False)

loaded_model=pickle.load(open('thetrain_model.sav','rb'))
load_model=pickle.load(open('scaledmodel.sav','rb'))

data=pd.read_csv("winequality-red.csv")


rad=st.sidebar.radio("Navigation",["Description","Prediction Model","Data Visualization"])

if rad=="Data Visualization":
    st.title("Volatile acidity vs Quality")
    sns.barplot(x='quality',y="volatile acidity",data=data)
    st.pyplot(fig=None)
    
    st.title("Citric acid vs Quality")
    sns.barplot(x='quality',y="citric acid",data=data)
    st.pyplot(fig=None)

    st.title("Heatmap for correlation analysis")
    plt.figure(figsize=(10,8))
    sns.heatmap(data.corr(),annot=True)
    st.pyplot(fig=None)


if rad=="Description":
    st.title(":red[Wine Quality Prediction]")
    st.image("wine.jpg")
    st.header("Contents")
    st.markdown("""
    ## "Wine is the most healthful and most hygienic of beverages"
    ### - Louis Pasteur
 
    """)
    st.markdown("""
    Yes, if you think deep down then you just notice that we are discussing wine, above quote seems to be right because all over the world wine was soo popular among people, and 5% of the population doesn’t know what is wine? sounds good.
    """)
    st.markdown("""
    :wine_glass: :wine_glass: :wine_glass: :wine_glass: :wine_glass: :wine_glass: :wine_glass: :wine_glass: :wine_glass: :wine_glass: :wine_glass: :wine_glass: :wine_glass: :wine_glass: :wine_glass: :wine_glass: :wine_glass: :wine_glass: :wine_glass: :wine_glass: :wine_glass: :wine_glass: :wine_glass: :wine_glass: :wine_glass: :wine_glass: :wine_glass: :wine_glass: :wine_glass: :wine_glass: :wine_glass: :wine_glass: :wine_glass: :wine_glass: :wine_glass: :wine_glass: :wine_glass: :wine_glass:
    """)
    st.markdown("""
    According to experts, the wine is differentiated according to its smell, flavor, and color, but we are not a wine expert to say that wine is good or bad. What will we do then? Here’s the use of Machine Learning comes, yes you are thinking to write we are using machine learning to check wine quality. ML have some techniques that will discuss below: 
    
    ### Wine Prediction Model using Machine Learning Techniques
    To the ML model, we first need to have data for that you don’t need to go anywhere just click here for the wine quality dataset. This dataset was picked up from the Kaggle.
    
    Here is first few rows of the data set.
     """)

    st.table(data.head())
    st.markdown("""
   #### Click the link to download the dataset.
    """)
    st.download_button(
        label="Download data as CSV",
        data="winequality-red.csv",
        file_name='winequality-red.csv',
        mime='text/csv'
    )
    st.markdown("""
    ### Data set Description
    If you download the dataset, you can see that several features will be used to classify the quality of wine, many of them are chemical, so we need to have a basic understanding of such chemicals.\n
    :red[Volatile acidity] :   Volatile acidity is the gaseous acids present in wine.\n
    :red[Fixed acidity] :   Primary fixed acids found in wine are tartaric, succinic, citric, and malic.\n
    :red[Residual sugar] :   Amount of sugar left after fermentation.\n
    :red[Citric acid] :    It is weak organic acid, found in citrus fruits naturally.\n
    :red[Chlorides] :   Amount of salt present in wine.\n
    :red[Free sulfur dioxide] :   So2 is used for prevention of wine by oxidation and microbial spoilage.\n
    :red[Total sulfur dioxide] \n
    :red[pH ]:   In wine pH is used for checking acidity\n
    :red[Density] \n
    :red[Sulphates] :    Added sulfites preserve freshness and protect wine from oxidation, and bacteria.\n
    :red[Alcohol] :   Percent of alcohol present in wine.\n
    """)
 
if rad=="Prediction Model":
    def scaler(values):
        input_data_as_numpy_array=np.asarray(values)
        input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)
        
        values=load_model.transform(input_data_reshaped)
        return values


    def winequality(scaled):
    
        scaled_as_numpy_array=np.asarray(scaled)
        scaled_reshaped=scaled_as_numpy_array.reshape(1,-1)

        x = scaler(scaled_reshaped)

        prediction=loaded_model.predict(x)
        print (prediction)

        if prediction[0]==0:
            return "Not good"
        elif prediction[0]==1:
            return "Good"

    def main():
        st.title("Wine Quality")
        residual_sugar=st.number_input("Residual sugar")
        st.write(residual_sugar)

        fixed_acidity=st.number_input("Fixed acidity")
        st.write(fixed_acidity)

        volatile_acidity=st.number_input("Volatile acidity")
        st.write(volatile_acidity)

        citric_acid=st.number_input("citric acid")
        st.write(citric_acid)

        chlorides=st.number_input("Chlorides")
        st.write(chlorides)

        free_sulfur_dioxide=st.number_input("Free Sulfur Dioxide")
        st.write(free_sulfur_dioxide)

        total_sulfur_dioxide=st.number_input("Total Sulfur Dioxide")
        st.write(total_sulfur_dioxide)

        density=st.number_input("Density")
        st.write(density)

        pH=st.number_input("pH")
        st.write(pH)

        sulphates=st.number_input("Sulphates")
        st.write(sulphates)

        alcohol=st.number_input("Alcohol")
        st.write(alcohol)

        Wine_Quality=''

        if st.button('Wine Test Result'):
            Wine_Quality=winequality([residual_sugar,fixed_acidity,volatile_acidity,citric_acid,chlorides,free_sulfur_dioxide,total_sulfur_dioxide,density,pH,sulphates,alcohol])

        st.success(Wine_Quality)

    if __name__=='__main__':
        main()

