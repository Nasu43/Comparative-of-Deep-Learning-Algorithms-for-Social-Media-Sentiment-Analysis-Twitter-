import streamlit as st
import pandas as pd
import numpy as np
import re
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import os

# Load the tokenizer
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Load the dataframe
df = pd.read_csv('Reddit_Data_nas.csv')
df.dropna(axis=0, inplace=True)
df['category'] = df['category'].map({-1.0: 'Negative', 0.0: 'Neutral', 1.0: 'Positive'})

# Preprocess text data using the loaded tokenizer
max_len = 50
X = tokenizer.texts_to_sequences(df['clean_text'])
X = pad_sequences(X, padding='post', maxlen=max_len)

# Encode target labels
y = pd.get_dummies(df['category'])

# Split data into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=1)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=1)

# Function to evaluate and print metrics
def evaluate_model(model_path, X_test, y_test, model_name):
    if os.path.exists(model_path):
        model = tf.keras.models.load_model(model_path)
        loss, accuracy, precision, recall = model.evaluate(X_test, y_test, verbose=1)
        return accuracy, precision, recall, model
    else:
        st.error(f"Model file '{model_path}' not found.")
        return None, None, None, None

# Function to plot confusion matrix
def plot_confusion_matrix(model, X_test, y_test, labels):
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_test.to_numpy(), axis=1)
    cm = confusion_matrix(y_true, y_pred_classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    
    fig, ax = plt.subplots()  # Create a new figure and axis
    disp.plot(cmap=plt.cm.Blues, ax=ax)
    st.pyplot(fig)  # Pass the figure to st.pyplot()

# Evaluate all models and store their metrics
acc_blstm, pre_blstm, rec_blstm, model_blstm = evaluate_model('blstmm_model.h5', X_test, y_test, 'B-Directional LSTM')
acc_gru, pre_gru, rec_gru, model_gru = evaluate_model('grru_model.h5', X_test, y_test, 'GRU')
acc_lstm, pre_lstm, rec_lstm, model_lstm = evaluate_model('lstmm_model.h5', X_test, y_test, 'LSTM')

# Create DataFrame for evaluation metrics
eva_p = ("Accuracy", "Precision", "Recall")
df1 = list(zip([acc_blstm, pre_blstm, rec_blstm], [acc_gru, pre_gru, rec_gru], [acc_lstm, pre_lstm, rec_lstm], eva_p))
df2 = pd.DataFrame(df1, columns=['b-direc_lstm_model', 'gru_model', 'lstm_model', 'evaluation'])
df2.set_index("evaluation", inplace=True)

# Create DataFrame for accuracy comparison
mole = ('b-direc_lstm_model', 'gru_model', 'lstm_model', 'evaluation')
acc_model = (acc_blstm, acc_gru, acc_lstm)
df3 = list(zip(acc_model, mole))
acm_df = pd.DataFrame(df3, columns=['Accuracy', 'evaluation'])
acm_df.set_index("evaluation", inplace=True)

# Create DataFrame for precision and recall comparison
pr_df = pd.DataFrame({
    'Model': ['B-Directional LSTM', 'GRU', 'LSTM'],
    'Accuracy': [acc_blstm, acc_gru, acc_lstm],
    'Precision': [pre_blstm, pre_gru, pre_lstm],
    'Recall': [rec_blstm, rec_gru, rec_lstm]
})

# Determine the best model
best_model = max([(acc_blstm, 'B-Directional LSTM'), (acc_gru, 'GRU'), (acc_lstm, 'LSTM')])[1]

# Streamlit app layout
st.title('Comparative of Deep Learning Algorithms for Social Media Sentiment Analysis (Twitter)')

option = st.selectbox('Select Option:', ['Result', 'Confusion Matrix', 'Result Table'])

if option == 'Result':
    st.subheader('Model Accuracy, Precision, and Recall Comparison')
    
    # Line graph for Accuracy, Precision, and Recall
    fig, ax = plt.subplots()
    pr_df.plot(x='Model', y=['Accuracy', 'Precision', 'Recall'], kind='line', marker='o', ax=ax)
    plt.title('Model Accuracy, Precision, and Recall Comparison')
    plt.ylabel('Metric Value')
    plt.xlabel('Model')
    plt.xticks(rotation=0)
    plt.grid(True)
    plt.legend(title='Metrics')
    st.pyplot(fig)  # Pass the figure to st.pyplot()
    
    st.subheader('Model Accuracy Comparison')
    
    # Line graph for Accuracy only
    fig, ax = plt.subplots()
    acm_df.plot(kind='line', marker='o', ax=ax)
    plt.title('Model Accuracy Comparison')
    plt.ylabel('Accuracy')
    plt.xlabel('Model')
    plt.xticks(rotation=0)
    plt.grid(True)
    plt.legend(title='Models')
    st.pyplot(fig)  # Pass the figure to st.pyplot()
    
    st.subheader('Best Model')
    st.write(f"The best model based on accuracy is: {best_model}")

elif option == 'Confusion Matrix':
    st.header('Confusion Matrices')
    if model_blstm:
        st.subheader('Bi-Directional LSTM Model')
        plot_confusion_matrix(model_blstm, X_test, y_test, ['Negative', 'Neutral', 'Positive'])

    if model_gru:
        st.subheader('GRU Model')
        plot_confusion_matrix(model_gru, X_test, y_test, ['Negative', 'Neutral', 'Positive'])

    if model_lstm:
        st.subheader('LSTM Model')
        plot_confusion_matrix(model_lstm, X_test, y_test, ['Negative', 'Neutral', 'Positive'])

elif option == 'Result Table':
    st.subheader('Comparison Accuracy Table')
    st.write(df2)
    
    st.subheader('Accuracy Only Table')
    st.write(acm_df)
    
    st.subheader('Best Model')
    st.write(f"The best model based on accuracy is: {best_model}")
