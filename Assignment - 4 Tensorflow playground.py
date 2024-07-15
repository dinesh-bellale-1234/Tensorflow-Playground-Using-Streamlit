#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions
from keras.models import Model # type: ignore
from keras.layers import Input, Dense # type: ignore
from keras.optimizers import Adam # type: ignore

# Function to load datasets
@st.cache_data
def load_data(file_path):
    return pd.read_csv(file_path, header=None)

def app():
    st.sidebar.title("Configuration for Model")
    num_hidden_layers = st.sidebar.slider("Number of Hidden Layers", 1, 5, 1)
    epochs=st.sidebar.number_input("enter the no of epochs",min_value=1,max_value=1000)
    batch_size=st.sidebar.slider("no of batch size",1,100,1)
    hidden_layers = []
    st.title("Tensorflow Playground")

    for i in range(num_hidden_layers):
        st.sidebar.markdown(f"### Hidden Layer {i+1}")
        units = st.sidebar.slider(f"Number of units for hidden layer {i+1}", 1, 10, 1)
        activation = st.sidebar.selectbox(f"Activation function for hidden layer {i+1}", ['tanh', 'sigmoid', 'linear', 'relu'])
        hidden_layers.append((units, activation))

    dataset_options =  {
        "ushape": r"C:\Users\deepa\Downloads\csv files\Multiple CSV\1.ushape.csv",
        "concerticcir1": r"C:\Users\deepa\Downloads\csv files\Multiple CSV\2.concerticcir1.csv",
        "concertriccir2": r"C:\Users\deepa\Downloads\csv files\Multiple CSV\3.concertriccir2.csv",
        "linearsep": r"C:\Users\deepa\Downloads\csv files\Multiple CSV\4.linearsep.csv",
        "outlier": r"C:\Users\deepa\Downloads\csv files\Multiple CSV\5.outlier.csv",
        "overlap": r"C:\Users\deepa\Downloads\csv files\Multiple CSV\6.overlap.csv",
        "xor": r"C:\Users\deepa\Downloads\csv files\Multiple CSV\7.xor.csv",
        "twospirals": r"C:\Users\deepa\Downloads\csv files\Multiple CSV\8.twospirals.csv"
    }
    dataset_choice = st.sidebar.selectbox("Choose a dataset", list(dataset_options.keys()))
    dataset_path = dataset_options[dataset_choice]

    if st.sidebar.button("Submit"):
        # Load dataset
        dataset = load_data(dataset_path)

        # Prepare data
        X = dataset.iloc[:, :-1].values
        Y = dataset.iloc[:, -1].values.astype(np.int_)

        # Model building
        input_layer = Input(shape=(X.shape[1],))

        def create_layer(units, activation):
            return Dense(units=units, activation=activation)

        x = input_layer
        for units, activation in hidden_layers:
            x = create_layer(units, activation)(x)
        output_layer = Dense(1, activation='sigmoid')(x)
        model = Model(inputs=input_layer, outputs=output_layer)

        # Compile model
        model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

        # Train model
        model.fit(X, Y, epochs=epochs, batch_size=batch_size, verbose=0)

        # Extract output from each neuron in each layer and plot decision regions
        layer_outputs = [layer.output for layer in model.layers[1:] if isinstance(layer, Dense)]
        for layer_num, layer_output in enumerate(layer_outputs):
            num_neurons = layer_output.shape[-1]
            for neuron_num in range(num_neurons):
                neuron_model = Model(inputs=model.input, outputs=layer_output[:, neuron_num])
                fig, ax = plt.subplots()
                plot_decision_regions(X, Y, clf=neuron_model, ax=ax)
                st.pyplot(fig)

if __name__ == "__main__":
    app()

