# general
import io

# data
import numpy as np
import pandas as pd

# machine learning
import keras

# data visualization
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import seaborn as sns

# @title
chicago_taxi_dataset = pd.read_csv(
    "https://download.mlcc.google.com/mledu-datasets/chicago_taxi_train.csv")

# Updates dataframe to use specific columns.
training_df = chicago_taxi_dataset[[
    'TRIP_MILES', 'TRIP_SECONDS', 'FARE', 'COMPANY', 'PAYMENT_TYPE', 'TIP_RATE']]

print('Read dataset completed successfully.')
print('Total number of rows: {0}\n\n'.format(len(training_df.index)))
training_df.head(200)

# @title Code - View dataset statistics

print('Total number of rows: {0}\n\n'.format(len(training_df.index)))
training_df.describe(include='all')

# @title Double-click or run to view answers about dataset statistics

answer = '''
What is the maximum fare? 				              Answer: $159.25
What is the mean distance across all trips? 		  Answer: 8.2895 miles
How many cab companies are in the dataset? 		      Answer: 31
What is the most frequent payment type? 		      Answer: Credit Card
Are any features missing data? 				          Answer: No
'''

# You should be able to find the answers to the questions about the dataset
# by inspecting the table output after running the DataFrame describe method.
#
# Run this code cell to verify your answers.

# What is the maximum fare?
max_fare = training_df['FARE'].max()
print("What is the maximum fare? \t\t\t\tAnswer: ${fare:.2f}".format(
    fare=max_fare))

# What is the mean distance across all trips?
mean_distance = training_df['TRIP_MILES'].mean()
print("What is the mean distance across all trips? \t\tAnswer: {mean:.4f} miles".format(
    mean=mean_distance))

# How many cab companies are in the dataset?
num_unique_companies = training_df['COMPANY'].nunique()
print("How many cab companies are in the dataset? \t\tAnswer: {number}".format(
    number=num_unique_companies))

# What is the most frequent payment type?
most_freq_payment_type = training_df['PAYMENT_TYPE'].value_counts().idxmax()
print("What is the most frequent payment type? \t\tAnswer: {type}".format(
    type=most_freq_payment_type))

# Are any features missing data?
missing_values = training_df.isnull().sum().sum()
print("Are any features missing data? \t\t\t\tAnswer:",
      "No" if missing_values == 0 else "Yes")

# @title Code - View correlation matrix
training_df.corr(numeric_only=True)

# @title Double-click to view answers about the correlation matrix

# Which feature correlates most strongly to the label FARE?
# ---------------------------------------------------------
answer = '''
The feature with the strongest correlation to the FARE is TRIP_MILES.
As you might expect, TRIP_MILES looks like a good feature to start with to train
the model. Also, notice that the feature TRIP_SECONDS has a strong correlation
with fare too.
'''
print(answer)


# Which feature correlates least strongly to the label FARE?
# -----------------------------------------------------------
answer = '''The feature with the weakest correlation to the FARE is TIP_RATE.'''
print(answer)

# @title Define plotting functions


def make_plots(df, feature_names, label_name, model_output, sample_size=200):

    random_sample = df.sample(n=sample_size).copy()
    random_sample.reset_index()
    weights, bias, epochs, rmse = model_output

    is_2d_plot = len(feature_names) == 1
    model_plot_type = "scatter" if is_2d_plot else "surface"
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=("Loss Curve", "Model Plot"),
                        specs=[[{"type": "scatter"}, {"type": model_plot_type}]])

    plot_data(random_sample, feature_names, label_name, fig)
    plot_model(random_sample, feature_names, weights, bias, fig)
    plot_loss_curve(epochs, rmse, fig)

    fig.show()
    return


def plot_loss_curve(epochs, rmse, fig):
    curve = px.line(x=epochs, y=rmse)
    curve.update_traces(line_color='#ff0000', line_width=3)

    fig.append_trace(curve.data[0], row=1, col=1)
    fig.update_xaxes(title_text="Epoch", row=1, col=1)
    fig.update_yaxes(title_text="Root Mean Squared Error",
                     row=1, col=1, range=[rmse.min()*0.8, rmse.max()])

    return


def plot_data(df, features, label, fig):
    if len(features) == 1:
        scatter = px.scatter(df, x=features[0], y=label)
    else:
        scatter = px.scatter_3d(df, x=features[0], y=features[1], z=label)

    fig.append_trace(scatter.data[0], row=1, col=2)
    if len(features) == 1:
        fig.update_xaxes(title_text=features[0], row=1, col=2)
        fig.update_yaxes(title_text=label, row=1, col=2)
    else:
        fig.update_layout(scene1=dict(
            xaxis_title=features[0], yaxis_title=features[1], zaxis_title=label))

    return
