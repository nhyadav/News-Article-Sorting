import yaml
import numpy as numpy
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker

# read the dataste
# plot the dataset
# save the plot of dataset

def train_count_plot(data, path):
    figure  = plt.figure(figsize=(10, 8))
    sns.countplot(x = 'Category',  hue = 'Category', data = data)
    plt.savefig(path)
    return

def train_frequency_plot(data, path):
    # percentage with frequency
    plt.figure(figsize=(10, 8))
    ax = sns.countplot(x = 'Category',  hue = 'Category', data = data)
    ax2 = ax.twinx()
    ax2.yaxis.tick_left()
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    ax2.yaxis.set_label_position('left')
    ax2.set_ylabel('Frequecy [%]')
    ncount = len(data)

    for p in ax.patches:
        x=p.get_bbox().get_points()[:,0]
        y=p.get_bbox().get_points()[1,1]
        ax.annotate('{:.1f}%'.format(100*y/ncount),(x.mean(),y), ha='center', va='bottom')
        ax.yaxis.set_major_locator(ticker.LinearLocator(11))
        ax2.set_ylim(0,100)
        ax.set_ylim(0, ncount)

        ax2.yaxis.set_major_locator(ticker.MultipleLocator(10))
        ax2.grid(None)
    plt.savefig(path)
    return

def test_count_plot(data, path):
    figure  = plt.figure(figsize=(10, 8))
    sns.countplot(x = 'Category',  hue = 'Category', data = data)
    plt.savefig(path)
    return

def test_frequency_plot(data, path):
    # percentage with frequency
    plt.figure(figsize=(10, 8))
    ax = sns.countplot(x = 'Category',  hue = 'Category', data = data)
    ax2 = ax.twinx()
    ax2.yaxis.tick_left()
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    ax2.yaxis.set_label_position('left')
    ax2.set_ylabel('Frequecy [%]')
    ncount = len(data)

    for p in ax.patches:
        x=p.get_bbox().get_points()[:,0]
        y=p.get_bbox().get_points()[1,1]
        ax.annotate('{:.1f}%'.format(100*y/ncount),(x.mean(),y), ha='center', va='bottom')
        ax.yaxis.set_major_locator(ticker.LinearLocator(11))
        ax2.set_ylim(0,100)
        ax.set_ylim(0, ncount)

        ax2.yaxis.set_major_locator(ticker.MultipleLocator(10))
        ax2.grid(None)
    plt.savefig(path)
    return


def get_parameter(path):
    with open(path, encoding='utf-8') as p:
        yaml_parameter = yaml.safe_load(p)
    return yaml_parameter

def perform_eda(path):
    params = get_parameter(path)
    train_path = params['data_source']['train']
    test_path = params['data_source']['sample_submission']
    eda_with_count = params['plot']['eda_count_plot']
    eda_with_frequency = params['plot']['eda_freq_plot']
    test_eda_with_count = params['plot']['testdata_eda_count_plot']
    test_eda_with_frequency = params['plot']['testdata_eda_freq_plot']
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    train_count_plot(train_data, eda_with_count)
    train_frequency_plot(train_data, eda_with_frequency)
    test_count_plot(test_data, test_eda_with_count)
    test_frequency_plot(test_data, test_eda_with_frequency)
    print(train_data.head())
    print(test_data.head())

if __name__ == "__main__":
    path = "E:\\DataScience_internship_with_ineuron\\newsarticalesorting\\newsarticlesorting\\params.yaml"
    perform_eda(path)