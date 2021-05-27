import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
matplotlib.use('TkAgg')

def box_plot(df, folder_name, name):
    plt.boxplot(df.textlen)
    plt.title(name+' review length')
    plt.savefig(f'{folder_name}/{name}_boxplot.png')
    plt.close()


def scatter_neg_pos_freq(term_freq_df, name, folder_name):
    plt.figure(figsize=(10,10))
    ax = sns.regplot(x="negative", y="positive",fit_reg=False, scatter_kws={'alpha':0.5},data=term_freq_df)
    plt.ylabel('Positive Frequency')
    plt.xlabel('Negative Frequency')
    plt.title(f'{name} Negative Frequency vs Positive Frequency')
    plt.savefig(f'{folder_name}/{name}_neg_pos_frequency.png')
    plt.close()


def top_n_normcdf_hmean_words(term_freq_df, n, name, folder_name):
    y_pos = np.arange(n)
    plt.figure(figsize=(10,10))
    plt.bar(y_pos, term_freq_df.sort_values(by='neg_normcdf_hmean', ascending=False)['neg_normcdf_hmean'][:n], align='center', alpha=0.5)
    plt.xticks(y_pos, term_freq_df.sort_values(by='neg_normcdf_hmean', ascending=False)['neg_normcdf_hmean'][:n].index,rotation=15)
    plt.ylabel('Frequency')
    plt.xlabel(f'Top {n} negative tokens')
    plt.title(f'{name} Top {n} tokens in negative texts')
    plt.savefig(f'{folder_name}/{name}_top_n_normcdf_hmean_neg_words')
    plt.close()

    plt.figure(figsize=(10,10))
    plt.bar(y_pos, term_freq_df.sort_values(by='pos_normcdf_hmean', ascending=False)['pos_normcdf_hmean'][:n], align='center', alpha=0.5)
    plt.xticks(y_pos, term_freq_df.sort_values(by='pos_normcdf_hmean', ascending=False)['pos_normcdf_hmean'][:n].index,rotation=15)
    plt.ylabel('Frequency')
    plt.xlabel(f'Top {n} positive tokens')
    plt.title(f'{name} Top {n} tokens in positive texts')
    plt.savefig(f'{folder_name}/{name}_top_{n}_normcdf_hmean_pos_words')
    plt.close()


def plot_neg_pos_normcdf_hmean(term_freq_df, name, folder_name):
    plt.figure(figsize=(10,10))
    ax = sns.regplot(x="neg_normcdf_hmean", y="pos_normcdf_hmean",fit_reg=False, scatter_kws={'alpha':0.5},data=term_freq_df)
    plt.ylabel('Positive Rate and Frequency CDF Harmonic Mean')
    plt.xlabel('Negative Rate and Frequency CDF Harmonic Mean')
    plt.title(f'{name} neg_normcdf_hmean vs pos_normcdf_hmean')
    plt.savefig(f'{folder_name}/{name}_neg_pos_normcdf_hmean.png')
    plt.close()

def create_term_freq_df_plots(term_freq_df, name, folder_name):
    scatter_neg_pos_freq(term_freq_df, name, folder_name)
    top_n_normcdf_hmean_words(term_freq_df, 20, name, folder_name)
    plot_neg_pos_normcdf_hmean(term_freq_df, name, folder_name)
