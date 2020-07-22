#!/usr/bin/python
import os
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime


def plot_curve(data_mat, legend, title, folder):
    if not os.path.exists(folder):
        os.mkdir(folder)
    # marker = itertools.cycle(('.', '+', 's', 'o', '*', 'v', 'x', 'p', 'd'))
    # linestyle = itertools.cycle(('-', '--', '-.', ':', ':'))
    fig = plt.figure()
    ax1 = plt.axes()
    for k in range(len(legend)):
        if legend[k] == 'Spain':
            ax1.plot(data_mat[:, k], marker='.', linestyle='-', label=f'{legend[k]}')
        elif legend[k] == 'CAT':
            ax1.plot(data_mat[:, k], marker='*', linestyle='-', label=f'{legend[k]}')
        else:
            ax1.plot(data_mat[:, k], linestyle=':', label=f'{legend[k]}')
    ax1.grid()
    ax1.set_xlabel('Days from estimated outbreak')
    plt.minorticks_on()
    ax1.grid(b=True, axis='both', which='minor', color='#999999', linestyle='--', alpha=0.2)
    ax1.title.set_text(title)
    ax1.legend()
    fig.savefig(os.path.join(folder, f'{title}.png'))
    # plt.show()
    plt.close(fig)


def update_datasets(file, list_of_countries):
    df = pd.read_excel(file, sheet_name='COVID-19-geographic-disbtributi')
    n_samples = (datetime.now().date() - datetime(2019, 12, 31).date()).days + 1
    infections_mat = None
    deaths_mat = None
    idx = np.ones((len(list_of_countries,)), dtype=np.bool)
    for s in range(len(list_of_countries)):
        infs = (df[df['countriesAndTerritories'] == list_of_countries[s]])['cases'].values
        deaths = (df[df['countriesAndTerritories'] == list_of_countries[s]])['deaths'].values
        print(f'{list_of_countries[s]} data points: {infs.size}')
        if n_samples == infs.size:
            if infections_mat is None:
                infections_mat = infs[:, None]
                deaths_mat = deaths[:, None]
            else:
                infections_mat = np.concatenate((infections_mat, infs[:, None]), axis=1)
                deaths_mat = np.concatenate((deaths_mat, deaths[:, None]), axis=1)
        else:
            idx[s] = False
    pd.DataFrame(infections_mat,
                 columns=np.array(list_of_countries)[idx]).to_csv("C:/Users/svra/Desktop/covid-graphs/covid-19-infections_auto.csv",
                                                                  index=False)
    pd.DataFrame(deaths_mat, columns=np.array(list_of_countries)[idx]).to_csv(
        "C:/Users/svra/Desktop/covid-graphs/covid-19-deaths_auto.csv", index=False)
    return idx


def adjust_pol_exp(y, title, folder):
    if not os.path.exists(folder):
        os.mkdir(folder)
    x = np.arange(y.shape[0])
    p5 = np.polyfit(x, y, 5)
    pexp = np.polyfit(x, np.log(0.0001 + y), 1, w=np.sqrt(y))

    x = np.arange(y.shape[0] + 5)
    fig = plt.figure()
    ax1 = plt.axes()
    ax1.plot(x, np.polyval(p5, x), 'b', linestyle=':', label='pol 5')
    ax1.plot(x, np.e ** np.polyval(pexp, x), 'r', linestyle=':', label='exp')
    ax1.plot(x, np.concatenate((y, np.nan * np.ones((5,))), axis=0), 'o', label='True')
    ax1.grid()
    plt.minorticks_on()
    ax1.grid(b=True, axis='both', which='minor', color='#999999', linestyle='--', alpha=0.2)
    ax1.title.set_text(title)
    plt.legend()
    ax1.set_xlabel('Days from estimated outbreak')
    fig.savefig(os.path.join(folder, f'{title}.png'))
    plt.close(fig)
    pred_pol5 = np.polyval(p5, x)[y.shape[0]:]
    pred_exp = np.e ** np.polyval(pexp, x)[y.shape[0]:]
    print(f'{title}: pol5 forecast = {pred_pol5}')
    print(f'{title}: exp forecast = {pred_exp}')


output_folder = f'C:/Users/svra/Desktop/covid-graphs/graphs/{datetime.now().date()}'
raw_file = "C:/Users/svra/Desktop/covid-graphs/COVID-19-geographic-disbtribution-worldwide-2020-03-30.xlsx"
list_of_countries = ['Austria', 'Belgium', 'Canada', 'China', 'Denmark', 'France', 'Germany', 'Iran', 'Italy', 'Japan',
                     'Netherlands', 'Norway', 'South Korea', 'Spain', 'Sweden', 'Switzerland', 'United_Kingdom',
                     'United_States_of_America']

valid_countries_idx = update_datasets(raw_file, list_of_countries)

if not os.path.exists(output_folder):
    os.mkdir(output_folder)

df_infections_ori = pd.read_csv("C:/Users/svra/Desktop/covid-graphs/covid-19-infections_auto.csv")
df_deaths_ori = pd.read_csv("C:/Users/svra/Desktop/covid-graphs/covid-19-deaths_auto.csv")
df_infections_ori = df_infections_ori.fillna(2500)

extra_rows = 11
df_infections = pd.concat((df_infections_ori, pd.DataFrame(np.zeros((extra_rows, len(df_infections_ori.columns))),
                                                           columns=df_infections_ori.columns)),
                          axis=0, ignore_index=True)
df_deaths = pd.concat((df_deaths_ori, pd.DataFrame(np.zeros((extra_rows, len(df_infections.columns))),
                                                   columns=df_infections.columns)),
                      axis=0, ignore_index=True)

df_infections_inv = df_infections.iloc[-1::-1, :]
df_deaths_inv = df_deaths.iloc[-1::-1, :]

# first_day = np.argmax(df_infections_inv.values > 0, axis=0)
# first_day = df_infections_inv.shape[0] - \
#             np.array([16, 11, 21, 74, 15, 11, 16, 16, 22, 42, 29, 14, 15, 42, 17, 15, 16, 14, 21])
magic_first_day_mat = np.array([16, 11, 21, 76 - extra_rows, 11, 18, 16, 23, 26, 44, 14, 15, 42 - extra_rows, 17, 15,
                                16, 14, 21])
first_day = df_infections_inv.shape[0] - magic_first_day_mat
first_day -= ((datetime.now().date() - datetime(2020, 3, 13).date()).days - 1)
first_day = first_day[valid_countries_idx]

infections_inv = np.ones(df_infections_inv.shape) * np.nan
deaths_inv = np.ones(df_deaths_inv.shape) * np.nan
n_samples = deaths_inv.shape[0]

for i in range(len(df_infections_inv.columns)):
    infections_inv[0:n_samples-first_day[i], i] = df_infections_inv.iloc[first_day[i]:, i].values
    deaths_inv[0:n_samples-first_day[i], i] = df_deaths_inv.iloc[first_day[i]:, i].values

infections_cumsum = np.cumsum(infections_inv, axis=0)
deaths_cumsum = np.cumsum(deaths_inv, axis=0)

# Look at significant curves similar to Spain
plot_curve(infections_inv, df_infections_inv.columns, 'Infections per day', os.path.join(output_folder, 'all'))
plot_curve(deaths_inv, df_infections_inv.columns, 'Deaths per day', os.path.join(output_folder, 'all'))
plot_curve(infections_cumsum, df_infections_inv.columns, '#Infections', os.path.join(output_folder, 'all'))
plot_curve(deaths_cumsum, df_infections_inv.columns, '#Deaths', os.path.join(output_folder, 'all'))

# Reduced set
names = ['China', 'France', 'Germany', 'Iran', 'Italy', 'South Korea', 'Spain', 'United_Kingdom', 'United_States_of_America']
idx = np.ones((len(names),), dtype=np.bool)
for i in range(len(names)):
    idx[i] = names[i] in df_infections.columns
names = list(np.array(names)[idx])
cols = []
for s in names:
    cols += [np.where(s == df_infections.columns)[0][0]]

# cols = [3, 6, 7, 8, 9, 13, 14, 17]

infections_inv_red = infections_inv[:, cols]
deaths_inv_red = deaths_inv[:, cols]
infections_cumsum_red = infections_cumsum[:, cols]
deaths_cumsum_red = deaths_cumsum[:, cols]

plot_curve(infections_inv_red, names, 'Infections per day', os.path.join(output_folder, 'reduced'))
plot_curve(deaths_inv_red, names, 'Deaths per day', os.path.join(output_folder, 'reduced'))
plot_curve(infections_cumsum_red, names, '#Infections', os.path.join(output_folder, 'reduced'))
plot_curve(deaths_cumsum_red, names, '#Deaths', os.path.join(output_folder, 'reduced'))

# France vs Italy vs Spain
names2 = ['France', 'Italy', 'Spain', 'Germany', 'China']
# df_new_deaths = pd.read_csv("G:/My Drive/COVID_19/new_deaths.csv", index_col=0)[names2]
# df_new_cases = pd.read_csv("G:/My Drive/COVID_19/new_cases.csv", index_col=0)[names2]
#
# infections_inv_red2 = np.ones(df_new_cases.shape) * np.nan
# deaths_inv_red2 = np.ones(df_new_deaths.shape) * np.nan
# n_samples = df_new_deaths.shape[0]
#
# magic_first_day_mat = np.array([16, 11, 21, 74 - extra_rows, 11, 18, 16, 23, 26, 44, 14, 15, 42 - extra_rows, 17, 15,
#                                 16, 14, 21])
# s_day = df_new_cases.shape[0] - magic_first_day_mat
# s_day -= ((datetime.now().date() - datetime(2020, 3, 13).date()).days - 1)
#
# for i in range(len(df_new_deaths.columns)):
#     infections_inv_red2[0:n_samples-first_day[i], i] = df_new_cases.iloc[first_day[i]:, i].values
#     deaths_inv_red2[0:n_samples-first_day[i], i] = df_new_deaths.iloc[first_day[i]:, i].values
#
# infections_cumsum_red2 = np.cumsum(infections_inv_red2, axis=0)
# deaths_cumsum_red2 = np.cumsum(deaths_inv_red2, axis=0)

idx = np.ones((len(names2),), dtype=np.bool)
for i in range(len(names2)):
    idx[i] = names2[i] in df_infections.columns
names2 = list(np.array(names2)[idx])
cols2 = []
for s in names2:
    cols2 += [np.where(s == df_infections.columns)[0][0]]

infections_inv_red2 = infections_inv[:, cols2]
deaths_inv_red2 = deaths_inv[:, cols2]
infections_cumsum_red2 = infections_cumsum[:, cols2]
deaths_cumsum_red2 = deaths_cumsum[:, cols2]

plot_curve(infections_inv_red2, names2, 'New cases per day', os.path.join(output_folder, 'sp_it_fr'))
plot_curve(deaths_inv_red2, names2, 'Deaths per day', os.path.join(output_folder, 'sp_it_fr'))
plot_curve(infections_cumsum_red2, names2, '#Cases', os.path.join(output_folder, 'sp_it_fr'))
plot_curve(deaths_cumsum_red2, names2, '#Deaths', os.path.join(output_folder, 'sp_it_fr'))

# Spanish forecast
col = np.where('Spain' == np.array(df_infections.columns))[0][0]
y = infections_inv[~np.isnan(infections_inv[:, col]), col]
adjust_pol_exp(y, 'New cases', os.path.join(output_folder, 'forecast'))

y = deaths_inv[~np.isnan(deaths_inv[:, col]), col]
adjust_pol_exp(y, 'New deaths', os.path.join(output_folder, 'forecast'))

y = infections_cumsum[~np.isnan(infections_cumsum[:, col]), col]
adjust_pol_exp(y, 'Total cases', os.path.join(output_folder, 'forecast'))

y = deaths_cumsum[~np.isnan(deaths_cumsum[:, col]), col]
adjust_pol_exp(y, 'Total deaths', os.path.join(output_folder, 'forecast'))

# Catalunya
offset = 8
n1 = infections_inv_red2.shape[0]
df_infections_cat = pd.read_csv("C:/Users/svra/Desktop/covid-graphs/catalunya_infections.csv")
n2 = df_infections_cat.shape[0] - offset + 1
infections_cat_cum = np.concatenate(((df_infections_cat['Cum'])[-offset::-1, None],
                                     np.nan * np.ones((n1 - n2, 1))), axis=0)
infections_cat = np.concatenate(((df_infections_cat['Daily'])[-offset::-1, None],
                                 np.nan * np.ones((n1 - n2, 1))), axis=0)
deaths_cat_cum = np.concatenate(((df_infections_cat['deaths_cum'])[-offset::-1, None],
                                     np.nan * np.ones((n1 - n2, 1))), axis=0)
deaths_cat = np.concatenate(((df_infections_cat['deaths_daily'])[-offset::-1, None],
                                 np.nan * np.ones((n1 - n2, 1))), axis=0)

infections_inv_red3 = np.concatenate((infections_inv_red2, infections_cat), axis=1)
infections_cumsum_red3 = np.concatenate((infections_cumsum_red2, infections_cat_cum), axis=1)
deaths_inv_red3 = np.concatenate((deaths_inv_red2, deaths_cat), axis=1)
deaths_cumsum_red3 = np.concatenate((deaths_cumsum_red2, deaths_cat_cum), axis=1)

death_rate = deaths_cumsum_red3 / infections_cumsum_red3
death_rate[np.isinf(death_rate)] = 0

names3 = names2 + ['CAT']
plot_curve(infections_inv_red3, names3, 'New cases', os.path.join(output_folder, 'cat'))
plot_curve(infections_cumsum_red3, names3, 'Total cases', os.path.join(output_folder, 'cat'))
plot_curve(deaths_inv_red3, names3, 'New deaths', os.path.join(output_folder, 'cat'))
plot_curve(deaths_cumsum_red3, names3, 'Total deaths', os.path.join(output_folder, 'cat'))
plot_curve(death_rate, names3, 'Death rate', os.path.join(output_folder, 'cat'))

y = infections_inv_red3[~np.isnan(infections_inv_red3[:, -1]), -1]
adjust_pol_exp(y, 'New cases', os.path.join(output_folder, 'cat_forecast'))

y = infections_cumsum_red3[~np.isnan(infections_cumsum_red3[:, -1]), -1]
adjust_pol_exp(y, 'Total cases', os.path.join(output_folder, 'cat_forecast'))

y = deaths_inv_red3[~np.isnan(deaths_inv_red3[:, -1]), -1]
adjust_pol_exp(y, 'New deaths', os.path.join(output_folder, 'cat_forecast'))

y = deaths_cumsum_red3[~np.isnan(deaths_cumsum_red3[:, -1]), -1]
adjust_pol_exp(y, 'Total deaths', os.path.join(output_folder, 'cat_forecast'))




