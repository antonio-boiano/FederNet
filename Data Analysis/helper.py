
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
import ipaddress

SERVER_PORT = 8080



def convert_pcap_to_csv(file_path, output_path=None):
    import os
    if output_path is None:
        output_path = file_path.replace(".pcap", ".csv")
        if os.path.isfile(output_path):
            return output_path
    os.system(
        "tshark -r " + file_path + " -Y 'not icmp' -T fields -e frame.time_epoch -e frame.number -e frame.len -e ip.src -e tcp.srcport -e udp.srcport -e ip.dst -e tcp.dstport -e udp.dstport -e tcp.len -e udp.length -e _ws.col.Protocol -E header=y -E separator=',' > " + output_path)
    return output_path

def check_if_file_is_csv(file_path):
    import os
    if file_path.endswith(".csv"):
        if os.path.isfile(file_path):
            return file_path
    elif file_path.endswith(".pcap"):
        if os.path.isfile(file_path):
            return convert_pcap_to_csv(file_path)
    else:
        return False

def check_file_list(file_list):
    out_list = []
    for file in file_list:
        csv_file = check_if_file_is_csv(file)
        if csv_file:
            out_list.append(csv_file)
        else:
            print(f"WARNING: File {file} is not a valid csv or pcap file, jumping it.")
    return out_list


def set_server_port(port):
    global SERVER_PORT
    SERVER_PORT = port

def get_server_port():
    return SERVER_PORT

def load_df(file_path):
    import pandas as pd
    df = pd.read_csv(file_path, sep=',')
    df['frame.time'] = df['frame.time_epoch'] - df['frame.time_epoch'].iloc[0]
    return df


def extract_flows(df, port,server_ip=None):
    import pandas as pd
    import numpy as np
    # df = pd.read_csv('saved_output/ps.csv' , sep=',')

    filtered_df = df.loc[(df['tcp.srcport'] == port) | (df['tcp.dstport'] == port)]
    # filtered_df = filtered_df.dropna()
    if server_ip is None:
        grouped_df = filtered_df.groupby(['tcp.srcport', 'tcp.dstport'])

        unique_names = pd.concat([df['tcp.srcport'], df['tcp.dstport']]).unique()

        unique_names = unique_names[~np.isnan(unique_names)]

        unique_names = unique_names.astype(int)

        unique_names = unique_names[unique_names != port]

        flow_list = []
        for elem in unique_names:
            group_1 = grouped_df.get_group((elem, port))
            group_2 = grouped_df.get_group((port, elem))
            merged_group = pd.concat([group_1, group_2]).sort_values(by=['frame.number'])
            flow_list.append(merged_group)

        return flow_list
    else:

        grouped_df = filtered_df.groupby(['ip.src', 'ip.dst'])

        unique_names_t = pd.concat([df['ip.src'], df['ip.dst']]).unique()

        unique_names_q = [x for x in unique_names_t if str(x) != 'nan']

        unique_names = [x for x in unique_names_q if str(x).startswith('10.0.')]

        unique_names = [x for x in unique_names if str(x) != server_ip]

        #unique_names = unique_names.tolist()

        flow_list = []
        for elem in unique_names:
            group_1 = grouped_df.get_group((str(elem), server_ip))
            group_2 = grouped_df.get_group((server_ip, str(elem)))
            merged_group = pd.concat([group_1, group_2]).sort_values(by=['frame.number'])
            flow_list.append(merged_group)

        return flow_list


# %%
def split_comm_flows(df, treshold):
    store_df = pd.DataFrame()
    list_of_flows = [store_df]
    df_time_diff = df.copy()
    df_time_diff['time.diff'] = df['frame.time'].diff()
    df_time_diff = df_time_diff.fillna(0)
    for index, row in df_time_diff.iterrows():
        if row['time.diff'] < treshold:
            list_of_flows[-1] = pd.concat([list_of_flows[-1], row.to_frame().T])
        else:
            store_df = row.to_frame().T
            list_of_flows.append(store_df)
    return list_of_flows


def extract_feat(df):
    df_res = pd.DataFrame()

    df_res['count'] = [df['frame.number'].count()]

    df_res['frame.time.mean'] = [df['frame.time'].mean()]
    df_res['frame.time.std'] = [df['frame.time'].std()]
    df_res['frame.time.min'] = [df['frame.time'].min()]
    df_res['frame.time.max'] = [df['frame.time'].max()]
    df_res['frame.time.median'] = [df['frame.time'].median()]
    df_res['frame.time.deltat'] = [df['frame.time'].max() - df['frame.time'].min()]
    df_res['frame.time.deltaoff'] = [df['frame.time'].diff().max()]

    df_res['frame.len.mean'] = [df['frame.len'].mean()]
    df_res['frame.len.std'] = [df['frame.len'].std()]
    df_res['frame.len.min'] = [df['frame.len'].min()]
    df_res['frame.len.max'] = [df['frame.len'].max()]
    df_res['frame.len.sum'] = [df['frame.len'].sum()]
    df_res['frame.len.median'] = [df['frame.len'].median()]

    df_res['tcp.len.mean'] = [df['tcp.len'].mean()]
    df_res['tcp.len.std'] = [df['tcp.len'].std()]
    df_res['tcp.len.min'] = [df['tcp.len'].min()]
    df_res['tcp.len.max'] = [df['tcp.len'].max()]
    df_res['tcp.len.sum'] = [df['tcp.len'].sum()]
    df_res['tcp.len.median'] = [df['tcp.len'].median()]

    return df_res


def extract_feat_list(list_of_flows):
    df_res = pd.DataFrame()
    for flow in list_of_flows:
        df_res = pd.concat([df_res, extract_feat(flow)], axis=0, ignore_index=True)
    return df_res

def sum_meas (x,e):

    # Create an array of measurements and errors
    x = np.array(x)
    e = np.array(e)

    # Calculate the weighted mean and uncertainty
    x_m = np.sum(x / e**2) / np.sum(1 / e**2)
    std_m= np.sqrt(1 / np.sum(e**2))

    return x_m,std_m

def extract_mean_std_feat_list(list_of_flows,name):
    mean_list = []
    std_list = []
    for flow in list_of_flows:
        mean_list.append(flow[name].mean())
        std_list.append(flow[name].std())

    x_m,std_m = sum_meas(mean_list, std_list)
    return x_m,std_m
    #return {'mean':x_m,'std':std_m}


# %%
def print_feat(df, name, bins, components, min=None, max=None, log=False, plt_hist=True, fig_name=None, label=None,
               norm_bin=False, show=True, compare_with=None,xlabel=None,ylabel=None):
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.stats import norm

    if fig_name is None:
        fig_name = name
        if log is True:
            fig_name = fig_name + '_log'

    plt.figure(fig_name)
    plt.title(fig_name)

    # Plot the histogram
    if min:
        df_plot = df[df[name] > min]
    elif max:
        df_plot = df[df[name] < max]
    else:
        df_plot = df

    data = np.array(df_plot[name])

    # Fit a mixture of two normal distributions
    from sklearn.mixture import GaussianMixture
    gmm = GaussianMixture(n_components=components)
    gmm.fit(data.reshape(-1, 1))

    if norm_bin is True:
        bin_norm = int((data.max() - data.min()) / bins)
    else:
        if compare_with is not None:
            # Plot the histogram
            df_1 = compare_with
            if min:
                df_plot_1 = df_1[df_1[name] > min]
            elif max:
                df_plot_1 = df_1[df_1[name] < max]
            else:
                df_plot_1 = df_1
            data_1 = np.array(df_plot_1[name])
            bin_norm = np.histogram(np.hstack((data, data_1)), bins=bins)[1]  # get the bin edges
        else:
            bin_norm = bins
    # Plot the histogram of the data

    if plt_hist is True:
        plt.hist(data, bins=bin_norm, density=True, label=label)
        if compare_with is not None:
            plt.hist(data_1, bins=bin_norm, density=True, label=label+'_compared')

    # Overlay the fitted mixture of two normal distributions
    x = np.linspace(data.min(), data.max(), len(data))
    y = np.exp(gmm.score_samples(x.reshape(-1, 1)))
    #plt.plot(x, y, linewidth=2, label=label)
    sns.distplot(data, hist=False, kde=True, rug=True,
                 color='darkblue',
                 kde_kws={'linewidth': 2})

    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)

    if log is True:
        plt.xscale('log')

    if label is not None:
        plt.legend()
    if show is True:
        plt.show()


def print_feat_list(list, name, bins, components, min=None, max=None, log=False, fig_name=None, label=None, plt_hist=True,
               show=True,xlabel=None,ylabel=None):
    def resize_list_minmax (list,name):
        min = list[0][name].min()
        max = list[0][name].max()
        min_len = len(list[0])
        out_list = []
        tmp_list = []
        for df in list:
            if df[name].min() > min:
                min = df[name].min()
            if df[name].max() < max:
                max = df[name].max()
        for df in list:
            df_new = df[df[name] > min]
            df_new = df_new[df_new[name] < max]
            if len(df_new) < min_len:
                min_len = len(df_new)
            tmp_list.append(df_new)
        for df in tmp_list:
            out_list.append(df.head(min_len))

        return out_list

    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.stats import norm

    if fig_name is None:
        fig_name = name+'_list'
        if log is True:
            fig_name = fig_name + '_log'

    plt.figure(fig_name)
    plt.title(fig_name)
    res_list = resize_list_minmax(list,name)
    np_list = []
    for df in res_list:
        # Plot the histogram
        if min:
            df_plot = df[df[name] > min]
        elif max:
            df_plot = df[df[name] < max]
        else:
            df_plot = df

        data = np.array(df_plot[name])

        if plt_hist is True:
            plt.hist(data, bins=bins, density=True, label=label)
            plt_hist = False

        # Fit a mixture of two normal distributions
        from sklearn.mixture import GaussianMixture
        gmm = GaussianMixture(n_components=components)
        gmm.fit(data.reshape(-1, 1))

        kde = sns.kdeplot(data=data)
        x, y = kde.get_lines()[0].get_data()

        # # Overlay the fitted mixture of two normal distributions
        # x = np.linspace(data.min(), data.max(), len(data))
        # y = np.exp(gmm.score_samples(x.reshape(-1, 1)))
        np_list.append(y)


    np_plot = np.array(np_list)

    mu = np_plot.mean(axis=0)
    sigma = np_plot.std(axis=0)

    x = np.linspace(1, len(mu), len(mu))
    X1_plus_sigma = mu + sigma
    X1_minus_sigma = mu - sigma

    plt.figure(fig_name+'tt')
    plt.plot(x, mu, label=label)
    plt.fill_between(x, X1_plus_sigma, X1_minus_sigma, alpha=0.2)

    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)


    if log is True:
        plt.xscale('log')

    if label is not None:
        plt.legend()
    if show is True:
        plt.show()
