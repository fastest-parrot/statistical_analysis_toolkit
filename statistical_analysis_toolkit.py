#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


def strength_of_evidence(p_value, alpha=0.05):
    assert(p_value == abs(p_value)) # should be positive
    if p_value > 0.1: return 'None/Weak'
    if p_value > 0.05 and p_value <= 0.1: return 'Moderate'
    if p_value > 0.01 and p_value <= 0.5: return 'Strong'
    if p_value <= 0.01: return 'Very Strong'
    return 'WTF'


# In[ ]:

def critical_t(alpha, df):
    return stats.t.ppf(1 - (alpha / 2), df=df)

def analyze_distribution(data, alpha=0.05, h0=0.0, color='b', is_log_data=False):
    #calc confidence level
    cl = 1-alpha
    
    #flatten data into one array
    all_values = data.values.flatten()
    #drop na values 
    all_values = all_values[~np.isnan(all_values)]
    sample_mean = all_values.mean()
    pop_std = all_values.std(ddof=1) #ddof = 1 implies pop std()
     
    df = len(all_values) - len(data.columns) #degrees of freedom = number of observations - number of classes
    
    t_stat = critical_t(alpha, df) #get critical value(s)
    
    #plotting stuff
    hist_fig, hist_ax = plt.subplots(len(data.columns), 1, sharex=True)
    quant_fig, quant_ax = plt.subplots(1, len(data.columns))
    box_fig, box_ax = plt.subplots(1, 1)
    box_fig.suptitle('Box Plot')
    data.boxplot(vert=False)
    hist_fig.suptitle('Group Histograms')
    quant_fig.suptitle('Quantile Plots (Raw)')
    group_summaries = []
      
    if not is_log_data: #if we aren't looking at the log data - we should add it to a Q-Q plot to test for normality
        quantlog_fig, quantlog_ax = plt.subplots(1, len(data.columns))
        quantlog_fig.suptitle('Quantile Plots (Log)')
    
    i = 0
    for x in data:
        #drop any shitty data
        clean_data  = data[x].dropna()
        #run some group stats
        group_se = clean_data.std()/len(clean_data)**0.5
        group_mean = clean_data.mean()
        group_std = clean_data.std()
        
        #plot a histogram
        group_df = len(clean_data) - 1
        group_t_stat = stats.t.ppf(1-(alpha/2),df=group_df)
        hist_ax[i].hist(clean_data, density=True)
        hist_ax[i].set_title(x)
        
        #plt a normal with same std and mean on top of hist
        x_lim = hist_ax[i].get_xlim()
        x_vals = np.linspace(x_lim[0], x_lim[1], 100)
        p = stats.norm.pdf(x_vals, group_mean, group_std)
        hist_ax[i].plot(x_vals, p, 'k', linewidth=2)
        
        #plot quantile plots of data and log x-form where appropriate
        stats.probplot(data[x].dropna(), plot=quant_ax[i])
        quant_ax[i].set_title(x)
        
        if not is_log_data:
            log_data = np.log(clean_data[clean_data != 0])
            quantlog_ax[i].set_title(x)
            stats.probplot(log_data, plot=quantlog_ax[i])

        
        #calc CL of std dev from X2 distribution
        #calc CL of mean from t-dist
        x2 = stats.chi2.ppf((1-alpha/2), df=group_df)
        lower_std = ((group_df*group_std**2)/x2)**0.5
        x2 = stats.chi2.ppf(alpha/2, df=group_df)
        upper_std = ((group_df*group_std**2)/x2)**0.5
        lower_mean = group_mean-group_t_stat*group_se
        upper_mean = group_mean+group_t_stat*group_se
        #lower, upper
        
        group_summaries.append(
            {
                'summary_type': f'Group-{x}',
                'p_value_perm':np.nan,
                'p_value_t_test':np.nan,
                'number_of_obs': len(clean_data),
                'mean':group_mean,
                'std_dev':group_std, 
                'std_err':group_se,
                'min':clean_data.min(),
                'max':clean_data.max(),
                'observed_diff': np.nan,
                f'{cl}_cl_mean':[round(lower_mean, 4), round(upper_mean, 4)],
                f'{cl}_cl_std':[round(lower_std, 4), round(upper_std, 4)],
                't_value': group_t_stat,
                't_value_perm': np.nan,
                'observed_t': np.nan,
                'df':group_df
            }
        )
        i+=1

    
    #lets assume the std err as sigma = s/sqrt(n)
    if len(data.columns) == 1:
        std_err = pop_std/(len(all_values)**0.5)
        mean = all_values.mean()
        chunk_size = int(round(len(all_values)/2, 0))
        observed_t = (mean - h0)/pop_std/len(all_values)**0.5
        ower_mean = mean-t_stat*std_err
        upper_mean = mean+t_stat*std_err
    else: #pooled SD
        group_1 = data.iloc[:,0].dropna()
        group_2 = data.iloc[:,1].dropna()
        group_1_std = group_1.std(ddof=1)
        group_2_std = group_2.std(ddof=1)
        n1 = len(group_1) - 1
        n2 = len(group_2) - 1
        mean = group_1.mean() - group_2.mean()
        pop_std = (((n1*group_1_std**2)+(n2*group_2_std**2))/(n1+n2))**0.5
        chunk_size = len(group_1) if len(group_1) < len(group_2) else len(group_2)
        std_err = pop_std * ((1/len(group_1)) + (1/len(group_2)))**0.5
        observed_t = (mean-h0)/(pop_std *((1/len(group_1)) + (1/len(group_2)))**0.5)
        lower_mean = mean-t_stat*std_err
        upper_mean = mean+t_stat*std_err
        x2 = stats.chi2.ppf((1-alpha/2), df=df)
        lower_std = ((df*pop_std**2)/x2)**0.5
        x2 = stats.chi2.ppf(alpha/2, df=df)
        upper_std = ((df*pop_std**2)/x2)**0.5
    
    p_perm, t_perm, diff = perm_test(data, h0, mean, 1000, len(all_values), chunk_size)
    p_value = 2.0*(1 - stats.t.cdf(observed_t, df=df))
    #TODO: recalc CL based on 1-sided vs 2 sided
    pooled_summary = pd.DataFrame.from_dict(
        {
            'summary_type': 'POOLED',
            'p_value_perm':p_perm,
            'p_value_t_test':p_value,
            'number_of_obs': len(all_values),
            'mean':mean,
            'std_dev':pop_std, 
            'std_err':std_err,
            'min':all_values.min(),
            'max':all_values.max(),
            'observed_diff': round(diff, 4),
            f'{cl}_cl_mean':[round(lower_mean, 4), round(upper_mean, 4)],
            f'{cl}_cl_std':[round(lower_std, 4), round(upper_std, 4)],
            't_value': t_stat,
            't_value_perm': t_perm,
            'observed_t': observed_t,
            'df':df
        }, orient='index').T.set_index(['summary_type'])
    group_summaries = [pd.DataFrame.from_dict(s, orient='index').T.set_index(['summary_type']) for s in group_summaries]
    group_summaries.append(pooled_summary)
    summary = pd.concat(group_summaries)
    
    #plot and shade critical regions
    norm_fig, norm_ax = plt.subplots(1, 1)
    band_size = 5 #could be param
    plot_min = mean - band_size  * pop_std
    plot_max = mean + band_size * pop_std
    x = np.linspace(plot_min, plot_max, 1000)
    iq = stats.norm(h0, pop_std)
    low = h0 - (t_stat*pop_std)
    high = h0 + (t_stat*pop_std)
    normal = iq.pdf(x)
    norm_ax.plot(x, normal, 'r-', lw=3, label=f'Norm x={round(mean, 4)} std={round(pop_std, 4)}') #plot the norm
    norm_ax.plot([mean, mean], [0, normal.max()], 'g-', lw=3, label=f'Sample Mean={round(mean, 4)}') #plot the sample mean
    norm_ax.plot([h0, h0], [0, normal.max()], 'y-', lw=3, label=f'H0={round(h0, 4)}') #plot the pop mean
    norm_ax.plot([low, low], [0, iq.pdf(low)], 'c-*', lw=3) #plot the observed t
    norm_ax.plot([high, high], [0, iq.pdf(high)], 'c-*', lw=3) #plot the observed t
    norm_ax.legend(bbox_to_anchor=(1,-0.05), loc="lower right", 
                bbox_transform=norm_fig.transFigure, ncol=3)
    
    norm_fig.suptitle(f'2 Sided T-Test Alpha={alpha} T-Stat={round(t_stat, 4)} p-value (AUC)={round(p_value, 4)}')
    p_low = x[np.logical_and(x <= low, x >= plot_min)]
    p_high = x[np.logical_and(x >= high, x <= plot_max)]
    norm_ax.fill_between(
        p_low,
        iq.pdf(p_low),
        color=color,
        alpha=0.5,
        linewidth=0,
    )
    norm_ax.fill_between(
        p_high,
        iq.pdf(p_high),
        color=color,
        alpha=0.5,
        linewidth=0,
    )
    return summary


# In[ ]:


def perm_test(data,  expected, observed, number_of_permutations=1000.0, sample_size=30, chunk_size=10):
    #flatten data into one array
    all_values = pd.DataFrame(data.values.flatten())
    #drop na values 
    all_values = all_values[~np.isnan(all_values)]
    assert(sample_size <= len(all_values))
    xbarholder = []
    counter = 0.0
    observed_diff = abs(expected - observed)
    for x in range(1, number_of_permutations):
        scramble = all_values.sample(sample_size)
        random_1 = scramble[0:chunk_size]
        random_2 = scramble[chunk_size:len(all_values)]
        assert(len(random_1) + len(random_2) == sample_size)
        diff = random_1.mean() - random_2.mean()
        xbarholder.append(diff.values[0])
        if abs(diff.values[0]) > observed_diff:
            counter += 1
    p_value = counter/number_of_permutations
    t_value = stats.t.ppf(p_value/2, df=number_of_permutations) #assume two-sided
    permutations = pd.DataFrame(pd.Series(xbarholder))
    #return another table of stats and histogram a-la-sas
    return p_value, t_value, observed_diff

