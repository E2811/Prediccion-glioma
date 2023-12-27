import pingouin as pg
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from dython.nominal import associations, cramers_v
import scipy.stats as stats


def correlation_matrix(df: pd.DataFrame, threshold: int, nominal_columns: list):
    """
    Function to perform correlation matrix, using Pearson coefficient for continuous variables
    and V Cramers for categorical variables
    Params
    -------
    df: Dataframe
    threshold: (int) thereshold from which you want to show correlation 
    nominal_columns: (list) list of categorical columns
    """
    df = df.select_dtypes(include=np.number)

    plt.figure(figsize=(30, 10))

    asociations = associations(df, nominal_columns)
    corr = asociations['corr']
    mask = np.triu(np.ones_like(corr, dtype=bool))

    mask |= np.abs(corr) < threshold
    corr = corr[~mask]  # fill in NaN in the non-desired cells

    wanted_cols = np.flatnonzero(np.count_nonzero(~mask, axis=1))
    wanted_rows = np.flatnonzero(np.count_nonzero(~mask, axis=0))
    corr = corr.iloc[wanted_cols, wanted_rows]

    annot = [[f"{val:.2f}" for val in row] for row in corr.to_numpy()]
    heatmap = sns.heatmap(corr, vmin=-1, vmax=1, annot=annot, fmt='', cmap='BrBG')
    heatmap.set_title('Triangle Correlation Heatmap', fontdict={'fontsize': 18}, pad=16)
    plt.show()


def distribution_of_x_over_y(df, x, y, title):
    distribution_of_x_over_y = (
            (df.groupby([x, y])['IDH1'].count() / df.groupby([x])['IDH1'].count()) * 100).reset_index().rename(
        columns={'IDH1': 'Percentage'})
    subtitle_font = 18

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
    sns.histplot(data=df, x=x, hue=y, ax=ax[0], multiple='dodge', stat='count', shrink=.8)
    sns.barplot(data=distribution_of_x_over_y, x=x, y='Percentage', hue=y, ax=ax[1])
    ax[0].legend(labels=['GBM', 'LGG'], title='Grado de Glioma', loc='upper right')
    ax[1].legend_.remove()
    ax[0].set_xlabel('')
    ax[0].set_ylabel('Observations')
    ax[1].set_xlabel('')
    ax[1].set_ylabel('Percentage')
    fig.text(0.5, -0.025, y, ha='center')
    plt.suptitle(title, fontsize=subtitle_font)
    fig.tight_layout()


def chi_square_test(df, x, y):
    """
    Function to perform chi square Test between two columns of a dataframe
    Params
    -------
    df: Dataframe
    x: (str) name column 1
    y: (str) name column 2
    """
    contingency_table = pd.crosstab(df[x], df[y])

    # Perform the Chi-Square test
    chi2_stat, p_value, dof, expected = stats.chi2_contingency(contingency_table)

    # Check the p-value to determine if there is a significant association
    alpha = 0.05  # Significance level
    if p_value < alpha:
        print("Reject the null hypothesis. There is a significant association between variables.")
    else:
        print("Fail to reject the null hypothesis. There is no significant association between variables.")

    print(f"Chi-Square Statistic: {chi2_stat}")
    print(f"P-value: {p_value}")
    print(f"Degrees of Freedom: {dof}")
    print("Expected Frequencies:")
    print(expected)


def ANOVA(df, x, y):
    """
    Function to perform Anova Test between two columns of a dataframe
    Params
    -------
    df: Dataframe
    x: (str)  name column 1
    y: (str)  name column 2
    """
    normality_test = pg.normality(data=df, dv=x, group=y)

    print('Normality test: \n', normality_test)

    homocedasticity_test = pg.homoscedasticity(data=df, dv=x, group=y, method='levene')

    print('Homedasticity test: \n', homocedasticity_test)

    anova_results = pg.anova(data=df, dv=x, between=y, detailed=True)

    p_value = anova_results.loc[0, 'p-unc']
    #     # Perform ANOVA
    #     f_statistic, p_value = stats.f_oneway(df.loc[df[x] == 0, y], df.loc[df[x] == 1, y])

    # Check the p-value to determine if there are significant differences
    alpha = 0.05  # Significance level
    if p_value < alpha:
        print("Reject the null hypothesis. There are significant differences between groups.")
    else:
        print("Fail to reject the null hypothesis. There are no significant differences between groups.")

    print(f"Anova results: {anova_results}")
    print(f"P-value: {p_value}")


def find_outliers_iqr(data):
    """
    Function to search for outliers using IQR
    Params
    -------
    data: Dataframe
    """
    q1 = data.quantile(0.25)
    q3 = data.quantile(0.75)
    IQR = q3 - q1
    lower_bound = q1 - 3 * IQR
    upper_bound = q3 + 3 * IQR
    outliers = data[(data < lower_bound) | (data > upper_bound)]
    return outliers
