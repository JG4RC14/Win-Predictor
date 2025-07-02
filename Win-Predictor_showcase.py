#Jorge Garcia Jr

import requests
from bs4 import BeautifulSoup
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.model_selection import GridSearchCV


pd.set_option('expand_frame_repr', False)
pd.set_option('display.max_rows', 20)
pd.set_option('max_colwidth', 4000)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

def data_new(url): #this function fetches all data from the fbref website using requests
    r = requests.get(url)
    soup = BeautifulSoup(r.text, features='html.parser')
    tables = soup.find_all('table')
    all_data = {}
    table_headers = {}
    for i, table in enumerate(tables, 1):
        rows = table.find_all('tr')
        table_data = []
        for row in rows:
            cells = row.find_all(['th', 'td'])
            row_data = [cell.get_text(strip=True) for cell in cells]
            table_data.append(row_data)
        all_data[f'table{i}'] = pd.DataFrame(table_data[1:])
        table_headers[f'table{i}'] = table_data[0]
    return all_data, table_headers

def formatting(all_data, table_headers): #This function cleans my data and adds the appropriate headers
    #this function cleans the data
    all_data['table1'].columns = table_headers['table1']
    table3_headers = [
        "Squad", "# Pl", "Age", "Poss", "MP", "Starts", "Min", "90s", "Gls", "Ast", "G+A",
        "G-PK", "PK", "PKatt", "CrdY", "CrdR", "xG", "npxG", "xAG", "npxG+xAG", "PrgC", "PrgP",
        "Gls/90", "Ast/90", "G+A/90", "G-PK/90", "G+A-PK/90", "xG/90", "xAG/90", "xG+xAG/90", "npxG/90", "npxG+xAG/90"
    ]
    table9_headers = [
        "Squad", "# Pl", "90s", "Gls", "Sh", "SoT", "SoT%", "Sh/90", "SoT/90", "G/Sh",
        "G/SoT", "Dist", "FK", "PK", "PKatt", "xG", "npxG", "npxG/Sh", "G-xG", "np:G-xG"
    ]
    table11_headers = [
        "Squad", "# Pl", "90s", "Cmp", "Att", "Cmp%", "TotDist", "PrgDist",
        "sCmp", "sAtt", "sCmp%", "mCmp", "mAtt", "mCmp%", "lCmp", "lAtt", "lCmp%",
        "Ast", "xAG", "xA", "A-xAG", "KP", "1/3", "PPA", "CrsPA", "PrgP"
    ]
    for i in range(3,12):
        current_table = all_data[f'table{i}']
        current_table.drop(0)
        current_table = current_table[1:]
        current_table = current_table.reset_index(drop=True)
        all_data[f'table{i}'] = current_table
    all_data['table3'].columns = table3_headers # This works for the second table
    all_data['table9'].columns = table9_headers
    all_data['table11'].columns = table11_headers
    return all_data

def print_tables(tables_data, imp_tables): #used to visualize the tables I am most interested in
    for i in range(0,4):
        print(tables_data[f'table{imp_tables[i]}'].head())


def data_grab(tables, imp_tables): # Gets the data we want and puts it all on one table
    for i in imp_tables:
        table = tables[f'table{i}']
        sorted_df = table.sort_values(by=['Squad'])
        sorted_df = sorted_df.reset_index(drop=True)
        tables[f'table{i}'] = sorted_df

    columns_table1 = ['Squad', 'MP', 'W', 'GA', 'xG', 'xGA']
    columns_table2 = ['Poss', 'PK']
    columns_table3 = ['Sh', 'SoT']
    columns_table4 = ['PrgP', '1/3', 'Cmp%', 'Cmp']

    # Assuming you have DataFrames for each table: df_table1, df_table2, and df_table3
    # Selecting desired columns from each DataFrame
    data_table1 = tables['table1'][columns_table1]
    data_table2 = tables['table3'][columns_table2]
    data_table3 = tables['table9'][columns_table3]
    data_table4 = tables['table11'][columns_table4]

    # Combining selected columns into a new DataFrame
    new_table = pd.concat([data_table1, data_table2, data_table3, data_table4], axis=1)
    return new_table

def corr_matrix(df):
    # Select numeric columns excluding 'Squad'
    numeric_columns = df.drop(columns='Squad')

    # Calculate correlation matrix
    correlation_matrix = numeric_columns.corrwith(df['W'])

    # Create DataFrame from correlation matrix
    corr_df = pd.DataFrame(correlation_matrix, columns=['Correlation with Wins'])

    return corr_df

def data_n_target(data):
    #simplifying the dataframe
    data = data[['W','MP', 'xG', 'xGA', 'Poss', 'Sh']]
    #splitting the data
    X = data.drop(columns=['W'])
    y = data['W']
    return X,y

def grid_search_linear_regression(X_train, y_train):
    # Create a pipeline with feature scaling and linear regression
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', LinearRegression())
    ])

    # Define the hyperparameters to tune
    param_grid = {
        'regressor__fit_intercept': [True, False]  # Whether to calculate the intercept for this model
    }

    # Create the grid search object
    grid_search = GridSearchCV(pipeline, param_grid, n_jobs=4, cv=5, scoring='neg_mean_squared_error')

    # Fit the grid search to the data
    grid_search.fit(X_train, y_train)

    # Get the best parameters and best score
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    # Save the best model using joblib
    best_model = grid_search.best_estimator_
    joblib.dump(best_model, '/Users/jorgegarciajr/PycharmProjects/Homework3/.venv/Data/project_linear_regression_model.pkl')

    return best_params, best_score

def data_complete(url):
    imp_tables = [1,3,9,11] #important tables out of the ones scraped off internet
    tables_data, table_headers = data_new(url)  # scraping the data, and their column headers to clean
    tables_data = formatting(tables_data, table_headers)  # cleans the data
    data = data_grab(tables_data, imp_tables)
    return data

def dataframe():
    urls = ['https://fbref.com/en/comps/9/Premier-League-Stats', #Premier League (England)
        'https://fbref.com/en/comps/11/Serie-A-Stats', #Seria A (Italy)
        'https://fbref.com/en/comps/12/La-Liga-Stats', # La Liga (Spain)
        'https://fbref.com/en/comps/13/Ligue-1-Stats', #Ligue 1 (France)
        'https://fbref.com/en/comps/20/Bundesliga-Stats', #Bundesliga (Germany)
    ]
    comp = []
    for url in urls:
        data = data_complete(url)
        comp.append(data)
    combined_df = pd.concat(comp, ignore_index=True)
    return combined_df


def evaluate_model(model, X_test, y_test):
    # Make predictions on the test data
    y_pred = model.predict(X_test)

    # Calculate evaluation metrics
    mse = mean_squared_error(y_test, y_pred)
    r_squared = r2_score(y_test, y_pred)
    # Add more metrics as needed

    # Create a dictionary to store the results
    evaluation_results = {
        'Mean Squared Error (MSE)': mse,
        'R-squared': r_squared,
        # Add more metrics as needed
    }

    return evaluation_results

def plot_predictions(test, y, y_pred):
    # Convert 'W' values to integers
    y_int = y.astype(int)

    # Sort data based on squad names
    test_sorted = test.sort_values(by='Squad')
    squad_names = test_sorted['Squad']
    W_values = y_int[test_sorted.index]
    y_pred_values = y_pred[test_sorted.index]

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(squad_names, W_values, label='W')
    ax.plot(squad_names, y_pred_values, color='red', marker='o', linestyle='-', linewidth=2, label='y_pred')
    ax.set_xlabel('Squad')
    ax.set_ylabel('W')
    ax.set_title('W vs y_pred for each Squad')
    ax.legend()
    ax.set_yticks(range(1, max(W_values) + 1))  # Set y-ticks from 1 to maximum W value
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()

def main():
    df = dataframe()
    #corr = corr_matrix(df)
    #print(df)

    #We will create a model using goals against, possession, and shots
    select_data = df[['W', 'MP', 'xG', 'xGA', 'Poss', 'Sh']]
    x,y = data_n_target(select_data)

    params,score = grid_search_linear_regression(x,y)
    model = joblib.load('/Users/jorgegarciajr/PycharmProjects/Homework3/.venv/Data/project_linear_regression_model.pkl')

    # I will be testing if the model can predict the outcome of the prior Premier League Season
    test = data_complete('https://fbref.com/en/comps/9/2022-2023/2022-2023-Premier-League-Stats')

    X,y = data_n_target(test)
    y_pred = model.predict(X)
    eval = evaluate_model(model,X,y)
    #plot_predictions(test, y, y_pred)


    #print(test)
    #print(X)
    #print(y_pred)
    #print(y)
    print(eval)

if __name__ == '__main__':
    main()