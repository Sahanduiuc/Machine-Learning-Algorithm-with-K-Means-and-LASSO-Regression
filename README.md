# Machine-Learning-Algorithm-with-K-Means-and-LASSO-Regression

This strategy dynamically chooses the top performing stocks (by Sharpe ratio) of each cluster, then uses an L1 regularization term (LASSO) to penalize the portfolio weights and achieve an all-long portfolio, with quarterly rebalancing, or at least, that's the goal here

Included in the source code are comments explaining step-by-step how the algorithm operates. 

More clusters results in lower beta (expected that)
Shifting market entry point by 17 days from month's start to try and coincide with the release of earnings reports resulted in -300% returns (unexpected, will have to test further to evaluate the best entry point)
Cluster size of 30 seemed to provide the best risk/return tradeoff
Kept penalization parameter at 2 for all tests
This algo would not survive the 2008-2009 financial crisis

Notes:
<ul>
<li>Exclude single stock clusters</li>
<li>Maximum allocation to single stock limited to 30%</li>
</ul>

<h1>The Algorithm</h1>

<pre>
<code>
"""
Stocks are selected using k-means clustering to get a diversified portfolio. 
Uses L1 regularization term to sparsify the portfolio weights and achieve an all-long portfolio.

"""
import pandas as pd
import numpy as np
import cvxpy as cvx
import math
from quantopian.pipeline.data import morningstar
from quantopian.pipeline.filters.morningstar import Q1500US
from quantopian.pipeline import Pipeline
from quantopian.algorithm import attach_pipeline, pipeline_output
from sklearn.cluster import KMeans



def initialize(context):
    
    # Initialize quarterly counter
    context.iMonths=0
    context.NMonths=3  # <- Sets quarterly rebalancing counter
    
    # # Set the slippage model
    set_slippage(slippage.VolumeShareSlippage(volume_limit=0.025, price_impact=0.1))
    
    # Set the commission model (Interactive Brokers Commission)
    set_commission(commission.PerShare(cost=0.001, min_trade_cost=0))
    
    # Attach pipeline
    my_pipe = make_pipeline()
    attach_pipeline(my_pipe, 'fundamental_line')
    
    schedule_function(schedule,
                      date_rules.month_start(),
                      time_rules.market_open(hours=0, minutes=5))
    
    # Record tracking variables at the end of each day.
    schedule_function(record_vars,
                      date_rules.every_day(),
                      time_rules.market_close(minutes=1))
    
    
       
def before_trading_start(context, data):
    
    # Pipeline_output returns the constructed dataframe.
    context.output = pipeline_output('fundamental_line')
       
def make_pipeline():

    # Get fundamental data from liquid universe for easier trades
    my_assets = morningstar.balance_sheet.total_assets.latest
    my_revenues = morningstar.income_statement.total_revenue.latest
    my_income = morningstar.cash_flow_statement.net_income.latest

    pipe = Pipeline(
              columns={
                'total_assets': my_assets,
                'total_revenues': my_revenues,
                'net_income': my_income,
              }, screen = Q1500US()
          )
    
    return pipe

def schedule(context, data):
    cluster_analysis(context, data)
    calculate_weights(context, data)
    place_order(context, data)
                
def cluster_analysis(context, data):
    # Get list of equities that made it through the pipeline
    context.output['return_on_asset'] = context.output.net_income / context.output.total_assets
    context.output['asset_turnover'] = context.output.total_revenues / context.output.total_assets
    context.output['feature'] = 0.5 * context.output.return_on_asset + 0.5 * context.output.asset_turnover
    context.output = context.output.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how='any')
    
    equity_list = context.output.index
    
    # Run k-means on feature matrix
    n_clust = 15
    alg = KMeans(n_clusters=n_clust, random_state=10, n_jobs=1)
    cluster_results = alg.fit_predict(context.output[['feature']])
    context.output['cluster'] = pd.Series(cluster_results, index=equity_list)
    
    # Remove single stock cluster
    cluster_count = context.output.cluster.groupby(context.output.cluster).count()
    context.output = context.output[context.output.cluster.isin(cluster_count[cluster_count > 1].index)]
    
    print('cluster result', cluster_count)
    
    # Get rolling window of past prices and compute returns
    context.prices = data.history(assets=equity_list, fields='price', bar_count=90, frequency='1d').dropna()
    
    log_return = context.prices.apply(np.log).diff()
    
    # Calculate expected returns and Sharpes
    mean_return = log_return.mean().to_frame(name='mean_return')
    return_std = log_return.std().to_frame(name='return_std')
    context.output = context.output.join(mean_return).join(return_std)
    context.output['sharpes'] = context.output.mean_return / context.output.return_std
    
    top_n_sharpes = 1
    best_n_sharpes = context.output.groupby('cluster')['sharpes'].nlargest(top_n_sharpes)
    #print('best n sharpes', best_n_sharpes)
    
    context.best_equities = best_n_sharpes.index.get_level_values(1)
    
def calculate_weights(context, data):
    """
    This rebalancing function is called every 3 months (quarterly).
    """
    log_return = context.prices[context.best_equities].apply(np.log).diff().dropna()
    mean_return = log_return.mean().to_frame(name='mean_return')
    
    R = log_return.values.T
    mus = mean_return.values
    
    # Penalization parameter
    gamma = 5

     # Convert to cvxpy matrix and solve optimization problem
    n = len(mus)
    w = cvx.Variable(n)
    R_weighted = w.T*R
    mus = mus.T*w

    cost = cvx.sum_squares(mus-R_weighted) + gamma * cvx.norm(w, 1)
    my_problem = cvx.Problem(cvx.Minimize(cost), [cvx.sum_entries(w)==1, w>=0, w<=0.3])
    opt_value = my_problem.solve()
    weights = np.asarray(w.value).T[0]
    context.weights_list = pd.Series(weights, index=context.best_equities)
    print("Sum of the weights", context.weights_list.sum())
    print("ORDERS", context.weights_list)
    
    
def place_order(context,data):
    
    context.iMonths += 1
    if (context.iMonths % context.NMonths) != 1:
        return
    
    for stock in context.portfolio.positions:
        if stock not in context.weights_list.index:
            order_target(stock, 0)
            print("Closing out", stock)

    for stock in context.weights_list.index:
        if data.can_trade(stock):
            order_target_percent(stock, context.weights_list[stock])

def record_vars(context, data):
    
    record(leverage=context.account.leverage)
</code>
</pre>

# Analysis in Jupytr Notebook 



# Conclusions


