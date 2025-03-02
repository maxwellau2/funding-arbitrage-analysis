import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

FUTURES_TAKER_FEE = 0.0220 / 100
SPOT_TAKER_FEE = 0.0280 / 100
FEES = FUTURES_TAKER_FEE + SPOT_TAKER_FEE
REBALANCE_COST_BPS = 0.0007  # 7bps rebalance cost on 100% of portfolio
INITIAL_CAPITAL = 1_000_000  # Starting capital


# to store the positions of the portfolio
class PositionBook:
    def __init__(self, commission: float):
        self.commission = commission
        self.positions: list[Position] = []

    def has_no_positions(self):
        return len(self.positions) == 0

    def get_position_strings(self):
        return set([p.ticker for p in self.positions])

    def open_position(self, ticker: str, allocated_captial:float, spot_open: float, furures_open: float, mode: str)->float:
        """returns the commision to be subtracted from the total portfolio value in USD"""
        avg_price = (spot_open + furures_open)/2
        amount = allocated_captial / avg_price
        self.positions.append(Position(ticker, amount, spot_open, furures_open, mode))
        return self.commission * allocated_captial

    def close_position(self, ticker: str, close_spot: float, close_futures: float):
        """returns the pnl to be added and the commision to be subtracted from the total portfolio value in USD"""
        for position in self.positions:
            if position.ticker == ticker:
                # get the USD value of the positions
                usd = position.get_usd_value(close_spot, close_futures)
                commision = self.commission * usd
                pnl = position.close(close_spot, close_futures)
                self.positions.remove(position)
                break
        return pnl - commision

# basically an enum class for position
class Modes:
    LONG_SPOT_SHORT_FUTURES = "long_spot_short_futures"
    LONG_FUTURES_SHORT_SPOT = "long_futures_short_spot"


# this class is solely used to store the information about the position and the pnl
class Position:
    def __init__(self, ticker: str, amount: float, spot_open: float, futures_open: float, mode: str):
        assert mode in [Modes.LONG_SPOT_SHORT_FUTURES, Modes.LONG_FUTURES_SHORT_SPOT], "Mode must be one of the following: long_spot_short_futures, long_futures_short_spot"
        self.ticker = ticker
        self.amount = amount
        self.spot_open = spot_open
        self.futures_open = futures_open
        self.mode = mode

    def get_usd_value(self, current_spot: float, current_futures: float):
        return self.amount * (current_spot + current_futures) / 2

    # def get_usd_value(self):
    #     return self.amount * (self.futures_open + self.spot_open) / 2

    def close(self, close_spot: float, close_futures: float) -> float:
        """returns the pnl"""
        if self.mode == Modes.LONG_SPOT_SHORT_FUTURES:
            spot_pnl = self.amount * (close_spot - self.spot_open)
            futures_pnl = self.amount * (self.futures_open - close_futures)
            return spot_pnl + futures_pnl
        elif self.mode == Modes.LONG_FUTURES_SHORT_SPOT:
            spot_pnl = self.amount * (self.spot_open - close_spot)
            futures_pnl = self.amount * (close_futures - self.futures_open)
            return spot_pnl + futures_pnl

class Engine:
    def __init__(self, initial_capital: float, max_num_assets: int, holding_period: int):
        self.initial_capital = initial_capital
        self.max_num_assets = max_num_assets
        self.funding_rate_df = None
        self.basis_df: pd.DataFrame = None
        self.spot_df: pd.DataFrame = None
        self.futures_df: pd.DataFrame = None
        self.processed_data: pd.DataFrame = None
        # we use this to store the current capital
        self.current_capital = initial_capital
        self.positions = PositionBook(commission=FEES)
        self.capital_history: list[float] = []
        self.timesteps: list[str] = []
        self.holding_period = holding_period
        self.periods_elapsed = 0
        self.load_data_all_in_one()
        print(f"Data Integrity check: {self.check_data_integrity()}")

    def load_funding_rates(self, funding_rate_df):
        self.funding_rate_df = funding_rate_df

    def load_prices(self, spot_df: pd.DataFrame, basis_df: pd.DataFrame):
        self.spot_df = spot_df
        self.basis_df = basis_df
        self.futures_df = spot_df + basis_df
        

    @staticmethod
    def heuristic(column: pd.Series, small_window: int = 5, big_window: int = 20, penalty: float = 0.4):
        heu = abs(column.rolling(small_window).mean()) / (column.rolling(small_window).std() + 1e-6)
        sign_change = np.sign(column) != np.sign(column.shift(1))
        heu[sign_change] *= penalty
        return heu

    # ================ this was a fun idea but it didn't work ================
    # @staticmethod
    # def heuristic(column: pd.Series, small_window: int = 5, big_window: int = 20, penalty: float = 0.4):
    #     sample_std = column.rolling(small_window).std()
    #     sample_mean = column.rolling(small_window).mean()
    #     population_std = column.rolling(big_window).std()
    #     population_mean = column.rolling(big_window).mean()
    #     sample_min = column.rolling(small_window).min()
    #     population_min = column.rolling(big_window).min()
    #     return (sample_mean*population_std+sample_min+population_min)/(sample_std*population_mean+sample_min+population_min)


    def preprocess_data(self):
        # here we use the heuristic to determine the positions to take
        self.processed_data = self.futures_df.apply(lambda col: Engine.heuristic(col, small_window=5, big_window=20, penalty=0.4))
        return self.processed_data

    def check_data_integrity(self):
        # check that the data is not none
        assert self.funding_rate_df is not None, "Funding rate data must be loaded before running the main loop"
        assert self.spot_df is not None, "Spot data must be loaded before running the main loop"
        assert self.futures_df is not None, "Futures data must be loaded before running the main loop"
        assert self.processed_data is not None, "Data must be preprocessed before running the main loop"
        return True

    def load_data_all_in_one(self):
        spot = pd.read_csv('data/spot_price_clean.csv')
        basis = pd.read_csv('data/basis_df_clean_cols.csv')
        funding_rates = pd.read_csv('data/eight_hour_funding_clean_cols.csv')
        spot.set_index('time', inplace=True)
        basis.set_index('time', inplace=True)
        funding_rates.set_index('time', inplace=True)
        # convert to datetime
        spot.index = pd.to_datetime(spot.index)
        basis.index = pd.to_datetime(basis.index)
        funding_rates.index = pd.to_datetime(funding_rates.index)
        # print(f"there were {len(basis.columns)} cols")  

        for i in basis.columns:
            if pd.isna(basis[i].iloc[-1]):
                basis = basis.drop(i, axis=1) 

        for i in funding_rates.columns:
            if i not in basis.columns:
                funding_rates.drop(i, axis=1, inplace=True)
        # set intersections to find valid columns (in case there are missing columns)
        spot_valid = spot[list(set(funding_rates.columns)&set(spot.columns))]
        basis_valid = basis[list(set(funding_rates.columns)&set(spot.columns))] 
        funding_rates_valid = funding_rates[list(set(funding_rates.columns)&set(spot.columns))]
        self.load_funding_rates(funding_rates_valid)
        self.load_prices(spot_valid, basis_valid)
        self.preprocess_data()

    def run(self):
        if not self.check_data_integrity():
            raise Exception("Data is not loaded")
        # we loop over the processed data
        for idx, row in self.processed_data[10:].iterrows():
            fr = self.pre_timestep_hook(idx)
            basis = self.strategy(idx,row)
            self.current_capital = self.current_capital + fr + basis
            self.capital_history.append(self.current_capital)
            self.timesteps.append(idx)
        # for each row, we check if we need to open a position
        # if we do, we open the position
        return self.capital_history, self.timesteps
    
    def pre_timestep_hook(self, idx:pd.Timestamp):
        # we use this hook to 
        # add the new funding rate premium to current capital
        if self.positions.has_no_positions():
            return 0
        pnl = 0
        for position in self.positions.positions:
            # look up funding rate
            funding_rate = self.funding_rate_df.loc[idx, position.ticker]
            # get usd value of position
            usd_value = position.get_usd_value(self.spot_df.loc[idx, position.ticker], self.futures_df.loc[idx, position.ticker])
            # add to pnl
            # check position type
            if position.mode == Modes.LONG_SPOT_SHORT_FUTURES:
                pnl += funding_rate * usd_value
            else:
                pnl -= funding_rate * usd_value
        # print(pnl)
        return pnl
    
    def extract_top_k(self, data: pd.DataFrame, k: int, idx: str) -> list[str]:
        sorted_tickers = data.loc[idx].dropna().sort_values(ascending=False).index
        top_k = sorted_tickers[:k]
        return top_k

    def strategy(self, idx:pd.Timestamp, row:pd.Series):
        self.periods_elapsed += 1
        if self.periods_elapsed % self.holding_period != 0:
            return 0
        if self.positions.has_no_positions():
            # initialise the positions
            positions = self.extract_top_k(self.processed_data, self.max_num_assets + 10, idx)
            commision = 0
            for ticker in positions:
                commision += self.positions.open_position(ticker, self.current_capital/self.max_num_assets, self.spot_df.loc[idx, ticker], self.futures_df.loc[idx, ticker], Modes.LONG_SPOT_SHORT_FUTURES)
            return -commision
        # get the current positions as string
        current_positions = self.positions.get_position_strings()
        top_k_2 = self.extract_top_k(self.processed_data, self.max_num_assets+10, idx)
        top_k = set(top_k_2[:self.max_num_assets])
        top_k_2 = set(top_k_2)
        # removable tickers are the current positions - top 20 (set operation)
        to_remove = current_positions - top_k_2
        to_add = top_k - current_positions
        # limit it to 2 tickers only
        rebalance_count = min(2, len(to_remove))  # Evict bottom 2 and add
        to_remove = list(sorted(to_remove, key=lambda x: self.processed_data.loc[idx, x])[:rebalance_count])
        to_add = list(to_add)[:rebalance_count]
        # print(top_10)
        pnl = 0
        for ticker in to_remove:
            # print("removing", ticker)
            # close the position
            pnl += self.positions.close_position(ticker, self.spot_df.loc[idx, ticker], self.futures_df.loc[idx, ticker])
        
        for ticker in to_add:
            # print("adding", ticker)
            # open the position
            # get the funding rate and see if it is positive or negative
            fr = self.funding_rate_df.loc[idx, ticker]
            if fr > 0:
                mode = Modes.LONG_SPOT_SHORT_FUTURES
            else:
                mode = Modes.LONG_FUTURES_SHORT_SPOT
            pnl -= self.positions.open_position(ticker, self.current_capital/self.max_num_assets, self.spot_df.loc[idx, ticker], self.futures_df.loc[idx, ticker], mode)

        # add rebalance cost
        rebalanced_assets = len(to_remove) + len(to_add)
        rebalance_cost = self.current_capital * (REBALANCE_COST_BPS * rebalanced_assets) / self.max_num_assets 
        return pnl - rebalance_cost

    @staticmethod
    def calculate_performance_metrics(portfolio_values, timestamps, risk_free_rate=0.02):
        # i got lazy and just chatGPT'd this code snippet...
        """
        Calculate performance metrics for a trading strategy.
        
        Parameters:
        portfolio_values (list/array): Portfolio values at each timestep
        timestamps (list/array): Corresponding timestamps (datetime objects or strings)
        risk_free_rate (float): Annual risk-free rate, default 0.02 (2%)
        
        Returns:
        dict: Dictionary containing all calculated metrics
        """
        # Convert inputs to numpy arrays
        portfolio_values = np.array(portfolio_values)
        
        # Convert timestamps to datetime if they're strings
        if isinstance(timestamps[0], str):
            timestamps = [pd.to_datetime(ts) for ts in timestamps]
        
        # Create a pandas Series for easier handling
        portfolio_series = pd.Series(portfolio_values, index=timestamps)
        
        # Calculate daily returns
        returns = portfolio_series.pct_change().dropna()
        
        # Calculate drawdowns
        running_max = portfolio_series.cummax()
        drawdowns = (portfolio_series / running_max) - 1
        max_drawdown = drawdowns.min()
        
        # Calculate total time period in years
        days_passed = (timestamps[-1] - timestamps[0]).days
        years = days_passed / 365.25
        
        # Calculate CAGR (Compound Annual Growth Rate)
        ending_value = portfolio_values[-1]
        beginning_value = portfolio_values[0]
        cagr = (ending_value / beginning_value) ** (1 / years) - 1
        
        # Calculate annualized volatility
        daily_std = returns.std()
        annualized_std = daily_std * np.sqrt(365)  # Assuming 365 trading days in a year
        
        # Calculate Sharpe Ratio
        excess_return = cagr - risk_free_rate
        sharpe_ratio = excess_return / annualized_std if annualized_std != 0 else 0
        
        # Calculate Sortino Ratio (using only negative returns for downside deviation)
        negative_returns = returns[returns < 0]
        downside_deviation = negative_returns.std() * np.sqrt(252)
        sortino_ratio = excess_return / downside_deviation if downside_deviation != 0 else 0
        
        # Format results
        metrics = {
            'cagr': cagr,
            'volatility_annualized': annualized_std,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
        }
        
        return metrics
    
if __name__ == "__main__":
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("Portfolio Performance", "Performance Metrics"),
        vertical_spacing=0.2,
        row_heights=[0.7, 0.3],
        specs=[[{"type": "scatter"}], [{"type": "table"}]]
    )
    
    metrics_data = {
        'Assets': [],
        'Holding Periods': [],
        'CAGR (%)': [],
        'Max Drawdown (%)': [],
        'Sharpe Ratio': [],
        'Sortino Ratio': [],
        'Annual Vol (%)': []
    }

    # the template is [sharpe ratio, holding period, asset count]
    sharpe_to_params: list[tuple[float,int,int]] = []
    
    # Loop through different asset counts
    for i in range(10, 70, 10):
        for j in range(5,21,5):
            engine = Engine(initial_capital=INITIAL_CAPITAL, max_num_assets=i, holding_period=j)
            result, timesteps = engine.run()
            
            # Calculate metrics
            metrics = Engine.calculate_performance_metrics(result, timesteps)
            
            # Store metrics for table
            metrics_data['Assets'].append(i)
            metrics_data['Holding Periods'].append(j)
            metrics_data['CAGR (%)'].append(f"{metrics['cagr']*100:.2f}")
            metrics_data['Max Drawdown (%)'].append(f"{metrics['max_drawdown']*100:.2f}")
            metrics_data['Sharpe Ratio'].append(f"{metrics['sharpe_ratio']:.2f}")
            metrics_data['Sortino Ratio'].append(f"{metrics['sortino_ratio']:.2f}")
            metrics_data['Annual Vol (%)'].append(f"{metrics['volatility_annualized']*100:.2f}")
            sharpe_to_params.append((metrics['sharpe_ratio'], j, i))
            # Add performance line
            fig.add_trace(
                go.Scatter(x=timesteps, y=result, name=f"{i} assets {j} periods"),
                row=1, col=1, 
            )
    
    # Add metrics table
    fig.add_trace(
        go.Table(
            header=dict(
                values=list(metrics_data.keys()),
                fill_color='paleturquoise',
                align='center',
                font=dict(size=12)
            ),
            cells=dict(
                values=[metrics_data[k] for k in metrics_data.keys()],
                fill_color='lavender',
                align='center',
                font=dict(size=11)
            )
        ),
        row=2, col=1
    )
    
    # Update layout
    fig.update_layout(
        title_text="Portfolio Performance Analysis",
        height=1080,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    # Update y-axis title for performance plot
    fig.update_yaxes(title_text="Portfolio Value ($)", row=1, col=1)
    
    fig.show()
    # extract the top 5 parameter combinations by sharpe ratio
    print("Top 5 Parameter Combinations by Sharpe Ratio:")
    sharpe_to_params.sort(key=lambda x: x[0], reverse=True)
    for i in sharpe_to_params[:5]:
        print("Sharpe Ratio:", i[0], "Holding Period:", i[1], "Assets:", i[2])
    
# Output generated in console:
# Top 5 Parameter Combinations by Sharpe Ratio:
# Sharpe Ratio: 19.047267292127266 Holding Period: 10 Assets: 40
# Sharpe Ratio: 18.868393965570988 Holding Period: 10 Assets: 20
# Sharpe Ratio: 18.56470249661055 Holding Period: 10 Assets: 50
# Sharpe Ratio: 18.34384913623874 Holding Period: 20 Assets: 20
# Sharpe Ratio: 18.315045757688335 Holding Period: 5 Assets: 40

# Output generated by code snippet can be found in output.html