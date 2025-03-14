# **Funding Arbitrage Strategy**

## **Overview**
This project implements a **funding arbitrage trading strategy** that systematically profits from the **funding rate mechanism** in perpetual futures markets. By maintaining a **market-neutral** stance, the strategy seeks to **capture funding rate premiums** while minimizing directional risk.

## **Methodology & Approach**
### 1️⃣ **Arbitrage Premise**
- In perpetual futures markets, **funding rates** are paid between long and short traders at regular intervals.
- The strategy takes **opposite positions in spot and futures markets** to earn funding rate premiums while avoiding price exposure.

### 2️⃣ **Trading Strategy**
- **Heuristic-Based Selection**: Assets are chosen based **solely on historical funding rates** using a simple formula.
- **Position Allocation**: Capital is allocated dynamically among multiple assets.
- **Holding Period & Rebalancing**: Positions are held for a fixed duration and rebalanced periodically.

### 3️⃣ **Backtesting Approach**
- The backtest simulates historical performance using real **funding rate data**.
- Performance metrics include **Sharpe Ratio, CAGR, and Max Drawdown**.

---

## **Heuristic Formula**
The heuristic used to determine asset selection is:
$$
\mathbf{Heuristic\ Function:}
$$

$$
\mathrm{heuristic}(\mathrm{column},\ \mathrm{window} = 5,\ \mathrm{penalty} = 0.4)
$$

$$
\mathrm{heu} = \frac{|\mathbb{E}_w[\mathrm{column}]|}{\sigma_w(\mathrm{column})}
$$

**where:**

$$
\mathbb{E}_w[\mathrm{column}] = \mathrm{column.rolling(window).mean()}
$$

is the **moving average** (expected return).

$$
\sigma_w(\mathrm{column}) = \mathrm{column.rolling(window).std()}
$$

is the **rolling standard deviation** (volatility).

$$
\mathrm{heu}
$$

is the computed heuristic value.

---

To **apply a penalty for sign changes**, we define:

$$
\mathrm{sign\_change} = (\mathrm{sign}(\mathrm{column}) \neq \mathrm{sign}(\mathrm{column}.shift(1)))
$$

$$
\mathrm{heu}[\mathrm{sign\_change}] *= \mathrm{penalty}
$$


```python
def heuristic(column: pd.Series, small_window: int = 5, big_window: int = 20, penalty: float = 0.4):
    heu = abs(column.rolling(small_window).mean()) / (column.rolling(small_window).std() + 1e-6)
    sign_change = np.sign(column) != np.sign(column.shift(1))
    heu[sign_change] *= penalty
    return heu
```

### 🔹 **How It Works**
- The heuristic is **based entirely on historical funding rates**, not price movements.
- It calculates a ratio of the **short-term moving average to short-term volatility**.
- If the **funding rate sign changes**, the heuristic value is penalized to reduce unstable selections.
- This helps identify **consistently high funding rate opportunities** while filtering out noise.
- Epsilon of 1e-6 is added to the volatility calculation to prevent division by zero errors.

---

### **Pseudocode for Ticker Selection in the Funding Arbitrage Strategy**  

The strategy selects **tickers (assets)** based on their funding rate characteristics, ensuring the highest-yielding assets are chosen while maintaining stability.

---

### **1️⃣ Retrieve Funding Rate Data**
- Extract the latest funding rate values for all available tickers.
- Apply the **heuristic function** to generate scores for each ticker.

---

### **2️⃣ Sort and Select the Top Tickers**
1. **Sort tickers** in descending order based on their heuristic scores.
2. Select the **top (max_assets + 10) tickers**:
   ```
   top_tickers = select_top_k(heuristic_scores, k = max_assets + 10)
   ```
   - **Why extra 10 tickers?**  
     - Provides flexibility in rebalancing.  
     - Reduces unnecessary turnover in positions.

---

### **3️⃣ Rebalancing Logic**
1. **Determine the current holdings**:
   ```
   current_positions = get_current_positions()
   ```

2. **Identify tickers to remove**:
   - If a currently held ticker is **not in the top (max_assets + 10)**, mark it for removal:
    ```
    to_remove = current_positions - top_tickers
    ```

3. **Identify tickers to add**:
   - If a top-ranked ticker is **not currently held**, mark it for addition:
    ```
    to_add = top_tickers[:max_assets] - current_positions
    ```

4. **Limit the number of changes**:
   ```
   rebalance_count = min(2, len(to_remove))
   ```
   - Only replace at most **2 tickers per rebalance** to **reduce unnecessary churn**.

---

### **4️⃣ Execute Trades**
- **Close positions** for tickers in `to_remove`:
  ```
  for ticker in to_remove[:rebalance_count]:
      close_position(ticker)
  ```
- **Open positions** for tickers in `to_add`:
  ```
  for ticker in to_add[:rebalance_count]:
      open_position(ticker)
  ```

---

### **5️⃣ Summary**
✅ **Prioritizes high-yield tickers** based on funding rates.  
✅ **Uses extra 10 tickers as a buffer** to improve stability.  
✅ **Limits rebalancing to 2 assets at a time** to reduce excessive turnover.  
✅ **Dynamically adapts** as market conditions change.  

---



## **Setup Instructions**

### **Requirements**
- **Python 3.8+**
- Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```

### **Running the Strategy**
1. **Preprocess Market Data**
   ```bash
   python dataminer.ipynb
   ```
   This script extracts and cleans funding rate data.

2. **Analyze Arbitrage Feasibility**
   ```bash
   python arb.ipynb
   ```
   This explains **why selecting an extra 10 tickers** improves stability.

3. **Backtest the Strategy**
   ```bash
   python BacktestEngine.py
   ```
   This executes the strategy using historical funding rates.

4. **Review Results**
   - If you don’t want to run the script, **open** `output.html` to see the results.

---

## **Suggested Reading Order**
To understand the methodology and reasoning, follow this order:

1. **`dataminer.ipynb`** → Extracts and processes funding rate data.
2. **`arb.ipynb`** → Analyzes why **including an extra 10 tickers** improves stability.
3. **`BacktestEngine.py`** → Implements and executes the backtest.

---

## **Component Breakdown**

### **1️⃣ Data Extraction (`dataminer.ipynb`)**
- Loads and cleans **funding rate data** from historical market records.
- Formats data for backtesting.

### **2️⃣ Arbitrage Rationale (`arb.ipynb`)**
- Analyzes the heuristic’s effectiveness.
- Explains why an **extra 10 tickers** help improve performance.

### **3️⃣ Backtest Engine (`BacktestEngine.py`)**
- **`Engine`** → Executes the strategy using **funding rate-based heuristics**.
- **`PositionBook`** → Tracks positions and applies trading fees.
- **`Performance Metrics`** → Evaluates **Sharpe Ratio, CAGR, and Drawdowns**.

---

## **Key Insights**
✅ **A heuristic approach filters unstable funding rate shifts.**  
✅ **Adding 10 extra tickers improves stability by reducing turnover.**  

---

## **Results & Visualization**
- To see **performance results** without running the script, open <a href="https://maxwellau2.github.io/funding-arbitrage-analysis/output.html">backtest results here</a>.
- It contains **interactive charts and backtest analysis**.

---
