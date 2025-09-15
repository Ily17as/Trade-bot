# Trade-Bot

Automatic stock trader based on ML/DL models.

## Project Description
The goal of this project is to develop an auto-trader for the stock market that, using several ML models (Computer Vision, Deep Learning, Reinforcement Learning, Stochastic Market Models), will determine whether a stock is:
- undervalued,
- fairly valued,
- overvalued.

The main value is accurate stock valuation predictions to increase traders' confidence in their investment decisions.

## Competitor and SOTA Research
- **TrendSpider** — custom heat maps and a fully automated AI trading bot.  
- **Danelfin** — reliable service assigning each stock an AI-generated score from 1 to 10.  

Our approach is unique: we combine **DL + CV + Reinforcement Learning + Stochastic Market Models**, which has not been tested in combination before.

## ML Pipeline Architecture
1. **Data Collection**  
   - [Yahoo Finance API](https://pypi.org/project/yfinance/)  
   - [T-Bank API](https://developer.tbank.ru/docs/api/t-api)  

2. **Preprocessing**  
   - Cleaning time series  
   - Price normalization  
   - Generating charts for the CV model  

3. **Models**
   - **Deep Learning** — analyzing stock historical data  
   - **Computer Vision** — analyzing price charts  
   - **Statistical methods** — Monte Carlo simulations  
   - **Reinforcement Learning** — trading decision making  

4. **Evaluation**
   - Precision / Recall  
   - Compound Annual Growth Rate (CAGR)  
   - Sharpe Ratio  
   - Max Drawdown  
   - Equity Curves  

5. **Deployment**
   - Planned integration into a **trading bot** for trade simulation  

## Success Metrics
- **Sharpe Ratio** — benchmark against professional traders  
- **CAGR** — capital growth on test data  
- **Precision/Recall** — prediction accuracy  

## Team
- **Daria Alexandrova** — CV model  
- **Kamilya Shakirova** — DL model  
- **Ilyas Galiev** — statistical model  

## Usage
```bash
# Clone repository
git clone https://github.com/Ily17as/Trade-bot.git
cd Trade-bot

# Install dependencies
pip install -r requirements.txt

# Run experiments
python main.py
```

## References
- [GitHub Repo](https://github.com/Ily17as/Trade-bot)  
- [TrendSpider](https://trendspider.com/)  
- [Danelfin](https://danelfin.com/)  
- [Research: Deep RL for trading](https://www.researchgate.net/publication/363302274_Deep_Reinforcement_Learning_Approach_for_Trading_Automation_in_The_Stock_Market)  
- [Research: DL in trading](https://www.sciencedirect.com/science/article/pii/S0957417424013319#s0140)  

---
*Project under development. Model integration and strategy testing are planned for future releases.*
