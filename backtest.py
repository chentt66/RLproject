#-*- encoding:utf-8 -*-
from __future__ import division
import pandas as pd
import numpy as np


class Analysis:
    """
    Backtest portfolio performance
    Input:
    --------
    A series of dataframe
    """

    def __init__(self, df, save):
        self.pct = df
        self.netvalue = self.pct.add(1).cumprod()
        self.mon_num = self.pct.shape[0]
        self.save = save

    # 累积收益率
    def calculate_return(self):
        total_return = self.netvalue.iloc[-1] - 1
        return total_return

    # 年化收益率
    def calculate_annualreturn(self):        
        annual_return = self.netvalue.iloc[-1] ** (12 / self.mon_num) - 1
        return annual_return

    # 最大回撤
    def calculate_mdd(self):
        drawdown = 1 - self.netvalue.div(self.netvalue.cummax())
        mdd = drawdown.max()
        return mdd

    # 年化波动率
    def calculate_volatility(self):
        monthly_std = self.pct.std()
        annual_std = monthly_std * np.sqrt(12)
        return annual_std

    # 信息比率 -- 有点问题
    def calculate_ratio(self):
        annual_return = self.netvalue.iloc[-1] ** (12 / self.mon_num) - 1
        annual_std = self.pct.std() * np.sqrt(12)
        ratio = annual_return / annual_std
        return ratio

    def report(self):
        df = pd.DataFrame()
        df["Cumulative Return"] = self.calculate_return().values
        df["MDD"] = self.calculate_mdd().values
        df["Annualized Volatility"] = self.calculate_volatility().values
        df["Annualized Return"] = self.calculate_annualreturn().values
        df["IR"] = self.calculate_ratio().values
        df.iloc[:, :-1] = df.iloc[:, :-1].applymap(lambda x: "{:.2f}%".format(x*100))
        df.iloc[:, -1] = df.iloc[:, -1].apply(lambda x: "{:.2f}".format(x))
      
        if self.save:
            df.to_csv("{}-结果统计.csv".format(self.name), encoding="utf-8")
        else:
            pass            
        
        return df


class Analysis_Yearly:
    """
    分析一个 Series 的结果
    """

    def __init__(self, pct_ts, period):
        self.pct = pct_ts
        self.num = 12 if period == "monthly" else 50
        total = np.append([0.], pct_ts)
        self.netvalue = pd.Series((total+1.).cumprod())

    def calculate_return(self):
        total_return = self.netvalue.iloc[-1] - 1
        return total_return

    def calculate_annualreturn(self):
        annual_return = self.netvalue.iloc[-1] ** (self.num / len(self.pct)) - 1
        return annual_return

    def calculate_mdd(self):
        dd = 1 - self.netvalue.div(self.netvalue.cummax())
        mdd = dd.max()
        return mdd

    def calculate_volatility(self):
        pct_std = self.pct.std()
        annual_std = pct_std * np.sqrt(self.num)
        return annual_std

    def calculate_ratio(self):
        annual_return = self.netvalue.iloc[-1] ** (self.num / len(self.pct)) - 1
        weekly_std = self.pct.std()
        ratio = annual_return / (np.sqrt(self.num) * weekly_std)
        return ratio

    def report(self):
        total_return = self.calculate_return()
        mdd = self.calculate_mdd()
        std = self.calculate_volatility()
        annual_return = self.calculate_annualreturn()
        ratio = self.calculate_ratio()
        return [total_return, mdd, std, annual_return, ratio]


def ana_year(filename, testname, start_year, end_year, period, save):
    df = pd.read_csv("{}.csv".format(filename), index_col=0, parse_dates=[0], encoding="utf-8")
    result_df = pd.DataFrame(0., index=np.arange(start_year, end_year+1), columns=["累积收益率", "最大回撤", "年化波动率", "年化收益率", "信息比率"])
    for year in np.arange(start_year, end_year+1):
        print("Start analyzing year {}".format(year))
        this_year = year
        ts = df.loc[str(this_year), testname]
        ana = Analysis_Yearly(ts, period)
        result_df.loc[year] = ana.report()
    ana =Analysis_Yearly(df[testname], period)
    result_df.loc["All"] = ana.report()
    df = result_df.copy()
    df.iloc[:, :-1] = df.iloc[:, :-1].applymap(lambda x: "{:.2f}%".format(x*100.))
    df.iloc[:, -1] = df.iloc[:, -1].apply(lambda x: "{:.2f}".format(x))
    if save==1:
        df.to_csv(r"{}-分年度报告.csv".format(filename), encoding="utf-8")
    else:
        pass
    return df