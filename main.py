from data_quality.report import Report
import pandas as pd

def main():
    daily_sales = pd.read_csv("data/ke_daily_sales.csv")
    visits = pd.read_csv("data/ke_visits.csv")
    d = {"sales": daily_sales, "relevance": visits}
    rep = Report()
    rep.fit(d)
    print(rep.to_str())


main()
