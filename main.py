import pandas as pd


def main():
    daily_sales = pd.read_csv("data/ke_daily_sales.csv")
    visits = pd.read_csv("data/ke_visits.csv")
    print(daily_sales)
    print("------")
    print(visits)


main()
