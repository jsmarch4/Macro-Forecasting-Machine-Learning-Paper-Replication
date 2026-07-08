from data_utils import load_replication_data

X, y = load_replication_data()

print(X.loc["1992-04-01"].describe())

print("\nLargest predictors:\n")

print(
    X.loc["1992-04-01"]
      .abs()
      .sort_values(ascending=False)
      .head(20)
)