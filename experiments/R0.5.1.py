import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import pairwise_distances
from scipy.stats import kendalltau
import GPy
import pickle
import argparse

class HammingKernel(GPy.kern.Kern):
    def __init__(self, input_dim, lengthscale=1.0, active_dims=None):
        super(HammingKernel, self).__init__(input_dim, active_dims, "hamming")
        self.lengthscale = GPy.core.Param("lengthscale", lengthscale, GPy.constraints.Positive())
        self.link_parameters(self.lengthscale)

    def K(self, X, X2=None):
        if X2 is None:
            X2 = X
        d = pairwise_distances(X, X2, metric="hamming")  # normalized Hamming distance
        return np.exp(-d / self.lengthscale)

    def Kdiag(self, X):
        return np.ones(X.shape[0])


parser = argparse.ArgumentParser()
parser.add_argument("--lengthscale", type=float, required=True)
args = parser.parse_args()

lengthscale = args.lengthscale

df = pd.read_csv(
    "~/data/gpr/processed_data/UBE4B_MOUSE_Klevit2013-nscor_log2_ratio/data.csv"
)
df = df.rename(columns={"seq": "sequence", "log_fitness": "viability"})

seqs = df["sequence"].astype(str).values
X_chars = np.array([list(s) for s in seqs], dtype=object)
enc = OrdinalEncoder(dtype=np.float64, handle_unknown="use_encoded_value", unknown_value=-1)
X = enc.fit_transform(X_chars)
y = df["viability"].values.reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

kernel = HammingKernel(input_dim=X_train.shape[1], lengthscale=lengthscale)
model = GPy.models.GPRegression(X_train, y_train, kernel)
model.constrain_positive()

try:
    model.optimize(messages=False, max_iters=2000)
    y_pred, y_var = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    tau, _ = kendalltau(y_test.flatten(), y_pred.flatten())
    log_lik = model.log_likelihood()

    results = {
        "lengthscale": float(model.kern.lengthscale),
        "log_likelihood": float(log_lik),
        "rmse": float(rmse),
        "mae": float(mae),
        "kendall_tau": float(tau),
    }

    output_dir = "R0.5.1_results"
    os.makedirs(output_dir, exist_ok=True)

    out_file = os.path.join(output_dir, f"result_ls{lengthscale}.pkl")
    with open(out_file, "wb") as f:pickle.dump({"model": model, "results": results, "y_pred": y_pred}, f)

    print(f"Completed: lengthscale={results['lengthscale']:.4f}, RMSE={results['rmse']:.4f}")

except Exception as e:
    print(f"Failed: lengthscale={lengthscale}, error={e}")
