import os
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModel, AutoConfig
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import kendalltau
import GPy
import pickle
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--variance", type=float, required=True)
parser.add_argument("--lengthscale", type=float, required=True)
args = parser.parse_args()

variance = args.variance
lengthscale = args.lengthscale

df = pd.read_csv("~/data/gpy/processed_data/UBE4B_MOUSE_Klevit2013-nscor_log2_ratio/data.csv")
df = df.rename(columns={"seq": "sequence", "log_fitness": "viability"})

embedding_file = "cache/embeddings.npy"
if os.path.exists(embedding_file):
    X = np.load(embedding_file)
else:
    esm_config = AutoConfig.from_pretrained("facebook/esm2_t6_8M_UR50D")
    esm_tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
    esm_model = AutoModel.from_pretrained("facebook/esm2_t6_8M_UR50D")
    embeddings = []
    esm_model.eval()
    with torch.no_grad():
        for seq in df["sequence"]:
            tokenized_input = esm_tokenizer(seq, return_tensors="pt")
            output = esm_model(**tokenized_input)
            seq_embedding = output.last_hidden_state.mean(dim=1).squeeze().numpy()
            embeddings.append(seq_embedding)
    X = np.vstack(embeddings)
    np.save(embedding_file, X)

y = df["viability"].values.reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

kernel = GPy.kern.RBF(input_dim=X_train.shape[1], variance=variance, lengthscale=lengthscale)
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
        "variance": variance,
        "lengthscale": lengthscale,
        "log_likelihood": log_lik,
        "rmse": rmse,
        "mae": mae,
        "kendall_tau": tau
    }

    output_dir = "slurm_gp_results"
    os.makedirs(output_dir, exist_ok=True)

    # Save results
    out_file = os.path.join(output_dir, f"result_var{variance}_ls{lengthscale}.pkl")
    with open(out_file, "wb") as f:
        pickle.dump({"model": model, "results": results, "y_pred": y_pred}, f)

    print(f"Completed: variance={variance}, lengthscale={lengthscale}, RMSE={rmse:.4f}")

except Exception as e:
    print(f"Failed: variance={variance}, lengthscale={lengthscale}, error={e}")
