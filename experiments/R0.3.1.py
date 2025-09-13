import torch
import gpytorch
import pandas as pd
from transformers import AutoTokenizer, AutoModel, AutoConfig
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import kendalltau
import os

esm_config = AutoConfig.from_pretrained("facebook/esm2_t6_8M_UR50D")
esm_tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
esm_model = AutoModel.from_pretrained("facebook/esm2_t6_8M_UR50D")

df = pd.read_csv(
    "data/processed_data/UBE4B_MOUSE_Klevit2013-nscor_log2_ratio/data.csv"
)
df = df.rename(columns={"seq": "sequence", "log_fitness": "viability"})

embedding_file = "cache/embeddings.pt"
if os.path.exists(embedding_file):
    X = torch.load(embedding_file)
else:
    embeddings = []
    with torch.no_grad():
        for seq in df["sequence"]:
            tokenized_input = esm_tokenizer(seq, return_tensors="pt")
            output = esm_model(**tokenized_input)
            seq_embedding = output.last_hidden_state.mean(dim=1).squeeze()
            embeddings.append(seq_embedding)
    X = torch.stack(embeddings)
    torch.save(X, embedding_file)

y = torch.tensor(df["viability"].values, dtype=torch.float32)

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

def run_exact_gp(
        X, y, test_size=0.2, seed=None, training_iterations=10000, lr=0.0001
):
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )

    X_train = X_train.float()
    y_train = y_train.float()
    X_test = X_test.float()
    y_test = y_test.float()

    # Likelihood and model
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = ExactGPModel(X_train, y_train, likelihood)

    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for i in range(training_iterations):
        optimizer.zero_grad()
        output = model(X_train)
        loss = -mll(output, y_train)
        loss.backward()
        if (i + 1) % 1000 == 0:
            print(f"Iter {i+1}/{training_iterations} - Loss: {loss.item():.4f}")
        optimizer.step()

    # Evaluation
    model.eval()
    likelihood.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        y_pred = likelihood(model(X_test)).mean

    rmse = mean_squared_error(y_test, y_pred, squared=False)
    mae = mean_absolute_error(y_test, y_pred)
    tau, _ = kendalltau(y_test.numpy(), y_pred.numpy())

    print(f"RMSE: {rmse:.4f}, MAE: {mae:.4f}, Kendall Tau: {tau:.4f}")
    return y_test, y_pred

y_test, y_pred = run_exact_gp(
    X, y, test_size=0.2, training_iterations=10000, lr=0.0001
)
