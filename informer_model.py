import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np

class InformerDummy(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(44, 50)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        out = self.linear(x_enc.squeeze(-1))
        return out.unsqueeze(-1)

@torch.no_grad()
def predict_future(model, data_tensor, scaler, input_len=44, pred_len=50, future_years=50):
    model.eval()
    device = torch.device("cpu")
    model.to(device)

    last_input = data_tensor[-input_len:].unsqueeze(0).unsqueeze(-1)
    time_input = torch.arange(0, input_len).unsqueeze(0).unsqueeze(-1).float()

    all_predictions = []
    current_input = last_input.to(device)
    current_time = time_input.to(device)

    for step in range(0, future_years, pred_len):
        current_pred_len = min(pred_len, future_years - step)
        time_dec = torch.arange(input_len + step, input_len + step + current_pred_len)
        time_dec = time_dec.unsqueeze(0).unsqueeze(-1).float().to(device)
        dec_input = torch.zeros(1, current_pred_len, 1).to(device)

        output = model(current_input, current_time, dec_input, time_dec)
        pred_output = output.cpu().numpy().reshape(-1, 1)
        pred_inverse = scaler.inverse_transform(pred_output)
        all_predictions.extend(pred_inverse.flatten()[:current_pred_len])

        new_input_part = output[:, -min(input_len, current_pred_len):, :]
        if current_pred_len >= input_len:
            current_input = new_input_part
        else:
            current_input = torch.cat([current_input[:, current_pred_len:, :], new_input_part], dim=1)

    return np.array(all_predictions)

def load_model_and_predict(df_global):
    scaler = MinMaxScaler()
    data_values = df_global['Global'].values.reshape(-1, 1)
    scaled = scaler.fit_transform(data_values)
    data_tensor = torch.tensor(scaled, dtype=torch.float32).squeeze()

    model = InformerDummy()
    for param in model.parameters():
        nn.init.normal_(param, mean=0, std=0.01)

    future_preds = predict_future(model, data_tensor, scaler, input_len=44, pred_len=50, future_years=50)
    last_year = df_global['Year'].max()
    pred_years = np.arange(last_year + 1, last_year + 1 + len(future_preds))

    df_pred = pd.DataFrame({
        'Year': pred_years,
        'Global': future_preds
    })
    return df_pred
