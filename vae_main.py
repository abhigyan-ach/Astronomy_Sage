
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
import os

import random
import time

def set_seed(seed):
    """Set all random seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return seed

# Generate or use a specific seed
set_seed(42)
print(f"Using seed: {42}")

# 1. Load and preprocess data
df = pd.read_csv("/home/acherjan/llama_experiments/astroque/merged_df_w3w4_final_all_colors.csv")
print(len(df))
df["w1_w2_color"] = df["w1mpro"] - df["w2mpro"]
df["w3_w4_color"] = df["w3mpro"] - df["w4mpro"]
df["j_m-h_m"] = df["j_m"] - df["h_m"]
df["j_m-k_m"] = df["j_m"] - df["k_m"]
df["h_m-k_m"] = df["h_m"] - df["k_m"]
df['h_m-j_m'] = df['h_m'] - df['j_m']
df['k_m-h_m'] = df['k_m'] - df['h_m']
df['k_m-j_m'] = df['k_m'] - df['j_m']
# Cross-domain colors:
df['g-K'] = df['g_psf'] - df['k_m']

print(df.columns)

n_columns = [["ug","uv","ui","ur","uz","vg","vr","vi","vz","gr","gi","gz","ri","rz","iz","j_m-h_m","h_m-k_m","j_m-k_m","w1_w2_color","w3_w4_color"],
["ug","uv","ui","ur","uz","vg","vr","vi","vz","gr","gi","gz","ri","rz","iz","j_m-h_m","h_m-k_m","j_m-k_m","w1_w2_color","w3_w4_color"],
["ug","uv","ui","ur","uz","vg","vr","vi","vz","gr","gi","gz","ri","rz","iz","j_m-h_m","h_m-k_m","j_m-k_m","w1_w2_color","w3_w4_color"],
["ug","uv","ui","ur","uz","vg","vr","vi","vz","gr","gi","gz","ri","rz","iz","j_m-h_m","h_m-k_m","j_m-k_m","w1_w2_color","w3_w4_color"],
["ug","uv","ui","ur","uz","vg","vr","vi","vz","gr","gi","gz","ri","rz","iz","j_m-h_m","h_m-k_m","j_m-k_m","w1_w2_color","w3_w4_color"]]
# n_columns = [["ug","uv","ui","ur","uz","vu","vg","vr","vi","vz","gu","gv","gr","gi","gz","ru","rv","rg","ri","rz","iu","iv","ig","ir","iz","zu","zv","zg","zr","zi","j_m-h_m","h_m-k_m","w1_w2_color"],
#         ["ug","uv","ui","ur","uz","vu","vg","vr","vi","vz","gu","gv","gr","gi","gz","ru","rv","rg","ri","rz","iu","iv","ig","ir","iz","zu","zv","zg","zr","zi","w3_w4_color","j_m-h_m","h_m-k_m","j_m-k_m"],
#         ["ug","uv","ui","ur","uz","vu","vg","vr","vi","vz","gu","gv","gr","gi","gz","ru","rv","rg","ri","rz","iu","iv","ig","ir","iz","zu","zv","zg","zr","zi","w1_w2_color","j_m-h_m","h_m-k_m","k_m-j_m"],
#         ["ug","uv","ui","ur","uz","vu","vg","vr","vi","vz","gu","gv","gr","gi","gz","ru","rv","rg","ri","rz","iu","iv","ig","ir","iz","zu","zv","zg","zr","zi","w3_w4_color","j_m-h_m","h_m-k_m","j_m-k_m"],
#         ["ug","uv","ui","ur","uz","vu","vg","vr","vi","vz","gu","gv","gr","gi","gz","ru","rv","rg","ri","rz","iu","iv","ig","ir","iz","zu","zv","zg","zr","zi","w1_w2_color","j_m-h_m","h_m-k_m","j_m-k_m"],["ug","uv","ui","ur","uz","vu","vg","vr","vi","vz","gu","gv","gr","gi","gz","ru","rv","rg","ri","rz","iu","iv","ig","ir","iz","zu","zv","zg","zr","zi","j_m-h_m","h_m-k_m","w1_w2_color"],
#         ["ug","uv","ui","ur","uz","vu","vg","vr","vi","vz","gu","gv","gr","gi","gz","ru","rv","rg","ri","rz","iu","iv","ig","ir","iz","zu","zv","zg","zr","zi","w3_w4_color","j_m-h_m","h_m-k_m","j_m-k_m"],
#         ["ug","uv","ui","ur","uz","vu","vg","vr","vi","vz","gu","gv","gr","gi","gz","ru","rv","rg","ri","rz","iu","iv","ig","ir","iz","zu","zv","zg","zr","zi","w1_w2_color","j_m-h_m","h_m-k_m","k_m-j_m"],
#         ["ug","uv","ui","ur","uz","vu","vg","vr","vi","vz","gu","gv","gr","gi","gz","ru","rv","rg","ri","rz","iu","iv","ig","ir","iz","zu","zv","zg","zr","zi","w3_w4_color","j_m-h_m","h_m-k_m","j_m-k_m"],
#         ["ug","uv","ui","ur","uz","vu","vg","vr","vi","vz","gu","gv","gr","gi","gz","ru","rv","rg","ri","rz","iu","iv","ig","ir","iz","zu","zv","zg","zr","zi","w1_w2_color","j_m-h_m","h_m-k_m","j_m-k_m"],["ug","uv","ui","ur","uz","vu","vg","vr","vi","vz","gu","gv","gr","gi","gz","ru","rv","rg","ri","rz","iu","iv","ig","ir","iz","zu","zv","zg","zr","zi","j_m-h_m","h_m-k_m","w1_w2_color"],
#         ["ug","uv","ui","ur","uz","vu","vg","vr","vi","vz","gu","gv","gr","gi","gz","ru","rv","rg","ri","rz","iu","iv","ig","ir","iz","zu","zv","zg","zr","zi","w3_w4_color","j_m-h_m","h_m-k_m","j_m-k_m"],
#         ["ug","uv","ui","ur","uz","vu","vg","vr","vi","vz","gu","gv","gr","gi","gz","ru","rv","rg","ri","rz","iu","iv","ig","ir","iz","zu","zv","zg","zr","zi","w1_w2_color","j_m-h_m","h_m-k_m","k_m-j_m"],
#         ["ug","uv","ui","ur","uz","vu","vg","vr","vi","vz","gu","gv","gr","gi","gz","ru","rv","rg","ri","rz","iu","iv","ig","ir","iz","zu","zv","zg","zr","zi","w3_w4_color","j_m-h_m","h_m-k_m","j_m-k_m"],
#         ["ug","uv","ui","ur","uz","vu","vg","vr","vi","vz","gu","gv","gr","gi","gz","ru","rv","rg","ri","rz","iu","iv","ig","ir","iz","zu","zv","zg","zr","zi","w1_w2_color","j_m-h_m","h_m-k_m","j_m-k_m"],["ug","uv","ui","ur","uz","vu","vg","vr","vi","vz","gu","gv","gr","gi","gz","ru","rv","rg","ri","rz","iu","iv","ig","ir","iz","zu","zv","zg","zr","zi","j_m-h_m","h_m-k_m","w1_w2_color"],
#         ["ug","uv","ui","ur","uz","vu","vg","vr","vi","vz","gu","gv","gr","gi","gz","ru","rv","rg","ri","rz","iu","iv","ig","ir","iz","zu","zv","zg","zr","zi","w3_w4_color","j_m-h_m","h_m-k_m","j_m-k_m"],
#         ["ug","uv","ui","ur","uz","vu","vg","vr","vi","vz","gu","gv","gr","gi","gz","ru","rv","rg","ri","rz","iu","iv","ig","ir","iz","zu","zv","zg","zr","zi","w1_w2_color","j_m-h_m","h_m-k_m","k_m-j_m"],
#         ["ug","uv","ui","ur","uz","vu","vg","vr","vi","vz","gu","gv","gr","gi","gz","ru","rv","rg","ri","rz","iu","iv","ig","ir","iz","zu","zv","zg","zr","zi","w3_w4_color","j_m-h_m","h_m-k_m","j_m-k_m"],
#         ["ug","uv","ui","ur","uz","vu","vg","vr","vi","vz","gu","gv","gr","gi","gz","ru","rv","rg","ri","rz","iu","iv","ig","ir","iz","zu","zv","zg","zr","zi","w1_w2_color","j_m-h_m","h_m-k_m","j_m-k_m"],["ug","uv","ui","ur","uz","vu","vg","vr","vi","vz","gu","gv","gr","gi","gz","ru","rv","rg","ri","rz","iu","iv","ig","ir","iz","zu","zv","zg","zr","zi","j_m-h_m","h_m-k_m","w1_w2_color"],
#         ["ug","uv","ui","ur","uz","vu","vg","vr","vi","vz","gu","gv","gr","gi","gz","ru","rv","rg","ri","rz","iu","iv","ig","ir","iz","zu","zv","zg","zr","zi","w3_w4_color","j_m-h_m","h_m-k_m","j_m-k_m"],
#         ["ug","uv","ui","ur","uz","vu","vg","vr","vi","vz","gu","gv","gr","gi","gz","ru","rv","rg","ri","rz","iu","iv","ig","ir","iz","zu","zv","zg","zr","zi","w1_w2_color","j_m-h_m","h_m-k_m","k_m-j_m"],
#         ["ug","uv","ui","ur","uz","vu","vg","vr","vi","vz","gu","gv","gr","gi","gz","ru","rv","rg","ri","rz","iu","iv","ig","ir","iz","zu","zv","zg","zr","zi","w3_w4_color","j_m-h_m","h_m-k_m","j_m-k_m"],
#         ["ug","uv","ui","ur","uz","vu","vg","vr","vi","vz","gu","gv","gr","gi","gz","ru","rv","rg","ri","rz","iu","iv","ig","ir","iz","zu","zv","zg","zr","zi","w1_w2_color","j_m-h_m","h_m-k_m","j_m-k_m"]]

# n_columns = [[col.strip() for col in cols[0].split(',')] for cols in n_columns]
l_dims = [3,4,5,6]

for l_dim in l_dims:
    for numeric_columns in n_columns:
            # Impute missing values
            print("Computing VAE for latent dimension: ", l_dim)
            print ("Numeric columns:")
            print(numeric_columns)
            imputer = SimpleImputer(strategy='median')
            df[numeric_columns] = imputer.fit_transform(df[numeric_columns])

            # Normalize
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(df[numeric_columns].values)

            object_ids = df['object_id'].values

            # 2. Define VAE (using original feature dimensions)
            class VAE(nn.Module):
                def __init__(self, input_dim, latent_dim=l_dim):
                    super(VAE, self).__init__()
                    hidden_dim = max(32, input_dim // 2)  # Adaptive hidden dimension
                    self.fc1 = nn.Linear(input_dim, hidden_dim)
                    self.fc21 = nn.Linear(hidden_dim, latent_dim)  # mean
                    self.fc22 = nn.Linear(hidden_dim, latent_dim)  # logvar
                    self.fc3 = nn.Linear(latent_dim, hidden_dim)
                    self.fc4 = nn.Linear(hidden_dim, input_dim)

                def encode(self, x):
                    h1 = torch.relu(self.fc1(x))
                    return self.fc21(h1), self.fc22(h1)

                def reparameterize(self, mu, logvar):
                    std = torch.exp(0.5*logvar)
                    eps = torch.randn_like(std)
                    return mu + eps*std

                def decode(self, z):
                    h3 = torch.relu(self.fc3(z))
                    return self.fc4(h3)

                def forward(self, x):
                    mu, logvar = self.encode(x)
                    z = self.reparameterize(mu, logvar)
                    return self.decode(z), mu, logvar

            def vae_loss(recon_x, x, mu, logvar):
                # Reconstruction + KL divergence losses summed over all elements and batch
                recon_loss = nn.MSELoss(reduction='none')(recon_x, x).sum(dim=1)
                kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
                return recon_loss + kld

            # 3. Train VAE on original scaled features
            input_dim = len(numeric_columns)  # Using original feature dimensions
            vae = VAE(input_dim)
            optimizer = optim.Adam(vae.parameters(), lr=1e-3)
            X_tensor = torch.tensor(X_scaled, dtype=torch.float32)  # Using original scaled features

            vae.train()
            epochs = 100
            batch_size = 64
            for epoch in range(epochs):
                perm = torch.randperm(X_tensor.size(0))
                for i in range(0, X_tensor.size(0), batch_size):
                    indices = perm[i:i+batch_size]
                    batch = X_tensor[indices]
                    optimizer.zero_grad()
                    recon_batch, mu, logvar = vae(batch)
                    loss = vae_loss(recon_batch, batch, mu, logvar).mean()
                    loss.backward()
                    optimizer.step()
                if (epoch+1) % 10 == 0:
                    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

            # 4. Compute reconstruction loss for each object
            vae.eval()
            with torch.no_grad():
                recon, mu, logvar = vae(X_tensor)
                # Calculate reconstruction loss in original feature space
                losses = nn.MSELoss(reduction='none')(recon, X_tensor).mean(dim=1).numpy()

            print(f"Using original feature space losses")

            # 5. Flag anomalies (e.g., top 1% highest losses)
            threshold = np.percentile(losses, 99)
            anomaly_indices = np.where(losses > threshold)[0]
            anomaly_object_ids = object_ids[anomaly_indices]

            print("Anomalous object IDs:", anomaly_object_ids)

            if os.path.exists('Final_Runs_Dim_{l_dim}.csv'):
                group_number = 'Final_Runs_Dim_{l_dim}_Group'+str(len(pd.read_csv('Final_Runs_Dim_{l_dim}.csv'))+1)
                os.makedirs(group_number, exist_ok=True)
            else:
                group_number = 'Final_Runs_Dim_{l_dim}_Group1'
                os.makedirs(group_number, exist_ok=True)
            
            # 6. Visualizations
            plt.figure(figsize=(15, 5))

            # Loss distribution
            plt.subplot(1, 3, 1)
            sns.histplot(losses, bins=50, kde=True)
            plt.axvline(threshold, color='red', linestyle='--', label=f'Anomaly threshold ({threshold:.4f})')
            plt.title('VAE Reconstruction Loss Distribution')
            plt.xlabel('Reconstruction Loss (MSE)')
            plt.ylabel('Count')
            plt.legend()

            # Boxplot
            plt.subplot(1, 3, 2)
            sns.boxplot(x=losses)
            plt.title('Boxplot of VAE Reconstruction Loss')
            plt.xlabel('Reconstruction Loss (MSE)')

            # Scatter plot
            plt.subplot(1, 3, 3)
            plt.scatter(range(len(losses)), losses, s=10, alpha=0.6, label='All objects')
            plt.scatter(anomaly_indices, losses[anomaly_indices], color='red', s=20, label='Anomalies')
            plt.title('Reconstruction Loss per Object')
            plt.xlabel('Object Index')
            plt.ylabel('Reconstruction Loss (MSE)')
            plt.legend()

            plt.tight_layout()
            plt.savefig(group_number+'/vae_analysis.png', dpi=300, bbox_inches='tight')
            plt.show()

            # 7. Save results
            with open(group_number+'/anomalous_object_ids.txt', 'w') as f:
                for oid in anomaly_object_ids:
                    f.write(f"{oid}\n")

            # Save losses
            loss_df = pd.DataFrame({
                'object_id': object_ids, 
                'reconstruction_loss': losses
            })
            loss_df.to_csv(group_number+'/vae_reconstruction_losses.csv', index=False)

            # Initialize variables to store overlapping anomalies for each dataset
            merc_galactic_overlaps = []
            merc_extragalactic_overlaps = []
            skymapper_confirmed_overlaps = []
            skymapper_candidates_overlaps = []

            df2 = pd.read_csv("/home/acherjan/llama_experiments/astroque/merc_galactic.csv")
            if 'object_id' in df2.columns:
                merc_galactic_overlaps = [int(oid) for oid in anomaly_object_ids if oid in df2['object_id'].values]
                print(f"\nFound {len(merc_galactic_overlaps)} anomalous objects that are also present in the Merc Galactic dataset:")
                if merc_galactic_overlaps:
                    print(merc_galactic_overlaps)
            else:
                print("\n 'object_id' column not found in Merc Galactic dataset")

            df2 = pd.read_csv("/home/acherjan/llama_experiments/astroque/merc_extragalactic.csv")
            if 'object_id' in df2.columns:
                merc_extragalactic_overlaps = [int(oid) for oid in anomaly_object_ids if oid in df2['object_id'].values]
                print(f"\nFound {len(merc_extragalactic_overlaps)} anomalous objects that are also present in the Merc Extra-Galactic dataset:")
                if merc_extragalactic_overlaps:
                    print(merc_extragalactic_overlaps)
            else:
                print("\n 'object_id' column not found in Merc Extra-Galactic dataset")

            df2 = pd.read_csv("/home/acherjan/llama_experiments/astroque/skymapper.csv")
            if 'object_id' in df2.columns:
                skymapper_confirmed_overlaps = [int(oid) for oid in anomaly_object_ids if oid in df2['object_id'].values]
                print(f"\nFound {len(skymapper_confirmed_overlaps)} anomalous objects that are also present in Adrian's confirmed dataset:")
                if skymapper_confirmed_overlaps:
                    print(skymapper_confirmed_overlaps)
            else:
                print("\n 'object_id' column not found in Adrian's confirmed dataset")

            df2 = pd.read_csv("/home/acherjan/llama_experiments/astroque/skymapper_candidates.csv")
            if 'object_id' in df2.columns:
                skymapper_candidates_overlaps = [int(oid) for oid in anomaly_object_ids if oid in df2['object_id'].values]
                print(f"\nFound {len(skymapper_candidates_overlaps)} anomalous objects that are also present in Adrian's un-confirmed dataset:")
                if skymapper_candidates_overlaps:
                    print(skymapper_candidates_overlaps)
            else:
                print("\n 'object_id' column not found in Adrian's un-confirmed dataset")

            # Create or append to Runs.csv
            runs_csv_path = 'Final_Runs_Dim.csv'

            # Check if Runs.csv exists
            if not os.path.exists(runs_csv_path):
                # Create new CSV with headers
                runs_df = pd.DataFrame(columns=['No.', 'Variables', 'Merc_Galactic', 'Merc_extra_galactic', 
                                            'Skymapper_Confirmed', 'Skymapper_candidates', 'Total'])
                run_number = 1
            else:
                # Read existing CSV to get the next run number
                runs_df = pd.read_csv(runs_csv_path)
                run_number = len(runs_df) + 1

            # Calculate total overlaps
            total_overlaps = len(merc_galactic_overlaps) + len(merc_extragalactic_overlaps) + len(skymapper_confirmed_overlaps) + len(skymapper_candidates_overlaps)

            # Prepare the new row data
            new_row = {
                'No.': run_number,
                'Variables': ','.join(numeric_columns),
                'Merc_Galactic': ','.join(map(str, merc_galactic_overlaps)) if merc_galactic_overlaps else '',
                'Merc_extra_galactic': ','.join(map(str, merc_extragalactic_overlaps)) if merc_extragalactic_overlaps else '',
                'Skymapper_Confirmed': ','.join(map(str, skymapper_confirmed_overlaps)) if skymapper_confirmed_overlaps else '',
                'Skymapper_candidates': ','.join(map(str, skymapper_candidates_overlaps)) if skymapper_candidates_overlaps else '',
                'Total': total_overlaps
            }

            # Add the new row to the dataframe
            runs_df = pd.concat([runs_df, pd.DataFrame([new_row])], ignore_index=True)

            # Save the updated dataframe
            runs_df.to_csv(runs_csv_path, index=False)

            print(f"\nRun {run_number} results saved to {runs_csv_path}")
            print(f"Variables used: {','.join(numeric_columns)}")
            print(f"Total overlaps found: {total_overlaps}")

# Read final results
df = pd.read_csv("Final_Runs_Dim.csv")
print("Combination with most confirmed anomalies:")
print(df.sort_values(by='Merc_Galactic', ascending=False).head(1))

print("Combination with most un-confirmed Skymapper anomalies:")
print(df.sort_values(by='Skymapper_candidates', ascending=False).head(1))

print("Combination with most total anomalies:")
print(df.sort_values(by='Total', ascending=False).head(1))