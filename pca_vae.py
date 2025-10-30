import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
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
# SEED = int(time.time()) % 100000  # Generate based on timestamp, or set manually
set_seed(42)
print(f"Using seed: {42}")
# 1. Load and preprocess data
#df = pd.read_csv("/home/acherjan/llama_experiments/astroque/merged_df_with_w3_w4_all_colors_complete.csv")
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
#cross-domain colors:
df['g-K'] = df['g_psf'] - df['k_m']

#numeric_columns = ['u_psf', 'v_psf', 'g_psf', 'r_psf', 'i_psf', 'z_psf', 'j_m', 'h_m', 'k_m','w1mpro','w2mpro','w3mpro','w4mpro'] 
# numeric_columns = ['u_psf','v_psf','g_psf','w1_w2_color','w3_w4_color'] # 1 anomaly
#numeric_columns = ['j_m', 'h_m', 'k_m','w3_w4_color','gi','vg','uv'] # 2 anomalous object ids 328922415 and 220721306
#numeric_columns = ['j_m', 'h_m', 'k_m','w3_w4_color', 'ur', 'vg','iz'] # 1 anomalous object id 220721306
#numeric_columns = ['j_m', 'h_m', 'k_m','w3_w4_color', 'zu','uv','ug'] #2 anomalies [np.int64(147793456), np.int64(220721306)]
# UV excess: u-g, u-v
# Optical slope: g-r, g-i
# Emission lines: g-i, r-z
# Reddening: v-g, r-z

print(df.columns)


#numeric_columns = ["ug","w3_w4_color","rz","j_m","h_m","k_m"]
# n_columns = [["ug","w3_w4_color","uv","j_m","h_m","k_m"],["ug","w1_w2_color","uv","j_m","h_m","k_m"],["gr","w3_w4_color","gi","j_m","h_m","k_m"],["gr","w1_w2_color","gi","j_m","h_m","k_m"],["gi","w3_w4_color","rz","j_m","h_m","k_m"],["gi","w1_w2_color","rz","j_m","h_m","k_m"],["vg","w3_w4_color","rz","j_m","h_m","k_m"],["vg","w1_w2_color","rz","j_m","h_m","k_m"]]
# n_columns = [
#     ["ug","w3_w4_color","uv","j_m-h_m","h_m-k_m","j_m-k_m"],
#     ["ug","w1_w2_color","uv","j_m-h_m","h_m-k_m","j_m-k_m"],
#     ["gr","w3_w4_color","gi","j_m-h_m","h_m-k_m","j_m-k_m"],
#     ["gr","w1_w2_color","gi","j_m-h_m","h_m-k_m","j_m-k_m"],
#     ["gi","w3_w4_color","rz","j_m-h_m","h_m-k_m","j_m-k_m"],
#     ["gi","w1_w2_color","rz","j_m-h_m","h_m-k_m","j_m-k_m"],
#     ["vg","w3_w4_color","rz","j_m-h_m","h_m-k_m","j_m-k_m"],
#     ["vg","w1_w2_color","rz","j_m-h_m","h_m-k_m","j_m-k_m"],
#     ["ug", "gi", "g-K", "j_m-h_m", "h_m-k_m", "w1_w2_color"]
# ]



n_columns = [["ug","uv","ui","ur","uz","vu","vg","vr","vi","vz","gu","gv","gr","gi","gz","ru","rv","rg","ri","rz","iu","iv","ig","ir","iz","zu","zv","zg","zr","zi"],["ug","uv","ui","ur","uz","vu","vg","vr","vi","vz","gu","gv","gr","gi","gz","ru","rv","rg","ri","rz","iu","iv","ig","ir","iz","zu","zv","zg","zr","zi"],["ug","uv","ui","ur","uz","vu","vg","vr","vi","vz","gu","gv","gr","gi","gz","ru","rv","rg","ri","rz","iu","iv","ig","ir","iz","zu","zv","zg","zr","zi"],["ug","uv","ui","ur","uz","vu","vg","vr","vi","vz","gu","gv","gr","gi","gz","ru","rv","rg","ri","rz","iu","iv","ig","ir","iz","zu","zv","zg","zr","zi"],["ug","uv","ui","ur","uz","vu","vg","vr","vi","vz","gu","gv","gr","gi","gz","ru","rv","rg","ri","rz","iu","iv","ig","ir","iz","zu","zv","zg","zr","zi"]]




#n_columns = [["ug","gi","rz","j_m-h_m","h_m-k_m","w1_w2_color"]]
for numeric_columns in n_columns:
        # if os.path.exists('Runs.csv') and ','.join(numeric_columns) in pd.read_csv('Runs.csv')['Variables'].values: 
        #     print("Variables already processed - moving to next set")
        #     continue #"Variables already processed - moving to next set"
        # print("Results for:   ",','.join(numeric_columns))
        # print("--------------------------------")
        # df_temp = pd.read_csv("Runs.csv")
        # print("Results for df:   ",df_temp['Variables'].values))
        # print("--------------------------------")
        # Impute missing values
        imputer = SimpleImputer(strategy='median')
        df[numeric_columns] = imputer.fit_transform(df[numeric_columns])

        # Normalize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df[numeric_columns].values)

        # Apply PCA
        n_components = 4  # You can adjust this - use less than original features
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X_scaled)

        # Print explained variance to help choose number of components
        print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
        print(f"Cumulative explained variance: {np.cumsum(pca.explained_variance_ratio_)}")

        object_ids = df['object_id'].values

        # 2. Define VAE (adjusted for PCA input dimension)
        class VAE(nn.Module):
            def __init__(self, input_dim, latent_dim=3):
                super(VAE, self).__init__()
                self.fc1 = nn.Linear(input_dim, 16)
                self.fc21 = nn.Linear(16, latent_dim)  # mean
                self.fc22 = nn.Linear(16, latent_dim)  # logvar
                self.fc3 = nn.Linear(latent_dim, 16)
                self.fc4 = nn.Linear(16, input_dim)

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

        # 3. Train VAE on PCA components
        input_dim = n_components  # Now using PCA components as input
        vae = VAE(input_dim)
        optimizer = optim.Adam(vae.parameters(), lr=1e-3)
        X_tensor = torch.tensor(X_pca, dtype=torch.float32)  # Using PCA components

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
            # Calculate reconstruction loss in PCA space
            losses_pca = nn.MSELoss(reduction='none')(recon, X_tensor).mean(dim=1).numpy()
            
            # Optional: Transform back to original space for interpretation
            X_recon_original = pca.inverse_transform(recon.numpy())
            X_original = scaler.inverse_transform(X_scaled)  # Original scaled data back to original space
            losses_original_space = np.mean((X_recon_original - df[numeric_columns].values)**2, axis=1)

        print(f"Using PCA space losses (recommended for anomaly detection)")
        losses = losses_pca

        # Alternative: Use original space losses for interpretability
        # print(f"Using original space losses (for interpretability)")
        # losses = losses_original_space

        # 5. Flag anomalies (e.g., top 1% highest losses)
        threshold = np.percentile(losses, 99)
        anomaly_indices = np.where(losses > threshold)[0]
        anomaly_object_ids = object_ids[anomaly_indices]

        print("Anomalous object IDs:", anomaly_object_ids)

        if os.path.exists('Runs_New_Set.csv'):
            group_number = 'Run_Group'+str(len(pd.read_csv('Runs_New_Set.csv'))+1)
            os.makedirs(group_number, exist_ok=True)
        else:
            group_number = 'Run_Group1'
            os.makedirs(group_number, exist_ok=True)
        
        # 6. Visualizations
        plt.figure(figsize=(15, 5))

        # Loss distribution
        plt.subplot(1, 3, 1)
        sns.histplot(losses, bins=50, kde=True)
        plt.axvline(threshold, color='red', linestyle='--', label=f'Anomaly threshold ({threshold:.4f})')
        plt.title('VAE Reconstruction Loss Distribution (PCA)')
        plt.xlabel('Reconstruction Loss (MSE)')
        plt.ylabel('Count')
        plt.legend()

        # Boxplot
        plt.subplot(1, 3, 2)
        sns.boxplot(x=losses)
        plt.title('Boxplot of VAE Reconstruction Loss (PCA)')
        plt.xlabel('Reconstruction Loss (MSE)')

        # Scatter plot
        plt.subplot(1, 3, 3)
        plt.scatter(range(len(losses)), losses, s=10, alpha=0.6, label='All objects')
        plt.scatter(anomaly_indices, losses[anomaly_indices], color='red', s=20, label='Anomalies')
        plt.title('Reconstruction Loss per Object (PCA)')
        plt.xlabel('Object Index')
        plt.ylabel('Reconstruction Loss (MSE)')
        plt.legend()

        plt.tight_layout()
        plt.savefig(group_number+'/vae_pca_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()

        # 7. Save results
        with open(group_number+'/anomalous_object_ids_pca.txt', 'w') as f:
            for oid in anomaly_object_ids:
                f.write(f"{oid}\n")

        # Save losses and PCA info
        loss_df = pd.DataFrame({
            'object_id': object_ids, 
            'reconstruction_loss_pca': losses_pca,
            'reconstruction_loss_original': losses_original_space if 'losses_original_space' in locals() else losses_pca
        })
        loss_df.to_csv(group_number+'/vae_pca_reconstruction_losses.csv', index=False)

        # Save PCA information
        pca_info = pd.DataFrame({
            'component': range(n_components),
            'explained_variance_ratio': pca.explained_variance_ratio_,
            'cumulative_variance': np.cumsum(pca.explained_variance_ratio_)
        })
        pca_info.to_csv(group_number+'/pca_component_info.csv', index=False)

        import os

        # Initialize variables to store overlapping anomalies for each dataset
        merc_galactic_overlaps = []
        merc_extragalactic_overlaps = []
        skymapper_confirmed_overlaps = []
        skymapper_candidates_overlaps = []

        #df2 = pd.read_csv("/home/acherjan/llama_experiments/astroque/matched_coordinates.csv")
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
        runs_csv_path = 'Runs_New_Set.csv'

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
        # 9. Additional PCA analysis
        plt.figure(figsize=(12, 4))

        # Plot explained variance
        plt.subplot(1, 2, 1)
        plt.plot(range(1, n_components+1), pca.explained_variance_ratio_, 'bo-')
        plt.title('Explained Variance Ratio by Component')
        plt.xlabel('Principal Component')
        plt.ylabel('Explained Variance Ratio')
        plt.grid(True)

        # Plot cumulative explained variance
        plt.subplot(1, 2, 2)
        plt.plot(range(1, n_components+1), np.cumsum(pca.explained_variance_ratio_), 'ro-')
        plt.title('Cumulative Explained Variance')
        plt.xlabel('Principal Component')
        plt.ylabel('Cumulative Explained Variance')
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(group_number+'/pca_variance_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()

df=pd.read_csv("Runs_New_Set.csv")
print("Combination with most confirmed anomalies:")
print(df.sort_values(by='Merc_Galactic', ascending=False).head(1))

print("Combination with most un-confirmed Skymapperanomalies:")
print(df.sort_values(by='Skymapper_candidates', ascending=False).head(1))

print("Combination with most total anomalies:")
print(df.sort_values(by='Total', ascending=False).head(1))