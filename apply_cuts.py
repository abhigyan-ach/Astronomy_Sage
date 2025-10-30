import pandas as pd
import numpy as np
from astroquery.gaia import Gaia

def gaia_bailer_jones_distances(input_file, batch_size=10000):
    # First read the input data to get source IDs
    print("Reading input data to get source IDs...")
    df_input = pd.read_csv(input_file)
    source_ids = df_input['source_id'].tolist()
    total_ids = len(source_ids)
    print(f"Found {total_ids} source IDs in input data")
    
    # Initialize empty list to store results
    all_results = []
    
    # Process in batches
    for i in range(0, total_ids, batch_size):
        batch_ids = source_ids[i:i + batch_size]
        batch_ids_str = ','.join(map(str, batch_ids))
        print("Batch size:",len(batch_ids))
        
        query = f"""
        SELECT TOP 100
            g.source_id,
            d.r_est, 
            d.r_lo, 
            d.r_hi,
            d.r_len,
            d.result_flag,
            d.modality_flag
        FROM external.gaiadr2_geometric_distance d
        JOIN gaiadr2.gaia_source g ON g.source_id = d.source_id
        WHERE g.source_id IN ({batch_ids_str})
        """
        
        print(f"\nProcessing batch {i//batch_size + 1} of {(total_ids + batch_size - 1)//batch_size}")
        print(f"Querying {len(batch_ids)} source IDs...")
        
        try:
            job = Gaia.launch_job_async(query)
            results = job.get_results()
            batch_df = results.to_pandas()
            all_results.append(batch_df)
            print(f"Successfully retrieved {len(batch_df)} results for this batch")
        except Exception as e:
            print(f"Error processing batch: {str(e)}")
            continue
    
    # Combine all results
    if all_results:
        df = pd.concat(all_results, ignore_index=True)
        print(f"\nTotal results retrieved: {len(df)}")
        
        # Save to CSV
        output_file = "bailer_jones_distances.csv"
        print(f"Saving results to {output_file}...")
        df.to_csv(output_file, index=False)
        print(f"Saved {len(df)} rows to {output_file}")
        
        return df
    else:
        print("No results were retrieved")
        return None

def gaia_table_access():
    Gaia.MAIN_GAIA_TABLE = "gaiadr2.gaia_source" 
    #dr3_table = Gaia.load_table('external.gaia_dr3')
    dr3_table =Gaia.load_table('gaiadr2.gaia_source')
    for column in dr3_table.columns:
       print(column.name, ":",column.description)
    # for column in dr3_table.columns:
    #     print(column.name)
  

def cut1(df):
    """Apply cut: t.j_m < 14.0"""
    return df[df['j_m'] < 14.0]

def cut2(df):
    """Apply cut: (t.j_m - 5 * log(100.0/g.parallax)) < 0.0"""
    return df[df['j_m'] - 5 * np.log(100.0/df['parallax']) < 0.0]

def cut3(df):
    """Apply cut: (t.j_m - t.k_m - 0.413*(m.ebmv_sfd)) > 0.85"""
    return df[(df['j_m'] - df['k_m'] - 0.413*df['ebmv_sfd']) > 0.85]

def cut4(df):
    """Apply cut using Bailer-Jones distances: j_m - 5 * log(r_est) < 0.0"""
    return df[df['j_m'] - 5 * np.log(df['r_est']) < 0.0]

def apply_all_cuts(input_file, output_file, bailer_jones_file):
    # Read the CSV files
    print("Reading the input CSV file...")
    df = pd.read_csv(input_file)
    initial_count = len(df)
    print(f"Initial number of rows in input file: {initial_count}")
    
    # # Read Bailer-Jones distances
    print("\nReading Bailer-Jones distances...")
    df_bailer_jones = pd.read_csv(bailer_jones_file)
    print("Number of source ids in Bailer-Jones distances:",len(df_bailer_jones))
    # Merge the dataframes
    # print("Merging with Bailer-Jones distances...")
    df = pd.merge(df, df_bailer_jones[['source_id', 'r_est']], 
                   on='source_id', how='inner')
    print(f"Number of rows after merging: {len(df)}")
    
    # Apply cuts sequentially
    print("\nApplying cuts...")
    
    # Cut 1
    #df = cut1(df)
    #print(f"After Cut 1 (j_m < 14.0): {len(df)} rows")
    
    # Cut 2
    #df = cut2(df)
   # print(f"After Cut 2 (j_m - 5*log(100/parallax) < 0.0): {len(df)} rows")
    
    # Cut 3
    df = cut3(df)
    print(f"After Cut 3 (j_m - k_m - 0.413*ebmv_sfd > 0.85): {len(df)} rows")
    
    #Cut 4
    df = cut4(df)
    print(f"After Cut 4 (j_m - 5*log(r_est) < 0.0): {len(df)} rows")
    
    # Save the filtered data
    print(f"\nSaving filtered data to {output_file}...")
    df.to_csv(output_file, index=False)
    print("Done!")
    
    return len(df)

if __name__ == "__main__":
    #input_file = "/Users/abhiacherjee/Desktop/DSI/Dr2_table.csv"
    input_file = "/Users/abhiacherjee/Desktop/Docs/DSI/Dr2_table_w3w4.csv"

    df = pd.read_csv(input_file)
    print(df.shape)
    output_file = "filtered_data_ebmv_cut_Dr2_table_after_bailer_jones.csv"
    
    # Get Bailer-Jones distances for matching source IDs
    #df_bailer_jones = gaia_bailer_jones_distances(output_file, batch_size=10000)
    #apply_all_cuts(input_file, output_file)
    # df_bailer_jones = gaia_bailer_jones_distances(output_file, batch_size=10000)
    
    # if df_bailer_jones is not None:
    #     # Apply all cuts
    final_count = apply_all_cuts(input_file, output_file, "bailer_jones_distances_final.csv")
    print(f"\nFinal number of rows in the filtered dataset: {final_count}")
    #gaia_table_access()
    # tables = Gaia.load_tables(only_names=True)
    # for table in tables:
    #    print(table.get_qualified_name())
    # df_bailer_jones = gaia_bailer_jones_distances()