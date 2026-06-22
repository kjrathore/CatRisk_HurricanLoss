# src/financial_loss.py
import os
import yaml
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

class FinancialLossEngine:
    def __init__(self, config_path: str = "config.yaml"):
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found at {config_path}.")
            
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
            
        self.mode = self.config["model_settings"]["resolution_mode"]
        self.damage_path = f"./data/{self.mode}/portfolio_damage_matrix.parquet"
        self.exposure_path = "./data/portfolio_exposure.csv"
        self.output_dir = f"./data/{self.mode}"
        
        os.makedirs(self.output_dir, exist_ok=True)

    def calculate_losses(self):
        print(f"\n--- Initializing Pure-Stream Financial Loss Engine [{self.mode.upper()} Mode] ---")
        
        if not os.path.exists(self.damage_path):
            raise FileNotFoundError(f"Damage matrix missing at {self.damage_path}.")
        if not os.path.exists(self.exposure_path):
            raise FileNotFoundError(f"Exposure data file missing at {self.exposure_path}.")

        # 1. Load small exposure data framework once
        df_exposure = pd.read_csv(self.exposure_path)
        df_exposure.columns = [c.lower().strip() for c in df_exposure.columns]
        
        # DYNAMIC LOOKUP: Find the Total Insured Value (TIV) column safely
        tiv_candidates = [c for c in df_exposure.columns if 'tiv' in c or 'value' in c or 'insured' in c]
        if not tiv_candidates:
            raise KeyError(f"Could not find a valuation or TIV column in exposure file. Available columns: {list(df_exposure.columns)}")
        tiv_col = tiv_candidates[0]
        print(f"-> Automatically mapped portfolio valuation data from column: '{tiv_col}'")
        
        # Build strict string-indexed maps for constant-time O(1) lookups
        val_map = dict(zip(df_exposure['property_id'].astype(str), df_exposure[tiv_col].astype(float)))
        deduct_map = dict(zip(df_exposure['property_id'].astype(str), df_exposure.get('deductible', pd.Series(0.0, index=df_exposure.index)).astype(float)))
        limit_map = dict(zip(df_exposure['property_id'].astype(str), df_exposure.get('policy_limit', df_exposure[tiv_col]).astype(float)))

        output_file = f"{self.output_dir}/portfolio_financial_matrix.parquet"
        
        # 2. Open file for strict streaming (Never loads the full table into RAM)
        parquet_file = pq.ParquetFile(self.damage_path)
        
        schema = pa.schema([
            ('event_id', pa.string()),
            ('property_id', pa.string()),
            ('ground_up_loss', pa.float64()),
            ('insured_loss', pa.float64())
        ])
        
        total_rows = 0
        
        # 3. Stream small mini-batches (e.g., 20,000 rows max per loop) to keep RAM footprint minimal
        with pq.ParquetWriter(output_file, schema, compression='snappy') as writer:
            print("Streaming rows through low-overhead PyArrow batches...")
            
            for batch in parquet_file.iter_batches(batch_size=20000, columns=['event_id', 'property_id', 'damage_ratio']):
                
                # Convert only the columns we need to raw arrays (avoids full DataFrame overhead)
                event_ids = batch.column('event_id').to_pylist()
                prop_ids = batch.column('property_id').to_pylist()
                damage_ratios = batch.column('damage_ratio').to_numpy()
                
                # Fast array allocations using dict definitions
                tiv = np.array([val_map.get(pid, 0.0) for pid in prop_ids], dtype=np.float64)
                deductibles = np.array([deduct_map.get(pid, 0.0) for pid in prop_ids], dtype=np.float64)
                limits = np.array([limit_map.get(pid, t) for pid, t in zip(prop_ids, tiv)], dtype=np.float64)
                
                # Financial core calculations
                ground_up_loss = tiv * damage_ratios
                loss_less_deductible = np.maximum(ground_up_loss - deductibles, 0.0)
                insured_loss = np.minimum(loss_less_deductible, limits)
                
                # Reconstruct directly to PyArrow arrays without intermediate Pandas steps
                out_batch = pa.RecordBatch.from_arrays([
                    pa.array(event_ids, type=pa.string()),
                    pa.array(prop_ids, type=pa.string()),
                    pa.array(ground_up_loss, type=pa.float64()),
                    pa.array(insured_loss, type=pa.float64())
                ], schema=schema)
                
                writer.write_batch(out_batch)
                total_rows += len(batch)
                
        print(f"SUCCESS: Financial loss calculation complete. Processed {total_rows:,} records without RAM inflation.")

        
if __name__ == "__main__":
    engine = FinancialLossEngine()
    engine.calculate_losses()