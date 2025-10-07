import pandas as pd
import numpy as np

odds_df = pd.read_csv('mlb_totals_odds.csv')

odds_df = odds_df.replace('Cleveland Indians','Cleveland Guardians')

odds_df.to_csv('cleaned_mlb_total_odds.csv')
