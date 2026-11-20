# Standard R-SIM split

The three Excel tables in `raw/` describe the RNAs, ligands and labeled interactions. The zero-based indices in `folds.json` refer to the interaction table in its original row order. Each fold has separate training, validation and test indices.

The tensors in `features/` are keyed by entity ID. Run `python create_data.py` from the repository root to assemble `processed/final_data_random.pt`.
