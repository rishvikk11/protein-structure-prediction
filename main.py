import extract_protein_data
from model import run_training_pipeline, ProteinVisualizer
import random
import pandas as pd
import pickle


def main():
    # Step 1: Download PDB files
    pdb_df = pd.read_csv('pdb_data_seq.csv')
    pdb_protein_df = pdb_df[pdb_df['macromoleculeType'] == 'Protein']
    pdb_ids = pdb_protein_df['structureId'].tolist()
    random_pdb_ids = random.sample(pdb_ids, 6000)
    extract_protein_data.download_pdb_files(random_pdb_ids)

    # Step 2: Extract, clean, and normalize the data
    # This is all handled inside extract_protein_data.py when it's imported
    extract_protein_data.load_cleaned_data()

    # After loading data, verify
    with open('cleaned_protein_data.pkl', 'rb') as f:
        data = pickle.load(f)
        print(f"Total proteins saved: {len(data)}")
        print("Sample protein:", data[0])

    # Step 3: Train & test the model
    run_training_pipeline()

    visualizer = ProteinVisualizer()
    
    # Example sequence or use your own
    #"MVLSEGEWQLVLHVWAKVEADVAGHGQDILIRLFKSHPETLEKFDRVKHLKTEAEMKASEDLKKHGVTVLTALGAILKKKGHHEAELKPLAQSHATKHKIPIKYLEFISEAIIHVLHSRHPGNFGADAQGAMNKALELFRKDIAAKYKELGYQG"
    test_sequence = input("Enter desired amino acid sequence: ")
    
    # Predict structure
    coords = visualizer.predict_from_sequence(test_sequence)
    
    # Visualize
    visualizer.interactive_3d_view(coords, test_sequence)

if __name__ == "__main__":
    main()
    
