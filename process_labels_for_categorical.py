import pickle

import pandas as pd


def process_labels_for_categorical(path_to_label_csv: str) -> pd.DataFrame:
    # Use this code to create a .csv file with the necessary format needed for
    # categorical emotion recognition model

    # Load Original label_consensus.csv file provided with dataset
    df = pd.read_csv(path_to_label_csv)

    # Define the emotions
    emotions = ["Angry", "Sad", "Happy", "Surprise", "Fear", "Disgust", "Contempt", "Neutral"]
    emotion_codes = ["A", "S", "H", "U", "F", "D", "C", "N"]

    # Create a dictionary for one-hot encoding
    one_hot_dict = {e: [True if e == ec else False for ec in emotion_codes] for e in emotion_codes}

    # Save the one-hot dictionary to a pickle file
    one_hot_dict_path = path_to_label_csv.replace('.csv', '_one_hot_dict.pkl')
    with open(one_hot_dict_path, 'wb') as f:
        pickle.dump(one_hot_dict, f)
    print(f"One-hot dictionary saved as {one_hot_dict_path}")

    # Filter out rows with undefined EmoClass
    df = df[df['EmoClass'].isin(emotion_codes)]

    # Apply one-hot encoding
    for i, e in enumerate(emotion_codes):
        df[emotions[i]] = df['EmoClass'].apply(lambda x: one_hot_dict[x][i])

    # Select relevant columns for the new CSV
    df_final = df[['FileName', *emotions, 'Split_Set']]

    # Save the processed data to a new CSV file
    path_label_processed = path_to_label_csv.replace('.csv', '_processed.csv')
    df_final.to_csv(path_label_processed, index=False)

    print(f"Processing complete. New file saved as {path_label_processed}")
    return df_final
