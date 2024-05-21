import pandas as pd
import os
import matplotlib.pyplot as plt
def plot_pandas(csv_file1,csv_file2, save_path,interval):
    # Read the CSV file into a pandas DataFrame
    df1 = pd.read_csv(csv_file1)
    df2 = pd.read_csv(csv_file2)
    # if not all(col in df.columns for col in expected_columns):
    #     raise ValueError("Error: Column names in CSV file do not match expected names.")
    # Extract 'episode' and 'return' columns

    # Extract 'epoch' and 'return' columns
    episode = df1['games'].iloc[::interval]
    return_value1 = df1['Mean reward'][::interval]
    return_value2 = df2['Mean reward'][::interval]

    # Plot episode-to-return graph
    plt.plot(episode, return_value1, label='baseline')
    plt.plot(episode, return_value2, label='Improved')
    plt.xlabel('Games')
    plt.ylabel('Mean Rewards')
    plt.legend()
    plt.title('TD(0) learning with different approach')
    plt.grid(True)
    
    plt.show
    plt.savefig(save_path)


if __name__ == '__main__':
    SAVE_PATH = os.path.abspath(os.path.dirname(__file__))

    csv_path1=  os.path.join(SAVE_PATH,'reward_base.csv')
    csv_path2=  os.path.join(SAVE_PATH,'reward_imp.csv')

    plot_pandas(csv_path1, csv_path2, 'reward.png',1)