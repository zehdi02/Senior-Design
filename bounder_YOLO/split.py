import os
import random
import shutil
import pandas as pd
# from details import calculate_statistics
from k_means_constrained import KMeansConstrained


# Function to split data using balanced K-means to create clusters of size 10 and split into 8/1/1
def split_manga109(annotations, stratify_columns):
    def split_df(df, rand=False):
        train_set, val_set, test_set = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        # loop through each row in the dataframe and assign to train/val/test set
        if rand:
            for _, row in df.iterrows():
                # get random value from 0-1 and assign to train/val/test set
                if random.random() <= 0.8:
                    train_set = pd.concat([train_set, row.to_frame().T], ignore_index=True)
                elif random.random() <= 0.9:
                    test_set = pd.concat([test_set, row.to_frame().T], ignore_index=True)
                else:
                    val_set = pd.concat([val_set, row.to_frame().T], ignore_index=True)
        else:
            train_set = df.sample(frac=0.8)
            val_set = df.drop(train_set.index).sample(frac=0.5)
            test_set = df.drop(train_set.index).drop(val_set.index)


    DATASET_TYPES = ['train', 'val', 'test']
    # Create train, val, test directories
    for dataset_type in DATASET_TYPES:
        os.makedirs(f'Manga109_YOLO/{dataset_type}/images', exist_ok=True)
        os.makedirs(f'Manga109_YOLO/{dataset_type}/labels', exist_ok=True)

    # loop through each manga in the annotations
    for manga in annotations['manga'].unique():
        manga_annotations = annotations[annotations['manga'] == manga]
        print(f"Processing manga: {manga}")

        for class_type in manga_annotations['type'].unique():
            print("processing class type: ", class_type)
            # get all annotations for the frame type
            df = manga_annotations[manga_annotations['type'] == class_type].copy()
            n_clusters = len(df) // 10

            # if not multiple of 10, randomly assign excess samples to train/val/test using 85/5/10 split
            if len(df) > n_clusters * 10:
                excess_samples = df.sample(n=len(df) - n_clusters * 10)
                df = df.drop(excess_samples.index)
                split_df(excess_samples, rand=True)
                print(excess_samples)

            # if after removing excess samples there are enough samples for clustering
            if len(df) > 10:
                # Use Balanced K-means to cluster the remaining data
                kmeans = KMeansConstrained(
                    n_clusters=n_clusters,
                    size_min=10,
                    size_max=10,
                )
                df['cluster'] = kmeans.fit_predict(df[stratify_columns])

                # Split data by cluster
                for cluster in df['cluster'].unique():
                    cluster_data = df[df['cluster'] == cluster]
                    split_df(cluster_data)

def main():
    ROOT_DIR = '../Manga109'
    NORMALIZED_ANNOTATIONS_CSV = os.path.join(ROOT_DIR, 'Manga109_processed.csv')
    annotations = pd.read_csv(NORMALIZED_ANNOTATIONS_CSV)
    # drop all except needed columns
    annotations = annotations[['manga', 'page_index', 'type', 'x_center', 'y_center', 'width', 'height', 'area', 'aspect_ratio']]
    # map type to class
    annotations['type'] = annotations['type'].map({'face': 0, 'body': 1, 'text': 2, 'frame': 3})
    stratify_columns = ['area', 'aspect_ratio']
    split_manga109(annotations, stratify_columns)

    # # Compare stats of each split against original to see how well stratification worked
    # stats = {'Manga109': calculate_statistics(annotations)}
    # stats['Train'] = calculate_statistics(train_df)
    # stats['Validation'] = calculate_statistics(val_df)
    # stats['Test'] = calculate_statistics(test_df)
    #
    # for split, split_stats in stats.items():
    #     print(f"Statistics for {split}:")
    #     for key, value in split_stats.items():
    #         print(f"{key}: {value}")
    #     print()

if __name__ == '__main__':
    main()
