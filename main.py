import preprocessing as pp
import embedding as em
import json


dataset_path = './dataset'
processed_path = './processed'

# pp.process_dataset(dataset_path, processed_path)
pp.plot_histogram(processed_path)
# em.create_dictionary(processed_path)
