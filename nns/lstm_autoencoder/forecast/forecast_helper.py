import os
import pandas as pd
import glob

extension = 'csv'
all_filenames = [i for i in glob.glob('forecast*.{}'.format(extension))]
#combine all files in the list
combined_csv = pd.concat([pd.read_csv(f) for f in all_filenames ])
#export to csv
combined_csv.to_csv( "aquatics-2021-06-01-wbears_rnn.csv", index=False, encoding='utf-8-sig')
