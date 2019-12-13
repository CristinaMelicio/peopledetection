import pandas as pd
import glob
import os

CLASSES = [
    "Person", 
    "Handcart", 
    "Phone", 
    "Bag", 
    "Documents", 
    "Helmet", 
    "Card", 
    "Wheelchair"]

#create dataframe
dataset_dir = r'/media/nas/PeopleDetection/up_realsense_frames/Data_Seq/labels/Sharp_Dataset'
label_files = glob.glob(os.path.join(dataset_dir, "*.csv"))
df_from_each_file = (pd.read_csv(f) for f in label_files)
df = pd.concat(df_from_each_file, ignore_index=True, sort=False)

#calc stats
total_frames = df['frame'].nunique()
total_annotations = df['frame'].count()
class_dist = df['class'].value_counts()


#print stats
print('Total Frames: %d\n' % total_frames)
print('Total Annotations: %d\n' % total_annotations)
print('Total Class Distribution:')
for index, value in class_dist.items():
    print("   %s: %d" % (index, value))

    