import os
import time
import argparse
import pandas as pd
from utils.dataset_h5 import Dataset_All_Bags, h5_to_patch


def log_check_file(logfname):
    """
    Count the number of patches for each WSI image.
    """
    lineList = [line.rstrip('\n').split(' ')[0] for line in open(logfname)]
    return lineList

def log_patches_count(logfname, filename, count):
    """
    Count the number of patches for each WSI image.
    """
    f = open(logfname, 'a+')
    f.write(filename+' '+str(count)+'\n')
    f.close()

def make_dirs(args):
    os.makedirs(args.feat_dir, exist_ok=True)
    for name in ["train", "val", "test"]:
        os.makedirs(os.path.join(args.feat_dir,name), exist_ok=True)
        for label in ["Melanoma", "Nevi", "Other"]:
            os.makedirs(os.path.join(args.feat_dir,name, label), exist_ok=True) 
    return

def compute_w_loader(file_path, output_path, bag_name, logfname, target_patch_size=-1):
    """
    args:
        file_path: directory of bag (.h5 file)
        output_path: directory to save extracted patches
        model: pytorch model
        batch_size: batch_size for computing features in batches
        verbose: level of feedback
    """
    dataset = h5_to_patch(file_path=file_path, target_patch_size=target_patch_size)
    n = len(dataset)
    for i in range(n):
        x, y = dataset[i]
        save_path = '_'.join([output_path,str(y[0]),str(y[1])])+'.png'
        x.save(save_path)
    log_patches_count(logfname, bag_name.rstrip(".h5"), n)
    return


parser = argparse.ArgumentParser(description='Feature Extraction')
parser.add_argument('--data_dir', type=str, default=None)
parser.add_argument('--other_patches_dir', type=str, default=None)
parser.add_argument('--csv_path', type=str, default=None)
parser.add_argument('--feat_dir', type=str, default=None)
parser.add_argument('--auto_skip', default=False, action='store_true')
parser.add_argument('--target_patch_size', type=int, default=-1,
                    help='the desired size of patches for optional scaling before feature embedding')
args = parser.parse_args()


if __name__ == '__main__':

    print('initializing dataset')
    csv_path = args.csv_path
    df = pd.read_csv(csv_path)
    bags_dataset = Dataset_All_Bags(args.data_dir, csv_path)
    logfname = os.path.join(args.feat_dir,'patch_count.txt')
    logfname_other = os.path.join(args.feat_dir,'patch_count_other.txt')
    if args.auto_skip:
        processed_list = log_check_file(logfname)        
    else:
        processed_list = []
        
    make_dirs(args)

    total = len(bags_dataset)
    
    for bag_candidate_idx in range(total):
        bag_candidate = bags_dataset[bag_candidate_idx]
        bag_name = os.path.basename(os.path.normpath(bag_candidate))
        target_folder = os.path.join(args.feat_dir,df['data_split'][bag_candidate_idx], df['label_name'][bag_candidate_idx])
        if '.h5' in bag_candidate:

            print('\nprogress: {}/{}'.format(bag_candidate_idx, total))
            print(bag_name)
            if args.auto_skip and bag_name.rstrip(".h5") in processed_list:
                print('skipped {}'.format(bag_name))
                continue

            output_path = os.path.join(target_folder, bag_name.split('.')[0])
            file_path = bag_candidate
            time_start = time.time()
            output_file_path = compute_w_loader(file_path, output_path, bag_name, logfname,
                                                target_patch_size=args.target_patch_size)
            time_elapsed = time.time() - time_start
            print('\nsplitting patches for {} took {} s'.format(output_file_path, time_elapsed))
        
        other_patch_path = os.path.join(args.other_patches_dir, bag_name)
        if os.path.isfile(other_patch_path):

            print('\nprocessing {} to extract other patches'.format(bag_name))
            
            other_target_folder = os.path.join(args.feat_dir,df['data_split'][bag_candidate_idx], 'Other')
            output_path = os.path.join(other_target_folder, bag_name.split('.')[0])
            file_path = other_patch_path
            time_start = time.time()
            output_file_path = compute_w_loader(file_path, output_path, bag_name, logfname_other,
                                                target_patch_size=args.target_patch_size)
            time_elapsed = time.time() - time_start
            print('\nsplitting patches for {} took {} s'.format(output_file_path, time_elapsed))
