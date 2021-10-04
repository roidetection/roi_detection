import os
# vipshome = 'C:\\Users\\liyao\\Documents\\vips-dev-8.10\\bin'
# os.environ['PATH'] = vipshome + ';' + os.environ['PATH']
import pyvips
import cv2
import h5py
import time
import torch
import argparse
import alphashape
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
from scipy import stats
from sklearn.cluster import OPTICS


parser = argparse.ArgumentParser(description='PCLA visualization script')
parser.add_argument('--exp_name', type=str, default='pcla_3class',help='experiment code for saving results')
parser.add_argument('--csv_path', type=str, default=r'D:\yaoli\detect_roi\dataset_csv\melanoma_binary_new.csv',
                        help='name of csv file that contains WSI slide ids')
parser.add_argument('--wsi_dir', type=str, default=r'D:\yaoli\data\melanoma\wsi',
                    help='data directory')
parser.add_argument('--results_dir', type=str, default=r'D:\yaoli\data\melanoma\pcla_3class',
                        help='folder in which to save model')
parser.add_argument('--xml_dir', type = str, default=r'D:\yaoli\data\melanoma\annotations\annotations_new', 
                    help='path to xml files')
parser.add_argument('--auto_skip', default=False, action='store_true')
parser.add_argument('--keep_bottom', default=False, action='store_true')
parser.add_argument('--patch_size', type=int, default=256)
parser.add_argument('--annotation_ratio', type=float, default=0.5, 
                    help='Ratio of annotated pixels to total pixels of small image patch to be counted as in region.')
parser.add_argument('--percent', type=float, default=0.2, help='top percentage of visualization')
args = parser.parse_args()
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

# map vips formats to np dtypes
format_to_dtype = {
    'uchar': np.uint8,
    'char': np.int8,
    'ushort': np.uint16,
    'short': np.int16,
    'uint': np.uint32,
    'int': np.int32,
    'float': np.float32,
    'double': np.float64,
    'complex': np.complex64,
    'dpcomplex': np.complex128,
}

# map np dtypes to vips
dtype_to_format = {
    'uint8': 'uchar',
    'int8': 'char',
    'uint16': 'ushort',
    'int16': 'short',
    'uint32': 'uint',
    'int32': 'int',
    'float32': 'float',
    'float64': 'double',
    'complex64': 'complex',
    'complex128': 'dpcomplex',
}

def combine_coordinates(contours_hull):
    contours = []
    for i in range(len(contours_hull[0])):
        contours.append([int(contours_hull[0][i]),int(contours_hull[1][i])])
    contours = np.array(contours)
    return contours

def get_percent_fromstat(slide_id, logfname):
    """
    Count the number of patches for each WSI image.
    """
    slide_ids = [line.rstrip('\n').split(' ')[0] for line in open(logfname)]
    num_patches = [int(line.rstrip('\n').split(' ')[1]) for line in open(logfname)]
    num_inregion = [int(line.rstrip('\n').split(' ')[3]) for line in open(logfname)]
    ind = slide_ids.index(slide_id)
    ratio = num_inregion[ind]/num_patches[ind]
    return ratio

def get_percent(hdf5_file_path, mask):
    """
    Count the number of patches for each WSI image.
    """
    coords, label =list(getPatchesReady(hdf5_file_path, percent=-1)) # -1 will get coords and label only
    num_inregion = 0
    for i in range(coords.shape[0]):
        p = coords[i]  ### patch coord
        p = p.astype('int32')
        img_mask = mask[p[1]:(p[1] + args.patch_size),p[0]:p[0] + args.patch_size]
        inregion = checkinout(img_mask, args.annotation_ratio)
        if inregion:
            num_inregion += 1    
    ratio = num_inregion/coords.shape[0]      
    return ratio


def log_check_file(logfname):
    """
    Count the number of patches for each WSI image.
    """
    lineList = [line.rstrip('\n').split(' ')[0] for line in open(logfname)]
    return lineList


def log_patches_count(logfname, slide_id, n, num_inregion_highlight, 
                      num_inregion, num_highlight):
    """
    Count the number of patches for each WSI image.
    """
    f = open(logfname, 'a+')
    f.write(slide_id+' '+str(n)+' '+str(num_inregion_highlight)+' '+str(num_inregion)+' '+str(num_highlight)+'\n')
    f.close()
    
    
def vips2numpy(vi):
    return np.ndarray(buffer=vi.write_to_memory(),
                      dtype=format_to_dtype[vi.format],
                      shape=[vi.height, vi.width, vi.bands])



# numpy array to vips image
def numpy2vips(a):
    height, width, bands = a.shape
    linear = a.reshape(width * height * bands)
    vi = pyvips.Image.new_from_memory(linear.data, width, height, bands,
                                      dtype_to_format[str(a.dtype)])
    return vi


def imopen(filename):
    im = pyvips.Image.new_from_file(filename, access="sequential")
    if not im.hasalpha():
        im = im.addalpha()
    return im


def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


def rgb(minimum, maximum, value):
    minimum, maximum = float(minimum), float(maximum)
    ratio = 2 * (value-minimum) / (maximum - minimum)
    b = int(max(0, 255*(1 - ratio)))
    r = int(max(0, 255*(ratio - 1)))
    g = 255 - b - r
    return r, g, b

def get_thrd(probs, percent):
    thrd = np.percentile(probs, int(100-percent*100))
    return thrd

def ScoreToColor(scores, percent, label):
    prob0 = scores[:,0]
    prob1 = scores[:,1]
    thrd0 = get_thrd(prob0, percent)
    thrd1 = get_thrd(prob1, percent)
    colorvalue = []
    label = label.item()
    for i in range(scores.shape[0]):
        if label ==0 and scores[i][0] > scores[i][1] and scores[i][0] > thrd0:
            colorvalue.append(0)
        elif label == 1 and scores[i][0] < scores[i][1] and scores[i][1] > thrd1:
            colorvalue.append(1)
        else:
            colorvalue.append(-1)
    return np.array(colorvalue)


def ScoreToRank(scores, label):
    prob0 = scores[:,0]
    prob1 = scores[:,1]
    label = label.item()
    if label == 0:
        probabilities_perc=[stats.percentileofscore(prob0, a, 'rank') for a in prob0]
    else:
        probabilities_perc=[stats.percentileofscore(prob1, a, 'rank') for a in prob1]
    probabilities_norm = NormalizeData(probabilities_perc)
    return probabilities_norm


def getPatchesReady(hdf5_file_path, percent):
    file = h5py.File(hdf5_file_path, 'r')
    scores = file['scores'][:]
    coords = file['coord'][:]
    label = file['pred'][:]
    file.close()
    if percent == -1:
        return coords, label.item()
    colorvalue = ScoreToColor(scores, percent, label)
    label = label.item()
    coords_highlight = []    
    for i in range(colorvalue.shape[0]):
        a = colorvalue[i] ### color value
        p = coords[i]  ### patch coord
        p = p.astype('int32')
        if a == label:
            coords_highlight.append(list(p)) 
    return np.array(coords_highlight)


def checkinout(patch_mask, annotation_ratio):
    """
    Checks whether or not an image chip should be saved
    :param mask: array of pixel in mask
    :return: bool
    """
    if np.sum(patch_mask>0) / patch_mask.size >= float(annotation_ratio):
        save = True
    else:
        save = False
    return save
          

def makemask(w, h, xml_path):
    """
    Reads xml file and makes annotation mask for entire slide image
    :param annotation_key: name of the annotation key file
    :param size: size of the whole slide image
    :param xml_path: path to the xml file
    :return: annotation mask
    :return: dictionary of annotation keys and color codes
    """
    # Import xml file and get root
    tree = ET.parse(xml_path)
    root = tree.getroot()
    # Generate annotation array and key dictionary
    mat = np.zeros((w, h), dtype='uint8')
    contours = []
    for reg in root.iter('Region'):
        points = []
        for child in reg.iter('Vertices'):
            for vert in child.iter('Vertex'):
                x = int(round(float(vert.get('X'))))
                y = int(round(float(vert.get('Y'))))
                points.append((x, y))
        cnt = np.array(points).reshape((-1, 1, 2)).astype(np.int32)
        cv2.fillPoly(mat, [cnt], 255)
        contours.append(cnt)
    return mat


def get_overlay(wsi_dir, attn_save_dir, overlay_save_dir, args):
    # Get attention score
    
    slide_ids = pd.read_csv(args.csv_path)['slide_id']
    
    for i in range(len(slide_ids)):
        slide_id = slide_ids[i]
        outputPath = os.path.join(overlay_save_dir,slide_id+'_con_'+args.exp_name+'.tiff')

        if args.auto_skip and os.path.isfile(outputPath):
            print('{} already exist in destination location, skipped'.format(slide_id))
            continue

        xml_path = os.path.join(args.xml_dir,slide_id+'.xml')
        hdf5_file_path = os.path.join(attn_save_dir, slide_id+'.h5')
        
        ### Read image ###
        wsi_image_path = os.path.join(wsi_dir, slide_id + '.svs')
        original_image = imopen(wsi_image_path)
                
        mask = makemask(original_image.height, original_image.width, xml_path)
        
        logfname_stat = os.path.join(args.results_dir,args.exp_name+'_stat.txt')
        if os.path.isfile(logfname_stat):
            percent = get_percent_fromstat(slide_id, logfname_stat)
        else:
            percent = get_percent(hdf5_file_path, mask)
        
        if args.keep_bottom and percent <= args.percent:
            percent = args.percent
            
        start = time.time()        
  
        ##### Get the coordinates of the largest cluster among highlighted patches ####
        coords_highlight = getPatchesReady(hdf5_file_path, percent)
        clustering =  OPTICS().fit(X=coords_highlight)
        cluster_labels = clustering.labels_
        unique, counts = np.unique(cluster_labels, return_counts=True)
        label_max_cluster = unique[np.argmax(counts[1:])+1]
        coords_max_cluster = coords_highlight[cluster_labels==label_max_cluster]
        
        #### convert image to Numpy array
        original_array = vips2numpy(original_image)
        #print("original_array Shape: ", original_array.shape)
        
        ######## Draw Contour Based on Cluster Coordinates ######### 
        
        alpha = 0.95 * alphashape.optimizealpha(coords_max_cluster)
        hull = alphashape.alphashape(coords_max_cluster, alpha)
        contours = combine_coordinates(hull.exterior.coords.xy)
        cv2.drawContours(original_array, [contours], -1, (0,255,0), 120, lineType=cv2.LINE_8)
        
        image_wcon = numpy2vips(original_array)
        ### Overlay
        image_wcon.write_to_file(outputPath, pyramid=True, tile=True, compression="jpeg")  
        end= time.time()
        print("Process {} took {} seconds".format(slide_id, end - start))
    return



# Create attention map
overlay_save_dir = os.path.join(args.results_dir, 'overlay')
attn_save_dir = os.path.join(args.results_dir, 'attention')
wsi_dir = args.wsi_dir

os.makedirs(overlay_save_dir, exist_ok=True)

# Get ROI overlay
print("=====>Get ROI on train dataset")
get_overlay(wsi_dir, attn_save_dir, overlay_save_dir, args)






