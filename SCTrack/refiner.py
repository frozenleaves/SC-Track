# -*- coding: utf-8 -*-
"""
Created on Thu Aug 1 12:08:06 2024
Refiner module to correct the tracking results from sctrack. This 
module will help fix the tracking error caused by sctrack due to occasional
false segmentations and miss assignments of cell tracking events.
@author: KuanY
"""

import ast # to convert a string representation of a list into a list
import re


import pandas as pd
from skimage import exposure, img_as_ubyte
import math
import numpy as np
from matplotlib.path import Path
from tqdm import tqdm
import cv2
import tifffile


def calculate_polygon_area(x, y):
    """
    Calculate the area of the polygon defined by a list of x and y coordinates 
    using the Shoelace formula
    """
    n = len(x)
    area = 0.0
    
    for i in range(n):
        j = (i + 1) % n
        area += x[i] * y[j] - y[i] * x[j]
        
    area = abs(area) / 2.0
    return area

def convert_to_list(string_list):
    """
    convert string_list into list
    """
    list_item = [ast.literal_eval(item) for item in string_list]
    
    return list_item


def calculate_distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)


def calc_euclidean_distance_between_frames(list_x, list_y):
    """
    Takes a list of centroid values of tracked object as a list of x and y values
    and calculate the distance between the points over time with the first frame 
    having a nan value
    """
    list_x1 = list_x[:-1]
    list_y1 = list_y[:-1]
    list_x2 = list_x[1:]
    list_y2 = list_y[1:]
    
    distance_list = [calculate_distance(x1, y1, x2, y2) for x1, y1, x2, y2 in zip(list_x1, list_y1, list_x2, list_y2)]
    distance_list.insert(0, float("nan"))
    return distance_list
    
def calc_difference_between_frames(ls):
    """
    Takes values of a timeseries of a tracked object and calculates the absolute
    difference between the current frame and the subsequent frame with the first 
    frame having a nan value
    """
    
    ls1 = ls[:-1]
    ls2 = ls[1:]
    
    diff_list = [abs(a1 - a2) for a1, a2 in zip(ls1, ls2)]
    diff_list.insert(0, float("nan"))
    return diff_list

def count_overlap(x1_points, y1_points, x2_points, y2_points, resolution=1):
    """
    Takes list of x and y points from two polygons and calculates the number
    of pixel overlap between both points. Adjust the resolution parameter to 
    change the precision of the overlap count. A lower resolution will be faster 
    but less precise, while a higher resolution will be slower but more precise.
    """
    
    # Create polygon paths from the given points
    polygon1_points = np.vstack((x1_points, y1_points)).T
    polygon2_points = np.vstack((x2_points, y2_points)).T
    polygon1_path = Path(polygon1_points)
    polygon2_path = Path(polygon2_points)
    
    # Determine the bounding box of the combined area of both polygons
    all_x_points = np.concatenate((x1_points, x2_points))
    all_y_points = np.concatenate((y1_points, y2_points))
    min_x, min_y = np.min(all_x_points), np.min(all_y_points)
    max_x, max_y = np.max(all_x_points), np.max(all_y_points)
    
    # Generate a grid of points within the bounding box
    x_grid, y_grid = np.meshgrid(np.arange(min_x, max_x, resolution), 
                                 np.arange(min_y, max_y, resolution))
    grid_points = np.vstack((x_grid.ravel(), y_grid.ravel())).T
    
    # Check which grid points are inside both polygons
    mask1 = polygon1_path.contains_points(grid_points)
    mask2 = polygon2_path.contains_points(grid_points)
    
    # Count the number of points inside both polygons
    overlap_mask = mask1 & mask2
    pixel_overlap_count = np.sum(overlap_mask)
    
    return pixel_overlap_count

def relative_overlap(pixel_overlap_count, pixel_area_1, pixel_area_2):
    """
    Calculates the relative overlap of the objects relative to the larger object.
    Expressed in fractions.
    """
    relative_size = pixel_overlap_count / max([pixel_area_1, pixel_area_2])
    
    return relative_size
    

def search_range(points, centroid_value, treshold):
    """
    Takes either x or y points and centroid value of the segmented object and 
    calculate the upper and lower bound values based on the treshold value.
    """
    min_point = min(points)
    max_point = max(points)
    distance_treshold = (max_point - min_point) * treshold
    lower_treshold = centroid_value - distance_treshold
    upper_treshold = centroid_value + distance_treshold
    
    return lower_treshold, upper_treshold
    
def remove_segment_errors_from_df(dataframe, tolerance = 2):
    """
    Removes abberant instance segmentations that copied into multiple frames.
    The function will look for duplicated centroid x and y values and will
    scan the dataframe to delete instances where both x and y centroid values
    are detected together.
    
    A tolerance of 2 is set by default as there are some instances where the duplicates
    are useful to maintain the occasional loss of missing segmentations.
    """
    
    # pull out central points for all objects to figure out duplicated objects
    centx = dataframe["center_x"].values.tolist()
    centy = dataframe["center_y"].values.tolist()
    
    # identify duplicated mask from x and y position. remove masks that are duplicated more than once.
    duplicate_x = []
    for num in centx:
        if centx.count(num) > tolerance and num not in duplicate_x:
            duplicate_x.append(num)
    
    duplicate_y = []
    for num in centy:
        if centy.count(num) > tolerance and num not in duplicate_y:
            duplicate_y.append(num)
    
    # delete duplicated objects from dataframe
    for x, y in zip(duplicate_x, duplicate_y):
        index_names = dataframe[(dataframe["center_x"] == x) & (dataframe["center_y"] == y)].index
        dataframe.drop(index_names, inplace = True)
        
    return dataframe


def preprocess_dataframe(dataframe):
    """
    Perform dataframe preprocessing to remove segmentation errors and convert
    mask_of_points into list instead of a string object.
    """
    dataframe = remove_segment_errors_from_df(dataframe)
    
    dataframe = dataframe.sort_values(by = ["cell_id", "frame_index"])
    
    dataframe["mask_of_x_points"] = convert_to_list(dataframe["mask_of_x_points"].values.tolist())
    dataframe["mask_of_y_points"] = convert_to_list(dataframe["mask_of_y_points"].values.tolist())
    
    return dataframe

def extract_similarity_and_overlap(dataframe, area_last_instance, previous_x_points, previous_y_points, index = -1):
    """
    Function to compute the degree of similarity in segmented area as well as the amount of overlap 
    of candidate object in subsequent frame compared to segmented object that is being linked. Will 
    return the values as float.
    """
    
    candidate_x_points = dataframe.iloc[index,7]
    candidate_y_points = dataframe.iloc[index,8]
    size_of_candidate_instance = calculate_polygon_area(candidate_x_points, candidate_y_points)
    
    # compute differences in area size and amount of relative overlap of objects
    relative_size_difference = min([area_last_instance, size_of_candidate_instance]) / max([area_last_instance, size_of_candidate_instance])
    overlap = relative_overlap(count_overlap(previous_x_points, previous_y_points, candidate_x_points, candidate_y_points), area_last_instance, size_of_candidate_instance)
    
    return relative_size_difference, overlap



def refine_cell_track(dataframe, treshold = 0.3, similarity_treshold = 0.85, overlap_treshold = 0.75):
    """
    Refines the tracking results to correct for reoccuring occasional tracking 
    errors. Takes the dataframe output from the sctrack track_tree_to_table function
    and returns a refined track tree with corrected tracks. 
    
    The function measures the relative movement of segmented cells from the last
    frame of the tracked object and checks the next frame for the presence of 
    a cell segmentation within a relative distance from the centroid x and y.
    
    If an segmented object is found, it will calculate the relative segmented area
    of the last frame of the segmented object and compare the area as well as the 
    relative overlap in segmentation between the previous frame and current frame.
    If the relative overlap and relative overlap exceeds the set treshold. It will
    be considered that there was an error in the track assignment and the 
    corresponding track ID for the subsequent frames corrected.
    """
 
    # pull out unique cell ids for track correction
    counts_df = dataframe["cell_id"].value_counts(dropna = False).to_frame(name = "counts")
    counts_df["cell_id"] = counts_df.index
    cell_list = counts_df[counts_df["counts"] > 10]["cell_id"].tolist()
    cell_list.sort()
    
    edit_counts = 0
     
    for i in tqdm(range(len(cell_list))):
        
        # pull out the single cell lineage from the previous instance as temp_df1
        cell = cell_list[i]
        
        # pull out updated list of cell ids from df
        latest_cell_ids = dataframe["cell_id"].values.tolist()
        
        
        if cell not in latest_cell_ids:
            continue
        
        temp_df1 = dataframe[dataframe["cell_id"] == cell]
        
        
        # extract center_x, center_y, last_x_points, last_y_points, last_frame values
        last_instance_x = temp_df1.iloc[-1,4]
        last_instance_y = temp_df1.iloc[-1,5]
        last_x_points = temp_df1.iloc[-1,7]
        last_y_points = temp_df1.iloc[-1,8]
        last_frame = temp_df1.iloc[-1,0]
        lower_x, upper_x = search_range(last_x_points, last_instance_x, treshold)
        lower_y, upper_y = search_range(last_y_points, last_instance_y, treshold)
        size_of_last_instance = calculate_polygon_area(last_x_points, last_y_points)
        
        # extract the subsequent frame for analysis. stored as temp_df2
        temp_df2 = None
        temp_df2 = dataframe[dataframe["frame_index"] == last_frame + 1]
        
        # Skip if no valid next frame is found
        if temp_df2 is None or len(temp_df2) == 0:
            continue
        
        
        # pull out single cell linstance that is within the search instance and store in temp_df3
        mask = (temp_df2["center_x"] > lower_x) & (temp_df2["center_x"] < upper_x) & (temp_df2["center_y"] > lower_y) & (temp_df2["center_y"] < upper_y)
        temp_df3 = temp_df2[mask]
        
        # conditional to skip following steps if there are no candidate cells that pass the treshold
        if len(temp_df3) == 0:
            # print(f"There is no candidates identified for {cell} in {last_frame}")
            continue
        
        if len(temp_df3) == 1:
            # print("There is a candidate cell")
            
            relative_size_difference, overlap = extract_similarity_and_overlap(temp_df3, size_of_last_instance, last_x_points, last_y_points, index = -1)
            
            # simple logical filter if the object overlap size and size similarity is above the set treshold
            is_same_cell = (relative_size_difference > similarity_treshold) & (overlap > overlap_treshold)
            
            # pull out candidate cell id
            candidate_id = temp_df3.iloc[0,2]
            
            if is_same_cell:
                # print(f"Match found for {cell}, changing {candidate_id} from frame {last_frame + 1}")
                
                # correct cell cell id of switch
                selection = (dataframe["cell_id"] == candidate_id) & (dataframe["frame_index"] >= last_frame + 1)
                current_parent_id = dataframe[dataframe['cell_id'] == cell]['parent_id'].iloc[0]
                dataframe.loc[selection, 'parent_id'] = current_parent_id
                dataframe.loc[selection, 'cell_id'] = cell
                
                track_new = re.match(r'^(.*)_[0-9]+$', cell).group(1)  #new track_id
                dataframe.loc[selection, 'track_id'] = track_new
                
                edit_counts = edit_counts + 1
        
        elif len(temp_df3) == 2:
            # print(f"There is two candidates for {cell}")
            
            relative_size_difference_1, overlap_1 = extract_similarity_and_overlap(temp_df3, size_of_last_instance, last_x_points, last_y_points, index = 0)        
            relative_size_difference_2, overlap_2 = extract_similarity_and_overlap(temp_df3, size_of_last_instance, last_x_points, last_y_points, index = 1)
            
            
            # simple logical filter if the object overlap size and size similarity is above the set treshold
            is_same_cell_1 = (relative_size_difference_1 > similarity_treshold) & (overlap_1 > (overlap_treshold - 0.2))
            is_same_cell_2 = (relative_size_difference_2 > similarity_treshold) & (overlap_2 > (overlap_treshold - 0.2))
            
            candidate_id = None
            
            if is_same_cell_1:             
                # pull out candidate cell id
                candidate_id = temp_df3.iloc[0,2]
                
                # print(f"Match found for {cell}, changing {candidate_id} from frame {last_frame + 1}")
                
            elif is_same_cell_2:
                # pull out candidate cell id
                candidate_id = temp_df3.iloc[1,2]
                
                # print(f"Match found for {cell}, changing {candidate_id} from frame {last_frame + 1}")
            
            else:
                # print(f"No matches found for {cell} in frame {last_frame}, consider reducing stringency of tresholds.")
                continue
            
            # correct cell cell id of switch
            print(f"Match found for {cell}, changing {candidate_id} from frame {last_frame + 1}")
            selection = (dataframe["cell_id"] == candidate_id) & (dataframe["frame_index"] >= last_frame + 1)
            current_parent_id = dataframe[dataframe['cell_id'] == cell]['parent_id'].iloc[0]
            dataframe.loc[selection, 'parent_id'] = current_parent_id
            dataframe.loc[selection, 'cell_id'] = cell
            
            track_new = re.match(r'^(.*)_[0-9]+$', cell).group(1)  #new track_id
            dataframe.loc[selection, 'track_id'] = track_new
            
            edit_counts = edit_counts + 1
        
        
        # to inform if there is more than two candidate cells. Currently nothing will be done.
        else:
            print(f"There is more than two candidates for {cell} in frame {last_frame}")

    # report number of edits
    # print(f"A total of {edit_counts} tracks were corrected")
        
    return dataframe, edit_counts


def refine_tracking(dataframe, cycle = 5):
    """
    Iterative function to run the refine_cell_track function for a number of cycles.
    Most of the tracking errors are corrected in the first pass but the subsequent
    passes are just to make sure that the remaining ones are corrected. Reduce the 
    number of cycles if the refining step takes too long. Will stop cycle if number
    of edits 0.
    """
    tempdf = dataframe
    
    
    for cycle in range(cycle):
        edits = 0
        # print(f"Refining track cycle number {cycle + 1}")
        tempdf, edits = refine_cell_track(tempdf)
        
        # print(f"Total edits is {edits} in cycle {cycle}")
        
        # check if there are no longer edits and exit refining cycle
        if edits == 0:
            break
    
    # sort dataframe 
    tempdf = tempdf.sort_values(by=["cell_id","frame_index"])
    
    return tempdf

def convert_df_mask_points(dataframe):
    """
    Function to convert mask points into a list of points.
    """
    
    dataframe["mask_of_x_points"] = convert_to_list(dataframe["mask_of_x_points"].values.tolist())
    dataframe["mask_of_y_points"] = convert_to_list(dataframe["mask_of_y_points"].values.tolist())
    
    return dataframe


def draw_bounding_box(image, x_points, y_points, cell_id):
    """
    Draws a bounding box with the object ID on the image. Takes the x_points and
    y_points from refined tracking results dataframe.  
    """
    
    min_x, max_x = np.min(x_points), np.max(x_points)
    min_y, max_y = np.min(y_points), np.max(y_points)

    # Draw the bounding box
    cv2.rectangle(image, (min_x, min_y), (max_x, max_y), (0, 0, 0), 2)
    
    # Put the text on the image
    cv2.putText(image, cell_id, (max_x, max_y), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 255), 1)
    return image

def generate_track_visualisation_from_df(dataframe, image):
    image_len, x_size, y_size = np.shape(image)
    frames = dataframe.frame_index.unique()
    
    if len(frames) != image_len:
        raise ValueError("Image sequence does not correspond to the number of tracked frames!") 
        
    imgs_with_bbox = np.empty((image_len, x_size, y_size), dtype = np.uint8)
    
    # generate new mask without confounding segmentations
    for i in tqdm(range(len(frames))):
        df_frame = dataframe[dataframe["frame_index"] == i]
        
        temp_img = image[i]
        low, hi = np.percentile(temp_img, (0.1, 99.9))
        temp_img = img_as_ubyte(exposure.rescale_intensity(temp_img, in_range=(low, hi)))
        
        for index, row in df_frame.iterrows():
            x_points = row["mask_of_x_points"]
            y_points = row["mask_of_y_points"]
            cell_type = row["cell_type"]
            cell_id = row["cell_id"]
            
            if cell_type != cell_type: # to catch nan values
                cell_info = cell_id
                
            elif cell_type == None:
                cell_info = cell_id
            
            else:
                cell_info = f"{cell_id}_{cell_type}"
            
            draw_bounding_box(temp_img, x_points, y_points, cell_info)
            
        imgs_with_bbox[i] = temp_img
    
    return imgs_with_bbox
