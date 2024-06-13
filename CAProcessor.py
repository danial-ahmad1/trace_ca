# Not all of these will be used, remove as needed
import os
import matplotlib.pyplot as plt
from IPython.display import display
from ipywidgets import interact, widgets
import numpy as np
import pandas as pd
from aicsimageio import AICSImage
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from scipy.ndimage import binary_dilation
from scipy.signal import butter, filtfilt
from tqdm import tqdm
from skimage import io, filters, morphology
from skimage.util import img_as_ubyte
from skimage.filters import rank
from skimage.color import rgb2gray
from skimage.morphology import disk
from skimage.io import imsave
import cv2
from skimage import measure, feature, color
from skimage.draw import disk
import heapq

class CAProcessor:
    def __init__(
            self,
            img_path,
            search_mod = 25,
            z_project = 10,
            keyframe = '/Users/moose/Desktop/trace_ca-local/key-frame-ca-norm2.tif',
            savepath = '/Users/moose/Desktop/trace_ca-local/'
            ):
        self.file_name = img_path
        self.search_mod = search_mod
        self.z_project = z_project
        self.keyframe = keyframe
        self.savepath = savepath
        self.flattened_im_data = []
        self.frame_mean = []
        self.frame_min = []
        self.frame_max = []
        self.frame_std = []
        self.frame_95 = []
        self.frame_05 = []
        self.voting_power = []
        self.vote = {}
        self.hist_key = {}
        self.mem_layer = 0
        self.bud_img = None
        self.bud_sub = None
        self.bud_matched = None
        self.xtest = 0
        self.bud_thresh = 0
        self.bud_brightest = None
        self.region_im = None
        self.region_im_filtered = None
        self.area_list = []
        self.intensity_list = []
        self.eccentricity_list = []

    def load_images(self):
        image = AICSImage(self.file_name)
        self.image_data = image.get_image_data("ZYX", S=0, T=0, C=0)

        self.key_img = io.imread(self.keyframe)

    def subtractflatfield(self, input_img):
        grayscale_img = input_img
        poly2d_fcn = lambda xy, a, b, c, d, e, f: a + b*xy[0] + c*xy[1] + d*xy[0]*xy[0] + e*xy[1]*xy[1] + f*xy[0]*xy[1]

        y, x = np.indices(grayscale_img.shape)

        x_co = x.flatten()
        y_co = y.flatten()
        pix_val = grayscale_img.flatten()

        p0 = [1, 1, 1, 1, 1, 1]
        popt, _ = curve_fit(poly2d_fcn, (x_co, y_co), pix_val, p0=p0) 
        flat_field_img = poly2d_fcn((x_co, y_co), *popt).reshape(grayscale_img.shape)
        fit_img = grayscale_img - (flat_field_img)

        return fit_img

    def background_subtract(self, img_dat):
        x1 = np.min(img_dat)
        x2 = []
        for i in range(len(img_dat)):
            x2.append(img_dat[i] - x1)

        x2 = np.maximum(x2, 0)

        return x2

    def means_match(self, input_img, kfimg):
            # kfmod = subtractflatfield(kfimg)
            # kfmod = kfmod-np.min(kfmod)
            kfmod = kfimg
            kfmean = np.mean(kfmod)

            # xmod_loss = []

            best_mean_diff = np.inf
            best_xmod = 0
            mean_diff = 0
            bftest = input_img
            bfmin = np.min(bftest)
                
            for xmod in tqdm(np.linspace(0.01, 10, 500), desc='Means Matching'):
                xmodtest = np.clip(xmod * (bftest), np.min(kfimg), np.max(kfimg))
                mean_xmodtest = np.mean(xmodtest)
                mean_diff = abs(mean_xmodtest - kfmean)

                # xmod_loss = mean_diff # Diagnostic
                    
                if mean_diff < best_mean_diff:
                    best_mean_diff = mean_diff
                    best_xmod = xmod

                if mean_diff < 0.0005:
                    print(f'Image is at an acceptable target, stopping iterations')
                    break

                bfimg = np.clip(best_xmod * (input_img - np.min(input_img)), np.min(kfimg), np.max(kfimg))

            return bfimg, best_xmod
    

    def remove_background(self):
        for i in tqdm(range(len(self.image_data)), desc='Removing background from stack'):
            self.flattened_im_data.append(self.subtractflatfield(self.image_data[i]))    

    def simple_stats(self):
        for i in tqdm(range(len(self.flattened_im_data)), desc='Calculating statistics'):
            self.frame_mean.append(np.mean(self.image_data[i]))
            self.frame_min.append(np.min(self.image_data[i]))
            self.frame_max.append(np.max(self.image_data[i]))
            self.frame_std.append(np.std(self.image_data[i]))
            self.frame_95.append(np.percentile(self.image_data[i], 95))
            self.frame_05.append(np.percentile(self.image_data[i], 5))

    def detect_peaks(self):
        print('Detecting peaks...')
        background_collect = self.background_subtract(self.frame_mean)
        self.peaks2, _ = find_peaks(background_collect, height = 1)

        if len(self.peaks2) == 0:
            max_key = {}
            for i in range(len(background_collect)):
                max_key[i] = background_collect[i]
            true_max = max(max_key.keys())
            self.peaks2 = [true_max]

        if len(self.peaks2) > 1:
            print('Multiple peaks detected, selecting the highest peak')
            max_key = {}
            for i in range(len(background_collect)):
                max_key[i] = background_collect[i]
            true_max = max(max_key.keys())
            self.peaks2 = [true_max]
        
        print(f'Peak detected at frame: {self.peaks2[0]}')

    def ensemble_vote(self):
        print('Finding membrane layer...')
        for i in range(self.peaks2[0]-self.search_mod, self.peaks2[0]):
            if i > 0:
                self.hist_key[i] = self.image_data[i]

        for i in range(self.peaks2[0], self.peaks2[0]+self.search_mod+1):
            if i < len(self.image_data):
                self.hist_key[i] = self.image_data[i]

        hist_stdev = {}
        for i in self.hist_key.keys():
            hist_stdev[i] = np.std(self.hist_key[i])

        hist_laplace = {}
        hist_laplace_focusemeasure = {}
        for i in self.hist_key.keys():
            hist_laplace[i] = cv2.Laplacian(self.hist_key[i], cv2.CV_64F)
            hist_laplace_focusemeasure[i] = np.var(hist_laplace[i])

        hist_tenengrad_focusemeasure = {}
        hist_squared_grad = {}
        for i in self.hist_key.keys():
            sobelx = cv2.Sobel(self.hist_key[i], cv2.CV_64F, 1, 0, ksize=5)
            sobely = cv2.Sobel(self.hist_key[i], cv2.CV_64F, 0, 1, ksize=5)

            magnitude = np.sqrt(sobelx**2 + sobely**2)
            squared_grad = (sobelx**2 + sobely**2)

            hist_tenengrad_focusemeasure[i] = np.var(magnitude)
            hist_squared_grad[i] = np.var(squared_grad)


        hist_brenner_focusemeasure = {}
        for i in self.hist_key.keys():
            shifted_right = np.roll(self.hist_key[i], -1, axis=1)
            shifted_down = np.roll(self.hist_key[i], -1, axis=0)

            diff_right = (shifted_right - self.hist_key[i])[:-1, :-1] ** 2
            diff_down = (shifted_down - self.hist_key[i])[:-1, :-1] ** 2

            sum_diff = np.sum(diff_right) + np.sum(diff_down)
            hist_brenner_focusemeasure[i] = sum_diff

        hist_max = {}
        for i in self.hist_key.keys():
            hist_max[i] = np.max(self.hist_key[i])

        weight_mat = [1, 1, 1, 1, 1, 1]
        # focus_margins = []

        # stdev_two_largest = heapq.nlargest(2, hist_stdev.values())
        # laplace_two_largest = heapq.nlargest(2, hist_laplace_focusemeasure.values())
        # tenengrad_two_largest = heapq.nlargest(2, hist_tenengrad_focusemeasure.values())

        stdev_stdev = np.std(list(hist_stdev.values()))
        laplace_stdev = np.std(list(hist_laplace_focusemeasure.values()))
        tenengrad_stdev = np.std(list(hist_tenengrad_focusemeasure.values()))
        squared_grad_stdev = np.std(list(hist_squared_grad.values()))
        brenner_stdev = np.std(list(hist_brenner_focusemeasure.values()))


        weight_mat[0] = stdev_stdev/np.mean(list(hist_stdev.values()))
        weight_mat[1] = laplace_stdev/np.mean(list(hist_laplace_focusemeasure.values()))
        weight_mat[2] = tenengrad_stdev/np.mean(list(hist_tenengrad_focusemeasure.values()))
        weight_mat[3] = squared_grad_stdev/np.mean(list(hist_squared_grad.values()))
        weight_mat[4] = brenner_stdev/np.mean(list(hist_brenner_focusemeasure.values()))


        focus_ensemble = []
        focus_ensemble.append(max(hist_stdev, key=hist_stdev.get))
        focus_ensemble.append(max(hist_laplace_focusemeasure, key=hist_laplace_focusemeasure.get))
        focus_ensemble.append(max(hist_tenengrad_focusemeasure, key=hist_tenengrad_focusemeasure.get))
        focus_ensemble.append(max(hist_squared_grad, key=hist_squared_grad.get))
        focus_ensemble.append(max(hist_brenner_focusemeasure, key=hist_brenner_focusemeasure.get))

        
        for i in range(len(focus_ensemble)):
            self.voting_power.append((focus_ensemble[i], weight_mat[i]))

        
        for frame_num, weighted_vote in self.voting_power:
            if frame_num in self.vote:
                self.vote[frame_num] += weighted_vote

            else:
                self.vote[frame_num] = weighted_vote


        print(f'Voting Results: {self.voting_power}')
        print(f'Final Vote - Membrane Layer at:  ' + str(max(self.vote, key=self.vote.get)))
        self.mem_layer = max(self.vote, key=self.vote.get)

    def z_composition(self):
        print('Composing Z-projection under membrane layer...')
        bud_test = []
        for i in range(self.z_project):
            bud_test.append(self.mem_layer-i-1)

        bud_test_img = []
        for i in bud_test:
            bud_test_img.append(self.flattened_im_data[i])

        bud_composite = np.max(bud_test_img, axis=0)

        self.bud_img = bud_composite

        self.bud_sub = (self.bud_img - np.min(self.bud_img)) / (np.max(self.bud_img) - np.min(self.bud_img))
        self.bud_sub = img_as_ubyte(self.bud_sub)
        print(f'Z-projection complete using {self.z_project} layers. Layer range: {bud_test}')

    def means_match_z_proc(self):
        print('Matching means of Z-projection to key image...')
        self.bud_matched, self.xtest = self.means_match(self.bud_sub, self.key_img)


    def morpho_proc(self):
        print('Finding budding events...')
        # Image processing for budding events
        # Threshold image
        self.bud_thresh = np.percentile(self.bud_matched, 98)
        self.bud_brightest = np.where(self.bud_matched > self.bud_thresh, 256, 0)

        # plt.figure(dpi=300)
        # plt.imshow(bud_brightest, cmap='gray')
        # plt.axis('off')
        # # plt.savefig('/Users/moose/Desktop/trace_ca-local/' + os.path.splitext(os.path.basename(file_name))[0] + '_binary.tif', dpi=500)
        # plt.show()

        # Morphological analysis
        closed_im = morphology.closing(self.bud_brightest, morphology.square(1))
        label_im = measure.label(closed_im)
        self.region_im = measure.regionprops(label_im, intensity_image=self.bud_matched)
        # for part in region_im:
        #     print('Label: {} Area: {}'.format(part.label, part.area))

    
        for part in self.region_im:
            self.area_list.append(part.area)


        # delete_small_components = filters.threshold_otsu(np.array(area_list)) 
        # area_list = [part for part in area_list if delete_small_components < part < 10000]


        for part in self.region_im:
            self.intensity_list.append(part.mean_intensity)

        int_cut = np.percentile(self.intensity_list, 50)
        self.intensity_list = [part for part in self.intensity_list if  int_cut < part]

        
        for part in self.region_im:
            self.eccentricity_list.append(part.eccentricity)

        self.eccentricity_list = [part for part in self.eccentricity_list if 0.05 < part < 0.99]

        area_list_thresh = np.percentile(self.area_list, 98)
        mean_comp = np.percentile(self.intensity_list, 98)
        std_mean_comp = np.std([part for part in self.intensity_list if part > mean_comp])
        # mean_comp = np.percentile(norm_flat, 99.8)
        lower_ecc = np.percentile(self.eccentricity_list, 2)
        higher_ecc = np.percentile(self.eccentricity_list, 98)

        filter_area_low = area_list_thresh 
        filter_eccentricity_low = lower_ecc
        filter_eccentricity_high = higher_ecc
        # region_im_filtered = [part for part in region_im if part.mean_intensity > mean_comp]
        # region_im_filtered = [part for part in region_im_filtered if filter_area_low < part.area < 10000]
        # region_im_filtered = [part for part in region_im_filtered if filter_eccentricity_low < part.eccentricity < filter_eccentricity_high]

        self.region_im_filtered = [
                                    part for part in self.region_im 
                                    if part.intensity_mean > np.max([mean_comp, 135])
                                    and np.max([filter_area_low, 15]) < part.area < 10000 
                                    and filter_eccentricity_low < part.eccentricity < filter_eccentricity_high
                                    and part.intensity_max > 250
                                    ]

        print('Raw Regions: {}'.format(len(self.region_im)))
        print('Filtered Regions: {}'.format(len(self.region_im_filtered)))

    def results_directory(self):
        if not os.path.exists(self.savepath + os.path.splitext(os.path.basename(self.file_name))[0] + '_results/'):
            os.makedirs(self.savepath + os.path.splitext(os.path.basename(self.file_name))[0] + '_results/')

    def filtered_coordinate_details(self):
        print('Generating coordinates and saving as CSV...')
        for part in self.region_im_filtered:
            print('Centroid: ({:.0f}, {:.0f}) | '
                    'Area: {} | '
                    'Eccentricity {:.2f} | '
                    'Mean Intensity {:.2f} | '
                    'Max Intensity {:.2f}'.format(part.centroid[0], 
                                                part.centroid[1], 
                                                part.area, 
                                                part.eccentricity, 
                                                part.mean_intensity, 
                                                part.intensity_max))
            
        # Save as CSV
        dfCoords = pd.DataFrame(columns=['Centroid X', 'Centroid Y', 'Area', 'Eccentricity', 'Mean Intensity', 'Max Intensity'])
        for part in self.region_im_filtered:
            dfCoords = dfCoords._append({'Centroid X': part.centroid[0], 
                            'Centroid Y': part.centroid[1], 
                            'Area': part.area, 
                            'Eccentricity': part.eccentricity, 
                            'Mean Intensity': part.mean_intensity, 
                            'Max Intensity': part.intensity_max}, ignore_index=True)
            dfCoords = dfCoords.round(3)
        dfCoords.to_csv(self.savepath + os.path.splitext(os.path.basename(self.file_name))[0] + '_results/' + '_detections.csv', index=False)
    
    def compose_figure(self):
        _, ax_alt = plt.subplots(dpi=300)
        ax_alt.set_facecolor('none')
        ax_alt.imshow(self.flattened_im_data[self.mem_layer-2], cmap='gray')
        for region in self.region_im_filtered:
            y, x = region.centroid
            radius = np.sqrt(region.area / np.pi)

            circle = plt.Circle((x, y), np.where(radius*5 < 15, radius*5, 15), fill=False, edgecolor='red')
            ax_alt.add_patch(circle)
        
        ax_alt.invert_yaxis()
        plt.axis('off')
        plt.title(os.path.basename(self.file_name))
        print(f'Saving figure as {self.savepath} + os.path.splitext(os.path.basename(self.file_name))[0] + "_results/" + "_detections.png"')
        plt.savefig(self.savepath + os.path.splitext(os.path.basename(self.file_name))[0] + '_results/' + '_detections.png')
        print('Figure saved')

    # def show_votes(self):
    #     fig, ax = plt.subplots(dpi=300)
    #     ax.plot(list(self.vote.keys()), list(self.vote.values()))
    #     ax.set_xlabel('Frame')
    #     ax.set_ylabel('Vote')
    #     plt.show()

    def run_CAProcessor(self):
        self.load_images()
        self.remove_background()
        self.simple_stats()
        self.detect_peaks()
        self.ensemble_vote()
        self.z_composition()
        self.means_match_z_proc()
        self.morpho_proc()
        self.results_directory()
        self.filtered_coordinate_details()
        self.compose_figure()
        print('Processing complete')