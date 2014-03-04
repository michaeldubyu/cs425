from PIL import Image, ImageDraw
import numpy as np
import csv
import math

def ReadKeys(image):
    """Input an image and its associated SIFT keypoints.

    The argument image is the image file name (without an extension).
    The image is read from the PGM format file image.pgm and the
    keypoints are read from the file image.key.

    ReadKeys returns the following 3 arguments:

    image: the image (in PIL 'RGB' format)

    keypoints: K-by-4 array, in which each row has the 4 values specifying
    a keypoint (row, column, scale, orientation).  The orientation
    is in the range [-PI, PI] radians.

    descriptors: a K-by-128 array, where each row gives a descriptor
    for one of the K keypoints.  The descriptor is a 1D array of 128
    values with unit length.
    """
    im = Image.open(image+'.pgm').convert('RGB')
    keypoints = []
    descriptors = []
    first = True
    with open(image+'.key','rb') as f:
        reader = csv.reader(f, delimiter=' ', quoting=csv.QUOTE_NONNUMERIC,skipinitialspace = True)
        descriptor = []
        for row in reader:
            if len(row) == 2:
                assert first, "Invalid keypoint file header."
                assert row[1] == 128, "Invalid keypoint descriptor length in header (should be 128)."
                count = row[0]
                first = False
            if len(row) == 4:
                keypoints.append(np.array(row))
            if len(row) == 20:
                descriptor += row
            if len(row) == 8:
                descriptor += row
                assert len(descriptor) == 128, "Keypoint descriptor length invalid (should be 128)."
                #normalize the key to unit length
                descriptor = np.array(descriptor)
                descriptor = descriptor / math.sqrt(np.sum(np.power(descriptor,2)))
                descriptors.append(descriptor)
                descriptor = []
    assert len(keypoints) == count, "Incorrect total number of keypoints read."
    print "Number of keypoints read:", int(count)
    return [im,keypoints,descriptors]

def AppendImages(im1, im2):
    """Create a new image that appends two images side-by-side.

    The arguments, im1 and im2, are PIL images of type RGB
    """
    im1cols, im1rows = im1.size
    im2cols, im2rows = im2.size
    im3 = Image.new('RGB', (im1cols+im2cols, max(im1rows,im2rows)))
    im3.paste(im1,(0,0))
    im3.paste(im2,(im1cols,0))
    return im3

def DisplayMatches(im1, im2, matched_pairs):
    """Display matches on a new image with the two input images placed side by side.

    Arguments:
     im1           1st image (in PIL 'RGB' format)
     im2           2nd image (in PIL 'RGB' format)
     matched_pairs list of matching keypoints, im1 to im2

    Displays and returns a newly created image (in PIL 'RGB' format)
    """
    im3 = AppendImages(im1,im2)
    offset = im1.size[0]
    draw = ImageDraw.Draw(im3)
    for match in matched_pairs:
        draw.line((match[0][1], match[0][0], offset+match[1][1], match[1][0]),fill="red",width=2)
    im3.show()
    return im3

def match(image1,image2):
    """Input two images and their associated SIFT keypoints.
    Display lines connecting the first 5 keypoints from each image.
    Note: These 5 are not correct matches, just randomly chosen points.

    The arguments image1 and image2 are file names without file extensions.

    Returns the number of matches displayed.

    Example: match('scene','book')
    """
    im1, keypoints1, descriptors1 = ReadKeys(image1)
    im2, keypoints2, descriptors2 = ReadKeys(image2)
    
    i = 0 # this is our counter into the first image keyframe
    matched_pairs = [] # our match list 

    for d1_row in descriptors1:
        # go through every keypoint descriptor
        best_list = []
        # save all the computed angles of dot prods here
        for d2_row in descriptors2:
            # go through every keypoint in the second image for this current keypoint and
            best_list.append(math.acos(np.dot(d1_row,d2_row)))
            # compute and save the angles of the dot products
        sorted_best_list = sorted(best_list)
        # after we're done going through them all, sort it to retrieve the top 2
        if sorted_best_list[0]/sorted_best_list[1]<0.6 :
            # 0.6 was used for part 3 as a threshold against scene/book
            # check that the ratio of the smallest to the next largest is less than a threshold
            keypoint2_index = best_list.index(sorted_best_list[0])
            # and then retrieve the keypoint
            matched_pairs.append([keypoints1[i],keypoints2[keypoint2_index]])
            # save it into our matched_pairs to use to draw later
        i += 1
        # increment the counter for indexing

    # RANSAC attempt begins here
    RANSAC_matches = []

    for i in range(10):
        RANSAC_subset = []
        # clear our subset

        # do the random selection 10 times
        # pick a random match from matched_pairs
        random_match = matched_pairs[np.random.randint(0,len(matched_pairs),size=1)]

        # calculate the change of orientation for this random match picked
        random_match_coo = abs(np.degrees(random_match[0][3] - random_match[1][3]))
        # calculate the change of scale for this random match picked
        random_match_cos = random_match[0][2]/random_match[1][2]

        # go through all the match pairs
        for all_other_matches in matched_pairs:

            # calculate change of orientation, scale for these match pairs
            all_other_coo = abs(np.degrees(all_other_matches[0][3] - all_other_matches[1][3]))
            all_other_cos = all_other_matches[0][2]/all_other_matches[1][2]

            # if the change of orientation is within +/- a threshold
            # and the change of scale is within 1.5 times and 0.5 times the random_match
            if all_other_coo <= random_match_coo+30 and all_other_coo >= random_match_coo-30:
                if all_other_cos >= 0.5*random_match_cos and all_other_cos <= 1.5*random_match_cos:
                    # add it into the subset
                    RANSAC_subset.append(all_other_matches)

        # if the subset is larger than our current match set
        if len(RANSAC_subset) > len(RANSAC_matches):
            RANSAC_matches = RANSAC_subset
            # update our matches

    # update the match pairs with our largeset match
    matched_pairs = RANSAC_matches
    #
    # END OF SECTION OF CODE TO REPLACE
    #
    im3 = DisplayMatches(im1, im2, matched_pairs)
    return im3

#Test run...
#match('scene','book')
#match('scene','library2')
#match('scene','basmati')
#match('scene','library')
match('library','library2')
#match('library2','library')
