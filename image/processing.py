import cv2
import numpy as np
from collections import deque
from .warning import *

import threading
import os
import csv
import time

def preprocessing(frame, mean, std):
    # normalize and quantize input
    # with paramaeters obtained during
    # model calibration
    frame *= (1 / 255)
    expd = np.expand_dims(frame, axis=0)
    quantized = (expd / std + mean)

    return quantized.astype(np.uint8)


def postprocessing(pred_obj, frame, mean, std, in_shape, out_shape, camera_offset_bias):
    # get predicted mask in shape (n_rows*n_cols, )
    # and reshape back to (n_rows, n_cols)
    # pred = pred_obj[1].reshape(in_shape)
    pred = np.squeeze(pred_obj, axis=(0, 3))

    # dequantize and cast back to float
    dequantized = (std * (pred - mean))
    dequantized = dequantized.astype(np.float32)

    # resize frame and mask to output shape
    frame = cv2.resize(frame, out_shape)
    mask = cv2.resize(dequantized, (frame.shape[1], frame.shape[0]))

    # perform closing operation on mask to smooth out lane edges
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_ERODE, kernel, iterations=4)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)

    # overlay frame and segmentation mask
    # frame[mask != 0] = (255, 0, 255)

    frame = process_lane_dpt(frame, mask, camera_offset_bias)

    return frame

def flood_fill_get_coords(start_x, start_y, mask, visited_mask, min_y_threshold):
    height, width = mask.shape[:2]
    coords = [] # List to store coordinates of the filled area

    # Check if the starting point itself is valid and eligible
    if not (0 <= start_x < width and 0 <= start_y < height and \
            start_y > min_y_threshold and \
            mask[start_y, start_x] and \
            not visited_mask[start_y, start_x]): # Check visited mask
        # Invalid start point: out of bounds, below threshold, not originally green, or already visited
        return coords

    q = deque([(start_x, start_y)]) # Initialize queue
    visited_mask[start_y, start_x] = True # Mark start point as visited
    coords.append((start_x, start_y)) # Add start point to coordinates list

    # Define neighbors for Manhattan distance <= 2
    neighbors_offsets = [
        # Distance 1
        (0, 1), (0, -1), (1, 0), (-1, 0),
        # Distance 2 (Straight)
        # (0, 2), (0, -2), (2, 0), (-2, 0),
        # # Distance 2 (Diagonal - these are Manhattan distance 2)
        # (1, 1), (1, -1), (-1, 1), (-1, -1),
    ]

    while q:
        px, py = q.popleft() # Get the next pixel to process

        # Check neighbors within Manhattan distance <= 2
        for dx, dy in neighbors_offsets:
            nx, ny = px + dx, py + dy

            # Check if the neighbor is within image bounds
            if 0 <= nx < width and 0 <= ny < height:
                # Check eligibility: y > threshold, in mask, AND not already visited
                if ny > min_y_threshold and \
                        mask[ny, nx] and \
                        not visited_mask[ny, nx]:
                    visited_mask[ny, nx] = True # Mark neighbor as visited
                    coords.append((nx, ny)) # Add neighbor coordinates
                    q.append((nx, ny)) # Add the neighbor to the queue for processing
    return coords

# --- START: Helper function to find min/max y points ---
def find_min_max_y_points(coords):
    if not coords:
        return None, None

    min_y = min(y for x, y in coords)
    max_y = max(y for x, y in coords)

    min_y_points_x = [x for x, y in coords if y == min_y]
    max_y_points_x = [x for x, y in coords if y == max_y]

    # Calculate average x, handle case where list might be empty (though unlikely if min/max y found)
    avg_x_at_min_y = int(np.mean(min_y_points_x)) if min_y_points_x else -1
    avg_x_at_max_y = int(np.mean(max_y_points_x)) if max_y_points_x else -1

    # Create points only if average x was valid
    min_point = (avg_x_at_min_y, min_y) if avg_x_at_min_y != -1 else None
    max_point = (avg_x_at_max_y, max_y) if avg_x_at_max_y != -1 else None

    return min_point, max_point

def calculate_line_fit(point1, point2):
    if not point1 or not point2:
        return None

    x1, y1 = point1
    x2, y2 = point2

    # Check for identical points or vertical line
    if x1 == x2:
        return None # Vertical line, slope is undefined

    # Calculate slope (m)
    m = (y2 - y1) / (x2 - x1)
    # Calculate y-intercept (b) using point1
    b = y1 - m * x1

    return (m, b)

def calculate_x_at_y(line_fit, y_target):
    if not line_fit:
        return None

    m, b = line_fit

    # Avoid division by zero for horizontal or near-horizontal lines
    if abs(m) < 1e-6:
        return None

    x_at_target_y = (y_target - b) / m
    return x_at_target_y

lane_departure_threshold = 3
lane_departure_counter = 0
def process_lane_dpt(img_rs, mask, camera_offset_bias):
    global lane_departure_counter

    left_angle = None
    right_angle = None
    angle_diff = None
    distance = None
    distance_ratio = None
    adaptive_threshold = None
    is_departure_condition = 0

    # --- START: Hough Transform to find and unmark HORIZONTAL lines from the mask ---
    # Prep mask for edge detection (use the raw ll_predict for morphology)
    height, width = img_rs.shape[:2]

    kernel_small = np.ones((3, 3), np.uint8)
    kernel = np.ones((5, 5), np.uint8)
    LL_mask_processed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_small)
    LL_mask_processed = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)
    LL_mask_processed = (mask * 255).astype(np.uint8)
    edges = cv2.Canny(LL_mask_processed, 30, 150)

    # Detect lines using Hough Transform
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=20, minLineLength=10, maxLineGap=20)
    # Create a temporary mask to mark pixels belonging to horizontal lines
    horizontal_lines_mask = np.zeros_like(mask, dtype=bool)  # Renamed variable
    horizontal_slope_threshold = 0.15  # Define flatness threshold
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            is_horizontal = False
            # Check for perfectly horizontal lines first
            if y1 == y2:
                is_horizontal = True
            # Check for near-horizontal lines (avoid division by zero)
            elif x2 != x1:
                slope = abs((y2 - y1) / (x2 - x1))  # Absolute slope
                if slope < horizontal_slope_threshold:
                    is_horizontal = True
            # Note: Perfectly vertical lines (x1==x2) will not be caught here
            if is_horizontal:
                # Draw this horizontal line segment onto a temporary uint8 mask
                temp_draw_mask = np.zeros(mask.shape, dtype=np.uint8)
                # Use thickness 20 as specified by user
                cv2.line(temp_draw_mask, (x1, y1), (x2, y2), 255, 17)
                # Update the boolean horizontal_lines_mask where the line was drawn
                horizontal_lines_mask[temp_draw_mask > 0] = True
    # Unmark the main mask where horizontal lines were detected
    # This sets pixels that are True in BOTH masks to False in the main mask
    mask[horizontal_lines_mask & (mask != 0)] = 0
    # mask_u8 = (mask * 255).astype(np.uint8)
    # return mask_u8
    # --- END: Hough Transform to find and unmark HORIZONTAL lines ---

    # Create a clean copy of the original image for lane line visualization
    lane_visualization = img_rs.copy()
    lane_visualization[mask != 0] = [0, 255, 0]  # Lane lines in green

    y_range_start = 220  # Start y for searching seed points
    y_range_end = 300 # End y for searching seed points
    min_y_threshold = 180
    height, width = lane_visualization.shape[:2]
    center_x = width // 2
    yellow_color = (0, 255, 255)
    blue_color = (255, 0, 0)
    red_color = (0, 69, 255)
    cyan_color = (255, 255, 0)
    
    # Draw a vertical line down the center of the image for reference
    cv2.line(lane_visualization, (center_x, 0), (center_x, height), yellow_color, 1)
    # Add cyan dots to mark the y-range search area on the center line
    if 0 <= y_range_start < height:
        cv2.circle(lane_visualization, (center_x, y_range_start), 2, cyan_color, -1)  # Radius 4, filled
    if 0 <= y_range_end < height:
        cv2.circle(lane_visualization, (center_x, y_range_end), 2, cyan_color, -1)  # Radius 4, filled

    visited = np.zeros_like(mask, dtype=bool)

    x_left, y_left = -1, -1
    x_right, y_right = -1, -1

    # Find all eligible pixels in the mask within the y-range
    candidate_y_coords, candidate_x_coords = np.where(mask)
    valid_indices = (candidate_y_coords >= y_range_start) & \
                    (candidate_y_coords <= y_range_end)
    eligible_pixels = list(zip(candidate_x_coords[valid_indices], candidate_y_coords[valid_indices]))

    detected_lines = []

    MIN_LANE_LINE_LENGTH_Y = 15
    for x, y in eligible_pixels:
        # Check if pixel has already been visited by a previous flood fill
        if not visited[y, x]:
            # Start flood fill from this eligible, unvisited point
            # Note: flood_fill_get_coords already checks mask[y,x] and min_y_threshold
            coords = flood_fill_get_coords(x, y, mask, visited, min_y_threshold)

            if coords:  # If flood fill found a connected area
                # Analyze the found segment
                min_p, max_p = find_min_max_y_points(coords)

                if coords:
                    min_p, max_p = find_min_max_y_points(coords)

                    if min_p and max_p:
                        # --- MODIFICATION START: Check line length ---
                        line_length_y = max_p[1] - min_p[1]  # Calculate vertical length
                        if line_length_y >= MIN_LANE_LINE_LENGTH_Y:
                            # Calculate fit
                            line_fit = calculate_line_fit(min_p, max_p)
                            x_at_bottom = calculate_x_at_y(line_fit, height)
                            if x_at_bottom is not None:
                                detected_lines.append({
                                    'coords': coords,
                                    'min_point': min_p,
                                    'max_point': max_p,
                                    'line_fit': line_fit,
                                    'x_at_bottom': x_at_bottom
                                })
                        else:
                            log = 123
                            # logger.debug(
                            #     f"Discarded short line segment: length={line_length_y} < {MIN_LANE_LINE_LENGTH_Y}")
                            # print("logger")
                    else:
                        # logger.warning("Could not find min/max points for a detected segment.")
                        # print("logger")
                        log=123

    # Perform flood fill starting from the found intersection points
    # Fill the connected area containing the left point, if found

    left_fit = None
    right_fit = None
    best_left_line = None
    best_right_line = None
    min_left_dist = float('inf')
    min_right_dist = float('inf')

    for line in detected_lines:
        x_at_bottom = line['x_at_bottom']  # Use the calculated average x at the lowest y
        dist_to_center = abs(x_at_bottom - center_x)

        # Check for left line candidate
        if x_at_bottom <= center_x:
            if dist_to_center < min_left_dist:
                min_left_dist = dist_to_center
                best_left_line = line
        # Check for right line candidate
        else:
            if dist_to_center < min_right_dist:
                min_right_dist = dist_to_center
                best_right_line = line

    # Assign the line fits and draw the selected lines
    if best_left_line and best_left_line['line_fit']:
        left_fit = best_left_line['line_fit']
        min_p_left = best_left_line['min_point']
        max_p_left = best_left_line['max_point']
        # Draw the selected left line
        cv2.line(lane_visualization, min_p_left, max_p_left, blue_color, 4)


    if best_right_line and best_right_line['line_fit']:
        right_fit = best_right_line['line_fit']
        min_p_right = best_right_line['min_point']
        max_p_right = best_right_line['max_point']
        # Draw the selected right line
        cv2.line(lane_visualization, min_p_right, max_p_right, red_color, 4)

    # Calculate vanishing point
    def calculate_vanishing_point(left_fit, right_fit):
        if not left_fit or not right_fit:
            return None
        m1, b1 = left_fit
        m2, b2 = right_fit
        if m1 != m2:  # Ensure lines aren't parallel
            x_vp = (b2 - b1) / (m1 - m2)
            y_vp = m1 * x_vp + b1
            return (int(x_vp), int(y_vp))
        return None

    def is_lane_departure_single_lane(slope, img_width):
        """
        Determine if a single lane's slope indicates lane departure

        Args:
            slope: Slope of the detected lane line
            img_width: Width of the image

        Returns:
            bool: True if the slope indicates lane departure
        """
        # Convert slope to angle in degrees
        angle = np.abs(np.degrees(np.arctan(slope)))

        # Define baseline thresholds for single lane detection
        # These thresholds can be tuned based on your specific road conditions
        min_departure_angle = 60.0 + camera_offset_bias

        # If the angle is very steep, it's likely a departure
        return angle > min_departure_angle
    # Get vanishing point
    vp = calculate_vanishing_point(left_fit, right_fit)

    if vp:
        log=0
        x_vp, y_vp = vp
        # Draw vanishing point visualizations
        if 0 <= x_vp < lane_visualization.shape[1] and 0 <= y_vp < lane_visualization.shape[0]:
            cv2.circle(lane_visualization, (x_vp, y_vp), 8, (255, 0, 255), -1)  # Purple fill

            # # Draw horizontal reference line
            h_line_y = int(lane_visualization.shape[0] * 0.8)
            # cv2.line(lane_visualization, (0, h_line_y),
            #          (lane_visualization.shape[1], h_line_y), (255, 100, 0), 2)

            # # Draw horizontal horizon line across the entire image at y-coordinate of vanishing point
            # cv2.line(lane_visualization, (0, y_vp), (lane_visualization.shape[1], y_vp), (255, 0, 255), 2)  # Magenta line

            # Calculate distance from vanishing point to the center of the horizon line
            img_center_x = lane_visualization.shape[1] // 2
            distance = abs(x_vp - img_center_x)

            left_angle = None
            right_angle = None
            angle_diff = None

            if left_fit:
                m, b = left_fit
                left_intersection_x = int((h_line_y - b) / m)
                cv2.circle(lane_visualization, (left_intersection_x, h_line_y), 5, (0, 255, 255), -1)
                left_angle_rad = np.arctan(m)
                left_angle = np.abs(np.degrees(left_angle_rad))
                angle_text_left = f"Left angle: {left_angle:.1f} deg"

                # Add text with background for visibility
                text_size = cv2.getTextSize(angle_text_left, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                cv2.rectangle(lane_visualization, (10, 30 - 25),
                              (10 + text_size[0], 30 + 5), (0, 0, 0), -1)
                cv2.putText(lane_visualization, angle_text_left, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            if right_fit:
                m, b = right_fit
                right_intersection_x = int((h_line_y - b) / m)
                cv2.circle(lane_visualization, (right_intersection_x, h_line_y), 5, (0, 255, 255), -1)
                right_angle_rad = np.arctan(m)
                right_angle = np.abs(np.degrees(right_angle_rad))
                angle_text_right = f"Right angle: {right_angle:.1f} deg"

                # Add text with background for visibility
                text_size = cv2.getTextSize(angle_text_right, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                cv2.rectangle(lane_visualization, (10, 70 - 25),
                              (10 + text_size[0], 70 + 5), (0, 0, 0), -1)
                cv2.putText(lane_visualization, angle_text_right, (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # Calculate angle difference if both angles are available
            if left_angle is not None and right_angle is not None:
                angle_diff = np.abs(left_angle - right_angle - abs(camera_offset_bias))

            # NEW CODE: Calculate the refined adaptive threshold
            # -----------------------------------------------------
            def calculate_adaptive_threshold(distance, img_width, angle_diff=None):
                """
                Calculate an adaptive threshold that's more sensitive in the
                critical 20-25 degree range where departures often occur
                """
                # Base values
                base_threshold = 10
                distance_ratio = distance / img_width

                # If we have the current angle difference, we can use it for fine-tuning
                angle_factor = 1.0
                if angle_diff is not None:
                    # Make threshold more sensitive when angle_diff is in the 18-25 degree range
                    if 18 <= angle_diff <= 25:
                        # Reduce the threshold when we're in this critical range
                        angle_factor = 0.85
                    elif angle_diff > 25:
                        # For very large angles, maintain standard threshold
                        angle_factor = 1.0
                    else:
                        # For small angles, increase threshold slightly to avoid false positives
                        angle_factor = 1.1

                # Modified logistic function with reduced effect for moderate distances
                max_increase = 45  # Reduced from 60 to lower the overall threshold
                midpoint = 0.18    # Increased from 0.15 to delay threshold increase
                steepness = 10     # Reduced from 12 to make the transition more gradual

                # Calculate the base adaptive threshold
                angle_increase = max_increase / (1 + np.exp(-steepness * (distance_ratio - midpoint)))

                # Apply a scaling factor for small distances to reduce threshold in critical range
                small_distance_factor = 1.0
                if distance_ratio < 0.1:  # For distances less than 10% of image width
                    small_distance_factor = 0.9  # Reduce threshold by 10%

                # Calculate final threshold with all factors
                adaptive_threshold = (base_threshold + angle_increase) * angle_factor * small_distance_factor

                return adaptive_threshold
            # -----------------------------------------------------

            # Calculate adaptive threshold with refined formula
            adaptive_threshold = calculate_adaptive_threshold(
                distance, lane_visualization.shape[1], angle_diff)

            # # Display the distance information
            # distance_text = f"Distance: {distance} px"
            # text_size = cv2.getTextSize(distance_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            # cv2.rectangle(lane_visualization, (img_center_x - text_size[0] // 2, y_vp - 30 - 5),
            #               (img_center_x + text_size[0] // 2, y_vp - 30 + 20), (0, 0, 0), -1)
            # cv2.putText(lane_visualization, distance_text, (img_center_x - text_size[0] // 2, y_vp - 15),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

            # # Display the adaptive threshold
            # threshold_text = f"Angle threshold: {adaptive_threshold:.1f} deg"
            # text_size = cv2.getTextSize(threshold_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            # cv2.rectangle(lane_visualization, (img_center_x - text_size[0] // 2, y_vp - 60 - 5),
            #               (img_center_x + text_size[0] // 2, y_vp - 60 + 20), (0, 0, 0), -1)
            # cv2.putText(lane_visualization, threshold_text, (img_center_x - text_size[0] // 2, y_vp - 45),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

            # Inside the frame processing logic where left_angle and right_angle are available
            if angle_diff is not None:
                # Use the adaptive threshold
                departure_angle_diff = adaptive_threshold

                # # Visualize the angle difference compared to threshold
                # comparison_text = f"Angle diff: {angle_diff:.1f} | Threshold: {departure_angle_diff:.1f}"
                # text_size = cv2.getTextSize(comparison_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                # cv2.rectangle(lane_visualization, (10, 140 - 25),
                #               (10 + text_size[0], 140 + 5), (0, 0, 0), -1)
                #
                # # Change text color based on whether we're above threshold
                # text_color = (0, 0, 255) if angle_diff > departure_angle_diff else (0, 255, 0)
                # cv2.putText(lane_visualization, comparison_text, (10, 140),
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)

                # Now compare with the adaptive threshold
                if angle_diff > departure_angle_diff and distance < lane_visualization.shape[1]*0.055:
                    lane_departure_counter += 1
                else:
                    decay_rate = min(4.0, 1 + (lane_departure_counter * 0.3))
                    lane_departure_counter = max(lane_departure_counter - decay_rate, 0)

                # print(f"Angle diff: {angle_diff:.1f} | Threshold: {departure_angle_diff:.1f}")
                if angle_diff > departure_angle_diff:
                    # print("Above threshold")
                    log=123

                if lane_departure_counter >= lane_departure_threshold:
                    if right_angle > left_angle:
                        side = 'r'
                        threading.Thread(
                            target=departure_warning,
                            args=(side,),
                            daemon=True
                        ).start()
                        cv2.putText(lane_visualization, "Right Lane Departure Detected", (10, height - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    else:
                        side = 'l'
                        threading.Thread(
                            target=departure_warning,
                            args=(side,),
                            daemon=True
                        ).start()
                        cv2.putText(lane_visualization, "Left Lane Departure Detected", (10, height - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    
        # left_angle = left_angle_value if 'left_angle_value' in locals() else None
        # right_angle = right_angle_value if 'right_angle_value' in locals() else None
        
        # distance = distance
        # distance_ratio = distance / lane_visualization.shape[1]
        # adaptive_threshold = adaptive_threshold
        # angle_diff = angle_diff
        # if angle_diff is not None and distance is not None:
        #     if angle_diff > adaptive_threshold and distance < lane_visualization.shape[1] * 0.055:
        #         is_departure_condition = 1
            
            # Draw vertical line from vanishing point to horizontal line
            # cv2.line(lane_visualization, (x_vp, y_vp), (x_vp, h_line_y), (0, 255, 255), 2)
            # cv2.circle(lane_visualization, (x_vp, h_line_y), 5, (255, 0, 0), -1)

    else:
        # left_angle = left_angle_value if 'left_angle_value' in locals() else None
        # right_angle = right_angle_value if 'right_angle_value' in locals() else None
        if left_fit:
            m, b = left_fit
            left_angle_rad = np.arctan(m)
            left_angle = np.abs(np.degrees(left_angle_rad))
            angle_text_left = f"Left angle: {left_angle:.1f} deg"

            # Add text with background for visibility
            text_size = cv2.getTextSize(angle_text_left, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            cv2.rectangle(lane_visualization, (10, 30 - 25),
                          (10 + text_size[0], 30 + 5), (0, 0, 0), -1)
            cv2.putText(lane_visualization, angle_text_left, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # Detect lane departure from left line slope
            is_departure = is_lane_departure_single_lane(m, lane_visualization.shape[1])
            if is_lane_departure_single_lane(m, lane_visualization.shape[1]):
                is_departure_condition = 1

            # Add text with single lane departure threshold
            departure_text = f"Left Lane: {'Departure' if is_departure else 'Normal'}"
            text_size = cv2.getTextSize(departure_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            text_color = (0, 0, 255) if is_departure else (0, 255, 0)
            cv2.rectangle(lane_visualization, (10, 110 - 25),
                          (10 + text_size[0], 110 + 5), (0, 0, 0), -1)
            cv2.putText(lane_visualization, departure_text, (10, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)

            if is_departure:
                lane_departure_counter += 1
            else:
                # Left detected, but not steep enough -> Decay
                decay_rate = min(4.0, 1 + (lane_departure_counter * 0.3))
                lane_departure_counter = max(lane_departure_counter - decay_rate, 0)

            if lane_departure_counter >= lane_departure_threshold:
                side = 'l'
                threading.Thread(
                    target=departure_warning,
                    args=(side,),
                    daemon=True
                ).start()
                cv2.putText(lane_visualization, "Left Lane Departure Detected", (width - 400, height - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        if right_fit:
            m, b = right_fit
            right_angle_rad = np.arctan(m)
            right_angle = np.abs(np.degrees(right_angle_rad))
            angle_text_right = f"Right angle: {right_angle:.1f} deg"

            # Add text with background for visibility
            text_size = cv2.getTextSize(angle_text_right, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            cv2.rectangle(lane_visualization, (10, 70 - 25),
                          (10 + text_size[0], 70 + 5), (0, 0, 0), -1)
            cv2.putText(lane_visualization, angle_text_right, (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # Detect lane departure from right line slope
            is_departure = is_lane_departure_single_lane(m, lane_visualization.shape[1])
            if is_lane_departure_single_lane(m, lane_visualization.shape[1]):
                is_departure_condition = 1

            # Add text with single lane departure threshold
            departure_text = f"Right Lane: {'Departure' if is_departure else 'Normal'}"
            text_size = cv2.getTextSize(departure_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            text_color = (0, 0, 255) if is_departure else (0, 255, 0)
            cv2.rectangle(lane_visualization, (10, 150 - 25),
                          (10 + text_size[0], 150 + 5), (0, 0, 0), -1)
            cv2.putText(lane_visualization, departure_text, (10, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)

            if is_departure:
                lane_departure_counter += 1
            else:
                # Right detected, but not steep enough -> Decay
                decay_rate = min(4.0, 1 + (lane_departure_counter * 0.3))
                lane_departure_counter = max(lane_departure_counter - decay_rate, 0)

            if lane_departure_counter >= lane_departure_threshold:
                # print("departure warning")
                side = 'r'
                threading.Thread(
                    target=departure_warning,
                    args=(side,),
                    daemon=True
                ).start()
                cv2.putText(lane_visualization, "Right Lane Departure Detected", (width - 400, height - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Write to CSV at the end of processing
    csv_filename = "image/record.csv"
    try:
        file_exists = os.path.isfile(csv_filename)
        with open(csv_filename, 'a', newline='') as csvfile:
            fieldnames = [
                'timestamp',
                'left_angle',
                'right_angle',
                'camera_offset',
                'angle_diff',
                'distance',
                'distance_ratio',
                'img_width',
                'adaptive_threshold',
                'is_departure'
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            if not file_exists:
                writer.writeheader()
            
            writer.writerow({
                'timestamp': time.time(),
                'left_angle': left_angle,
                'right_angle': right_angle,
                'camera_offset': camera_offset_bias,
                'angle_diff': angle_diff,
                'distance': distance,
                'distance_ratio': distance_ratio,
                'img_width': lane_visualization.shape[1],
                'adaptive_threshold': adaptive_threshold,
                'is_departure': is_departure_condition
            })
    except Exception as e:
        print(f"CSV write error: {str(e)}")

    return lane_visualization