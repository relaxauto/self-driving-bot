#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 23:10:15 2020

@author: JoseCLee
"""
import numpy as np
import cv2

def detect_edges(image, low_threshold=200, high_threshold=400):
  return cv2.Canny(image, low_threshold, high_threshold)

def region_of_interest(edges):
  height, width = edges.shape
  mask = np.zeros_like(edges)

  # only focus bottom half of the screen
  polygon = np.array([[
      (0, height * 1 / 2),
      (width, height * 1 / 2),
      (width, height),
      (0, height),
  ]], np.int32)

  cv2.fillPoly(mask, polygon, 255)
  cropped_edges = cv2.bitwise_and(edges, mask)
  return cropped_edges

def mask_lane(image,low_threshold=200, high_threshold=400):
  '''
  detect region of interest
  '''
  image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  edges = detect_edges(image,low_threshold,high_threshold)
  roi = region_of_interest(edges)

  return roi

def detect_line_segments(cropped_edges):
  # tuning min_threshold, minLineLength, maxLineGap is a trial and error process by hand
  rho = 1  # distance precision in pixel, i.e. 1 pixel
  angle = np.pi / 180  # angular precision in radian, i.e. 1 degree
  min_threshold = 10  # minimal of votes
  line_segments = cv2.HoughLinesP(cropped_edges, rho, angle, min_threshold, 
                                  np.array([]), minLineLength=8, maxLineGap=4)

  return line_segments

def make_points(frame, line):
  height, width, _ = frame.shape
  slope, intercept = line
  y1 = height  # bottom of the frame
  y2 = int(y1 * 1 / 2)  # make points from middle of the frame down

  # bound the coordinates within the frame
  x1 = max(-width, min(2 * width, int((y1 - intercept) / slope)))
  x2 = max(-width, min(2 * width, int((y2 - intercept) / slope)))
  return [[x1, y1, x2, y2]]

def average_slope_intercept(frame, line_segments):
  """
  This function combines line segments into one or two lane lines
  If all line slopes are < 0: then we only have detected left lane
  If all line slopes are > 0: then we only have detected right lane
  """
  lane_lines = []
  if line_segments is None:
      #logging.info('No line_segment segments detected')
      pass
      return lane_lines

  height, width, _ = frame.shape
  left_fit = []
  right_fit = []

  boundary = 1/3
  left_region_boundary = width * (1 - boundary)  # left lane line segment should be on left 2/3 of the screen
  right_region_boundary = width * boundary # right lane line segment should be on left 2/3 of the screen

  for line_segment in line_segments:
      for x1, y1, x2, y2 in line_segment:
          if x1 == x2:
              #logging.info('skipping vertical line segment (slope=inf): %s' % line_segment)
              continue
          fit = np.polyfit((x1, x2), (y1, y2), 1)
          slope = fit[0]
          intercept = fit[1]
          if slope < 0:
              if x1 < left_region_boundary and x2 < left_region_boundary:
                  left_fit.append((slope, intercept))
          else:
              if x1 > right_region_boundary and x2 > right_region_boundary:
                  right_fit.append((slope, intercept))

  left_fit_average = np.average(left_fit, axis=0)
  if len(left_fit) > 0:
      lane_lines.append(make_points(frame, left_fit_average))

  right_fit_average = np.average(right_fit, axis=0)
  if len(right_fit) > 0:
      lane_lines.append(make_points(frame, right_fit_average))

  #logging.debug('lane lines: %s' % lane_lines)  # [[[316, 720, 484, 432]], [[1009, 720, 718, 432]]]

  return lane_lines

def detect_lane(frame):
  '''
  Detect coordinates of lanes from both side (or 1 side)
  Could go wrong if distracted by background
  '''
  edges = detect_edges(frame)
  cropped_edges = region_of_interest(edges)
  line_segments = detect_line_segments(cropped_edges)
  lane_lines = average_slope_intercept(frame, line_segments)

  return lane_lines

def display_lines(frame, line_color=(0, 255, 0), line_width=2):
  '''
  Draw line into original picture
  '''
  lines = detect_lane(frame)
  line_image = np.zeros_like(frame)
  if lines is not None:
      for line in lines:
          for x1, y1, x2, y2 in line:
            cv2.line(line_image, (x1, y1), (x2, y2), line_color, line_width)
  line_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
  return line_image

def display_lines_w_headlines(frame, line_color=(0, 255, 0), line_width=2):
  '''
  Draw additional direction line (red) into original picture
  '''
  height, width, _ = frame.shape
  lines = detect_lane(frame)
  line_image = np.zeros_like(frame)
  if lines is not None:
      for line in lines:
          for x1, y1, x2, y2 in line:
            cv2.line(line_image, (x1, y1), (x2, y2), line_color, line_width)
  
  # headline direction
  if len(lines) == 1: ## 1 lane
    x1, _, x2, _ = lines[0][0]
    x_offset = x2 - x1
    y_offset = int(height / 2)
  else:
    _, _, left_x2, _ = lines[0][0]
    _, _, right_x2, _ = lines[1][0]
    #mid = int(width / 2)
    x_offset = (left_x2 + right_x2) / 2 #- mid
    y_offset = int(height / 2)
  
  # headline starting point
  x1 = int(width / 2)
  y1 = height

  cv2.line(line_image, (x1, y1), (int(x_offset), y_offset), (0, 0, 255), 5)
  #line_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
  line_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)

  return line_image