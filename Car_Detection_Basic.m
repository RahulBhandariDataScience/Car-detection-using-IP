% Assignment_291755.m - Finalised Image Processing Script
% Detecting vehicles using:
% 1. Thresholding Detector (Lab 2)
% 2. GMM Foreground Detector (Lab 3)
% 3. YOLOv4 Deep Learning Detector (Lab 4)
% Includes frame-wise visualisation, metric calculation, and side-by-side comparison.

clc; clear; close all;

%% ---------------- Load video and ground truth (Lab 1) ----------------
% The ground truth was prepared and labelled using MATLAB's Video Labeler (Lab 1)
trafficVid = VideoReader('traffic.mj2');
load('export_ground_truth.mat');  % Loads variable: gTruth with labelled 'Car' positions

%% ---------------- Parameters and Setup ----------------
threshold_value = 145;  % Lab 2: manually tuned threshold
SE = strel('square', 5);  % Morphological structuring element

% GMM Background subtraction setup (Lab 3)
foreground_detector = vision.ForegroundDetector('NumGaussians', 5, ...
    'NumTrainingFrames', 50, 'LearningRate', 0.005);

% YOLOv4 detector (Lab 4) using pretrained COCO weights
% Requires: Computer Vision Toolbox Model for YOLO v4
yolo_detector = yolov4ObjectDetector("tiny-yolov4-coco");
vehicle_labels = ["car", "truck", "bus", "motorcycle", "bike"];

% Image layout setup
frameSize = [trafficVid.Height, trafficVid.Width];
padding = 20;

% Metrics storage
P_thr = []; R_thr = []; F1_thr = [];
P_gmm = []; R_gmm = []; F1_gmm = [];
P_yolo = []; R_yolo = []; F1_yolo = [];

frame_no = 1;

%% ---------------- Main Loop for Frame-by-Frame Processing ----------------
while hasFrame(trafficVid)
    img = readFrame(trafficVid);
    traffic_gray = im2gray(img);

    %Ground Truth Extraction (Lab 1)
    gt = gTruth.LabelData.Car{frame_no};
    if isa(gt, "struct"), gt = gt.Position; end
    if isempty(gt), gt = zeros(0, 4); end
    gt_table = table({gt}, 'VariableNames', {'Car'});

    %% ---------- Thresholding Detector (Lab 2) ----------
    % Simple binary thresholding after median filter
    smoothed = medfilt2(traffic_gray, [3 3]);
    T = adaptthresh(smoothed, 0.55);
    binary_img = imbinarize(smoothed, T);
    binary_img = imclose(binary_img, SE);
    binary_img = bwareaopen(binary_img, 500);

    CC = bwconncomp(binary_img);
    props = regionprops(CC, 'BoundingBox');
    threshold_bboxes = reshape([props.BoundingBox], 4, []).';
    pred_thr = table({threshold_bboxes}, 'VariableNames', {'Car'});

    if ~isempty(threshold_bboxes)
        [p, r] = bboxPrecisionRecall(pred_thr, gt_table, 0.5);
    else, p = 0; r = 0;
    end
    f1 = 2*p*r/(p+r+eps);
    P_thr(end+1) = p; R_thr(end+1) = r; F1_thr(end+1) = f1;

    %% ---------- GMM Foreground Detector (Lab 3) ----------
    fgMask = foreground_detector.step(traffic_gray);
    fgMask = imfill(fgMask, 'holes');
    fgMask = imopen(fgMask, SE);
    fgMask = bwareaopen(fgMask, 500);

    CC = bwconncomp(fgMask);
    props = regionprops(CC, 'BoundingBox');
    gmm_bboxes = reshape([props.BoundingBox], 4, []).';
    pred_gmm = table({gmm_bboxes}, 'VariableNames', {'Car'});

    if ~isempty(gmm_bboxes)
        [p, r] = bboxPrecisionRecall(pred_gmm, gt_table, 0.5);
    else, p = 0; r = 0;
    end
    f1 = 2*p*r/(p+r+eps);
    P_gmm(end+1) = p; R_gmm(end+1) = r; F1_gmm(end+1) = f1;

    %% ---------- YOLOv4 Detector (Lab 4) ----------
    [allBoxes, scores, labels] = detect(yolo_detector, img);
    idx = ismember(string(labels), vehicle_labels);
    yolo_bboxes = allBoxes(idx, :);
    yolo_labels = labels(idx);
    pred_yolo = table({yolo_bboxes}, 'VariableNames', {'Car'});

    if ~isempty(yolo_bboxes)
        [p, r] = bboxPrecisionRecall(pred_yolo, gt_table, 0.5);
    else, p = 0; r = 0;
    end
    f1 = 2*p*r/(p+r+eps);
    P_yolo(end+1) = p; R_yolo(end+1) = r; F1_yolo(end+1) = f1;

    %% ---------------- Visualisation ----------------
    vis_thr = insertShape(traffic_gray, 'Rectangle', threshold_bboxes, ...
        'Color', 'green', 'LineWidth', 4);
    vis_thr = insertText(vis_thr, [10, 10], 'Thresholding', ...
        'BoxColor', 'black', 'TextColor', 'white', 'FontSize', 20);

    vis_gmm = insertShape(traffic_gray, 'Rectangle', gmm_bboxes, ...
        'Color', 'blue', 'LineWidth', 4);
    vis_gmm = insertText(vis_gmm, [10, 10], 'GMM', ...
        'BoxColor', 'black', 'TextColor', 'white', 'FontSize', 20);

    vis_yolo = img;
    for i = 1:size(yolo_bboxes,1)
        vis_yolo = insertObjectAnnotation(vis_yolo, 'rectangle', yolo_bboxes(i,:), ...
            string(yolo_labels(i)), 'TextColor', 'red', 'FontSize', 20, 'LineWidth', 4);
    end
    vis_yolo = insertText(vis_yolo, [10, 10], 'YOLOv4', ...
        'BoxColor', 'black', 'TextColor', 'white', 'FontSize', 20);

    spacer = uint8(zeros(frameSize(1), padding, 3));
    combined = [vis_thr, spacer, vis_gmm, spacer, vis_yolo];

    imshow(imresize(combined, 1.2));
    title(sprintf('Frame %d | Threshold | GMM | YOLOv4', frame_no));
    drawnow;

    frame_no = frame_no + 1;
end

%% ---------------- Final Metrics Summary ----------------
fprintf("\n--- FINAL PERFORMANCE METRICS (Mean over all frames) ---\n");
fprintf("Thresholding:  Precision = %.2f | Recall = %.2f | F1 Score = %.2f\n", ...
    mean(P_thr), mean(R_thr), mean(F1_thr));
fprintf("GMM:           Precision = %.2f | Recall = %.2f | F1 Score = %.2f\n", ...
    mean(P_gmm), mean(R_gmm), mean(F1_gmm));
fprintf("YOLOv4:        Precision = %.2f | Recall = %.2f | F1 Score = %.2f\n", ...
    mean(P_yolo), mean(R_yolo), mean(F1_yolo));
