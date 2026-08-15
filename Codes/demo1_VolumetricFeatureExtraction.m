%% =========================================================================
%  Demo_VolumetricFeatureExtraction.m
%  =========================================================================
%  Purpose : End-to-end demo for extracting volumetric features
%            from a paired CT NIfTI image and one or more segmentation masks
%            (also NIfTI format).
%
%  Workflow:
%    1. Point to a folder tree that contains patient sub-folders.
%    2. Each sub-folder must hold:
%         - one CT file  : name contains "CT.nii"
%         - ≥1 mask file : name contains "RTS"  (e.g. RTS_Tumor.nii)
%    3. For every patient × mask pair the script calls
%       Standard_VolFea_Extract() to obtain Base + Advance features.
%    4. All results are collected and saved to a .mat file.
%
%  Dependencies (add to MATLAB path before running):
%    - nii_tool          : read NIfTI files
%    - Standard_VolFea_Extract.m
%    - calculateMaxDiameter.m
%    - Helper functions  : getSUVbox, cc2bw, merge2Structs  (Provided in Wu Lab toolbox)
%
%  Author  : Hui Xu  (hxu12@mdanderson.org)
%  Date    : 2025-10-01  |  Last updated: 2026
% =========================================================================

clc; clear; close all;

%% -------------------------------------------------------------------------
%  0. CONFIGURATION  –  edit these paths before running
% -------------------------------------------------------------------------

% Root folder: each sub-folder = one patient
mainpath = '.\NIFTI';      % <-- change to your data root

% Folder where the toolbox functions live (nii_tool, getSUVbox, etc.)
addpath(genpath('.\Tools'));

% Output folder for the saved feature table
savepath = '\save_path';   % <-- change to your output dir

% Feature category: 'base' | 'advance' | 'all'
featureCategory = 'all';

%% -------------------------------------------------------------------------
%  1. DISCOVER PATIENT FOLDERS
% -------------------------------------------------------------------------

patientFolders = dir(mainpath);
patientFolders = patientFolders([patientFolders.isdir] ...
                   & ~ismember({patientFolders.name}, {'.','..'}));

fprintf('Found %d patient folder(s) under: %s\n\n', ...
        length(patientFolders), mainpath);

%% -------------------------------------------------------------------------
%  2. MAIN LOOP  –  iterate over patients
% -------------------------------------------------------------------------

Pre_Fea_patient_all = cell(length(patientFolders), 1);

for i = 1 : length(patientFolders)

    patientDir  = fullfile(mainpath, patientFolders(i).name);
    subFiles    = dir(patientDir);
    subFiles    = subFiles(3:end);          % skip '.' and '..'

    fprintf('--- Patient %d / %d : %s ---\n', ...
            i, length(patientFolders), patientFolders(i).name);

    % ------------------------------------------------------------------
    %  2a. Identify CT file and all RTS mask files in this patient folder
    % ------------------------------------------------------------------
    CTFilePath   = '';
    maskpath_all = {};
    SS_all       = {};
    p            = 1;

    for m = 1 : length(subFiles)
        fname = subFiles(m).name;

        % CT volume
        if contains(fname, 'CT.nii', 'IgnoreCase', true)
            CTFilePath = fullfile(patientDir, fname);
        end

        % Segmentation mask(s)
        if contains(fname, 'RTS', 'IgnoreCase', true)
            maskpath_all{p} = fullfile(patientDir, fname);   %#ok<AGROW>
            tokens  = strsplit(fname, '.');
            segName = strrep(tokens{1}, 'RTS_', '');
            SS_all{p} = segName;                              %#ok<AGROW>
            p = p + 1;
        end
    end

    % ------------------------------------------------------------------
    %  2b. LTAd CT volume
    % ------------------------------------------------------------------
    if isempty(CTFilePath) || ~isfile(CTFilePath)
        warning('No CT file found for patient %s – skipping.\n', ...
                patientFolders(i).name);
        continue
    end

    fprintf('  Loading CT : %s\n', CTFilePath);
    CT       = nii_tool('load', CTFilePath);
    ct_image = CT.img;   %#ok<NASGU>  (kept for potential downstream use)

    % ------------------------------------------------------------------
    %  2c. Loop over masks and extract features
    % ------------------------------------------------------------------
    if isempty(maskpath_all)
        warning('No RTS mask found for patient %s – skipping.\n', ...
                patientFolders(i).name);
        continue
    end

    features_p = struct([]);

    for p = 1 : length(maskpath_all)

        mask_path = maskpath_all{p};

        if ~isfile(mask_path)
            warning('  Mask not found: %s\n', mask_path);
            continue
        end

        fprintf('  [%d/%d] Extracting features for segment: %s\n', ...
                p, length(maskpath_all), SS_all{p});

        % Load mask
        mask       = nii_tool('load', mask_path);

        % ---------------------------------------------------------------
        %  FEATURE EXTRACTION  (core function call)
        % ---------------------------------------------------------------
        features = Standard_VolFea_Extract(CT, mask, featureCategory);

        % Annotate with identifiers
        extractName      = [patientFolders(i).name, '.', SS_all{p}];
        features.Location = {extractName};
        features.name     = {patientFolders(i).name};

        % Accumulate
        if isempty(features_p)
            features_p = features;
        else
            features_p(p) = features;  % structs with identical fields merge cleanly
        end

    end % mask loop

    % ------------------------------------------------------------------
    %  2d. Sanity check: confirm all sub-files belong to this patient
    % ------------------------------------------------------------------
    uniqueNames = unique({features_p.name});
    if numel(uniqueNames) == 1 && strcmp(uniqueNames{1}, patientFolders(i).name)
        fprintf('  [OK] All sub-files validated for patient: %s\n\n', ...
                uniqueNames{1});
    else
        error('Sub-files do not all belong to patient %s !', ...
              patientFolders(i).name);
    end

    Pre_Fea_patient_all{i} = features_p;

end % patient loop

%% -------------------------------------------------------------------------
%  3. SAVE RESULTS
% -------------------------------------------------------------------------

mkdir(savepath);
outFile = fullfile(savepath, 'Case_v1_all.mat');
save(outFile, 'Pre_Fea_patient_all');
fprintf('\nFeature extraction complete.\nResults saved to: %s\n', outFile);


%% =========================================================================
%  LOCAL HELPER FUNCTIONS
%  (Standard_VolFea_Extract and calculateMaxDiameter are separate .m files;
%   include them in the same folder or on the MATLAB path)
% =========================================================================


%% =========================================================================
%  FUNCTION: Standard_VolFea_Extract
%  =========================================================================
%  Extracts Base and/or Advance volumetric features from a CT / mask pair.
%
%  Inputs
%    CT       - NIfTI struct loaded via nii_tool (CT volume)
%    mask     - NIfTI struct loaded via nii_tool (binary segmentation)
%    category - 'base' | 'advance' | 'all'
%
%  Outputs
%    Feature_all - struct with fields depending on category:
%
%      BASE features
%        Met      Metastasis flag (0/1)
%        TC       Tumor Count
%        TV       Total Tumor Volume (cm³)
%        LA       Largest 2-D Area summed across all lesions (cm²)
%        LD       Largest Diameter summed across all lesions (cm)
%        LTV      Largest tumor Volume (cm³)
%        LTA      Largest tumor Area (cm²)
%        LTD      Largest tumor Diameter (cm)
%        L2A      Largest-Two-Lesion Area (cm²)
%        L2D      Largest-Two-Lesion Diameter (cm)
%        ATS      Average Tumor Size (cm³)
%
%      ADVANCE features (computed on 1×1×1 mm isotropic resampling)
%        MaxEquivD    Equivalent diameter of largest lesion (mm)
%        SumEquivD    Sum of equivalent diameters
%        StdEquivD    Std of equivalent diameters
%        MaxMaxAxis   Longest principal axis across all lesions (mm)
%        MinMaxAxis   Shortest principal axis across all lesions (mm)
%        SumMaxAxis   Sum of longest principal axes
%        StdMaxAxis   Std of longest principal axes
%        MaxSolidity  Max solidity across lesions
%        MeanSolidity Mean solidity
%        StdSolidity  Std of solidity
%        Skew         Skewness of CT intensities inside mask
%        Kurt         Kurtosis of CT intensities inside mask
%        Entropy      Entropy of CT intensities inside mask

function [Feature_all] = Standard_VolFea_Extract(CT, mask, category)

if strcmpi(category, 'all')
    category = {'base', 'advance'};
end

% Validate image/mask sizes
if ~isequal(ndims(mask.img), ndims(CT.img))
    error('Mismatch in image dimensions between CT and mask.');
end

% Pixel dimensions (mm)
if ~isequal(CT.hdr.pixdim(2:4), mask.hdr.pixdim(2:4))
    error('Pixel dimensions of CT and mask are inconsistent.');
end
pixeldim  = double(CT.hdr.pixdim(2:4));   % [dx dy dz] in mm

mask_image = mask.img;
CT_image   = CT.img;

%% ---- BASE features -------------------------------------------------------
if sum(contains(category, 'base', 'IgnoreCase', true))

    cc       = bwconncomp(mask_image);
    stats    = regionprops3(cc, 'Basic');
    tumor_count = cc.NumObjects;
    sortedStats = sortrows(stats);

    if tumor_count > 0
        Met = 1;
        TC  = tumor_count;

        % Total tumor volume (cm³)
        TV = round(sum(stats.Volume) * prod(pixeldim) / 1000, 4);

        % Largest-one volume (cm³)
        LTV = round(sortedStats.Volume(end) * prod(pixeldim) / 1000, 4);

        % Per-lesion max 2-D area and max diameter
        Area_all     = zeros(tumor_count, 1);
        maxDiameter  = zeros(tumor_count, 1);

        for obj = 1 : tumor_count
            BW = cc2bw(cc, ObjectsToKeep=obj);
            [~, maskBox, ~] = getSUVbox(CT_image, BW);

            % Area along z-axis slices
            Area = squeeze(sum(maskBox, [1,2])) ...
                   * pixeldim(1) * pixeldim(2) * 0.01;   % cm²
            Area_all(obj) = max(Area);

            % Max diameter per slice
            Diameters = zeros(1, size(maskBox, 3));
            for s = 1 : size(maskBox, 3)
                slice = maskBox(:,:,s);
                Diameters(s) = calculateMaxDiameter(slice) ...
                               * pixeldim(1) * 0.1;       % cm
            end
            maxDiameter(obj) = max(Diameters);
        end

        LA  = sum(Area_all);
        LTA = max(Area_all);
        LA_desc = sort(Area_all, 'descend');
        L2A = sum(LA_desc(1 : min(2, end)));

        LD  = sum(maxDiameter);
        LTD = max(maxDiameter);
        LD_desc = sort(maxDiameter, 'descend');
        L2D = sum(LD_desc(1 : min(2, end)));

        % Longest Z-axis range
        % [~, ~, boxBound] = getSUVbox(CT_image, mask_image);
        % Boxdim = boxBound(:,2) - boxBound(:,1) + 1;
        % LR  = Boxdim .* pixeldim .* 0.1;   % cm
        % LZR = LR(3);

        ATS = TV/TC;

    else
        Met=0; TC=0; TV=0; LA=0; LD=0; LTV=0;
        LTA=0; LTD=0; LZR=0; L2A=0; L2D=0; ATS=0;
    end

    Fea_base = struct('Met',Met,'TC',TC,'TV',TV,'LA',LA,'LD',LD, ...
                      'LTV',LTV,'LTA',LTA,'LTD',LTD,'LZR',LZR, ...
                      'L2A',L2A,'L2D',L2D,'ATS',ATS);
    Feature_all = Fea_base;
end

%% ---- ADVANCE features ----------------------------------------------------
if sum(contains(category, 'advance', 'IgnoreCase', true))

    % Resample mask to 1×1×1 mm isotropic
    targetSpacing = [1 1 1];
    newSize  = round(size(CT_image) .* (pixeldim ./ targetSpacing));
    mask_iso = imresize3(mask_image, newSize, 'nearest');

    cc_iso  = bwconncomp(mask_iso);
    stats2  = regionprops3(cc_iso, 'all');

    if cc_iso.NumObjects > 0
        MaxEquivD   = max(stats2.EquivDiameter);
        SumEquivD   = sum(stats2.EquivDiameter);
        StdEquivD   = std(stats2.EquivDiameter);

        max_axis    = max(stats2.PrincipalAxisLength, [], 2);
        MaxMaxAxis  = max(max_axis);
        SumMaxAxis  = sum(max_axis);
        StdMaxAxis  = std(max_axis);

        min_axis    = min(stats2.PrincipalAxisLength, [], 2);
        MinMaxAxis  = max(min_axis);

        MaxSolidity  = max(stats2.Solidity);
        MeanSolidity = mean(stats2.Solidity);
        StdSolidity  = std(stats2.Solidity);

        [CTbox, ~, ~] = getSUVbox(CT_image, mask_image);
        Skew    = skewness(CTbox(:));
        Kurt    = kurtosis(CTbox(:));
        Entropy = entropy(CTbox(:));
    else
        MaxEquivD=0; SumEquivD=0; StdEquivD=0;
        MaxMaxAxis=0; MinMaxAxis=0; SumMaxAxis=0; StdMaxAxis=0;
        MaxSolidity=0; MeanSolidity=0; StdSolidity=0;
        Skew=0; Kurt=0; Entropy=0;
    end

    Fea_advance = struct( ...
        'MaxEquivD',MaxEquivD,'SumEquivD',SumEquivD,'StdEquivD',StdEquivD, ...
        'MaxMaxAxis',MaxMaxAxis,'MinMaxAxis',MinMaxAxis, ...
        'SumMaxAxis',SumMaxAxis,'StdMaxAxis',StdMaxAxis, ...
        'MaxSolidity',MaxSolidity,'MeanSolidity',MeanSolidity, ...
        'StdSolidity',StdSolidity,'Skew',Skew,'Kurt',Kurt,'Entropy',Entropy);
    Feature_all = Fea_advance;
end

%% ---- Merge if both categories requested ----------------------------------
if sum(contains(category,'base','IgnoreCase',true)) && ...
   sum(contains(category,'advance','IgnoreCase',true))
    Feature_all = merge2Structs(Fea_base, Fea_advance);
end

end   % Standard_VolFea_Extract


%% =========================================================================
%  FUNCTION: calculateMaxDiameter
%  =========================================================================
%  Computes the maximum caliper diameter (Feret diameter) of a 2-D binary mask.
%
%  Input
%    mask        - logical 2-D array (1 = foreground, 0 = background)
%
%  Output
%    maxDiameter - maximum boundary-to-boundary distance (pixels)

function maxDiameter = calculateMaxDiameter(mask)

    if isempty(mask) || ~islogical(mask)
        error('Input must be a non-empty logical 2-D array.');
    end

    boundaries   = bwboundaries(mask);
    maxDiameter  = 0;

    for k = 1 : length(boundaries)
        boundary    = boundaries{k};
        distMatrix  = pdist2(boundary, boundary);
        maxDiameter = max(maxDiameter, max(distMatrix(:)));
    end

end   % calculateMaxDiameter

%% =========================================================================
%  FUNCTION: getSUVbox
%  =========================================================================
function [SUVbox, maskBox, boxBound] = getSUVbox(volume,mask)
% COMPUTATION OF THE SMALLEST BOX CONTAINING THE ROI
volume = double(volume);
[boxBound] = computeBoundingBox(mask);
maskBox = mask(boxBound(1,1):boxBound(1,2),boxBound(2,1):boxBound(2,2),boxBound(3,1):boxBound(3,2));
SUVbox = volume(boxBound(1,1):boxBound(1,2),boxBound(2,1):boxBound(2,2),boxBound(3,1):boxBound(3,2));
end

%% =========================================================================
%  FUNCTION: merge2Structs
%  =========================================================================
function combinedStruct = merge2Structs(struct1, struct2)
    combinedStruct = struct();
    fields1 = fieldnames(struct1);
    for i = 1:length(fields1)
        combinedStruct.(fields1{i}) = struct1.(fields1{i});
    end
    fields2 = fieldnames(struct2);
    for i = 1:length(fields2)
        combinedStruct.(fields2{i}) = struct2.(fields2{i});
    end
end

%% =========================================================================
%  FUNCTION: cc2bw  %   Copyright 2023 The MathWorks, Inc.
%  =========================================================================
function bw = cc2bw(cc, options)
    arguments
        cc (1, 1) struct {checkCC}
        options.ObjectsToKeep {mustBeInteger, mustBeValidSelection} ...
                                                = 1:length(cc.PixelIdxList)
    end

    bw = false(cc.ImageSize);

    if isnumeric(options.ObjectsToKeep)
        % Avoid wasteful multiple copies of the input to output if the same
        % object is selected multiple times
        sel = unique(options.ObjectsToKeep);
    else
        sel = options.ObjectsToKeep;
    end

    pixIdxListToKeep = cc.PixelIdxList(sel);

    for k = 1:length(pixIdxListToKeep)
        bw(pixIdxListToKeep{k}) = true;
    end
end

function mustBeValidSelection(sel)
    % Allow empty selections. This can occur if the condition the user
    % specified resulted in no objects being selected.
    if ~isempty(sel)
        mustBeVector(sel);
    end
end
