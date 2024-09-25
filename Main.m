%Analysis of Network Features for Intrusion Detection Based on Bio-Inspired Algorithms

clear, clc, close;

Runs = 20; %Number of runs

% Load dataset
%load unsw.mat 
load nsl.mat

label = categ;
feat = feat;

% Number of k in K-nearest neighbor
opts.k = 2*floor(sqrt(size(feat,1))/2)+1; 

% Define k-fold training and validation sets
HO2 = cvpartition(categ,'k',10);

% Common parameter settings 
opts.N = 10;     % number of solutions
opts.T = 40;    % maximum number of iterations
opts.thres = 0.5;
opts.NR = 1; 
opts.Modelf = HO2; %K-fold Model

% Ratio of validation data
ho = 0.2;
HO1 = cvpartition(label,'HoldOut',ho);
opts.Model = HO1;

%% Whale Optimization Algorithm (WOA) 

% Parameter of WOA
opts.b = 1;

% Perform feature selection 

for i = 1:Runs
    FS     = FeatureSelection('woa',feat,label,opts);
    % Define index of selected features
    sf_idx = FS.sf_idx;
    % Metrics using KNN from the selected features 
    [Acc(i),FMe(i),Recall(i),Prec(i),FPR(i),FNR(i),TNR(i)] = KNN(feat,label,opts)
    Sf(i,:) = FS.sf;  
    fprintf('\n Run: %g',i); %Run count
end

%Metrics for WOA for each Selected Features Run
WOA.Acc    = Acc; 
WOA.FMe    = FMe; 
WOA.Recall = Recall;
WOA.Prec   = Prec;
WOA.FPR    = FPR;
WOA.FNR    = FNR;
WOA.TNR    = TNR;
WOA.Sf     = Sf; %Selected features 0=no / 1=yes (Each row = 1 Run)

%% Ant Lion Optimizer (ALO)

clear FS

% Perform feature selection 

for i = 1:Runs
    FS     = FeatureSelection('alo',feat,label,opts);
    % Define index of selected features
    sf_idx = FS.sf_idx;
    % Metrics using KNN from the selected features 
    [Acc(i),FMe(i),Recall(i),Prec(i),FPR(i),FNR(i),TNR(i)] = KNN(feat,label,opts)
    Sf(i,:) = FS.sf;  
    fprintf('\n Run: %g',i); %Run count
end

%Metrics for ALO for each Selected Features Run
ALO.Acc    = Acc; 
ALO.FMe    = FMe; 
ALO.Recall = Recall;
ALO.Prec   = Prec;
ALO.FPR    = FPR;
ALO.FNR    = FNR;
ALO.TNR    = TNR;
ALO.Sf     = Sf; %Selected features 0=no / 1=yes (Each row = 1 Run)

%% Krill Herd Optmization (KH) 

clear FS 

% Parameter of KH
opts.C  = 1;  % Crossover flag [Yes=1]
opts.NK = 10; % Number of krill

% Perform feature selection 

for i = 1:Runs
    FS     = FeatureSelection('kh',feat,label,opts);
    % Define index of selected features
    sf_idx = FS.sf_idx;
    % Metrics using KNN from the selected features 
    [Acc(i),FMe(i),Recall(i),Prec(i),FPR(i),FNR(i),TNR(i)] = KNN(feat,label,opts)
    Sf(i,:) = FS.sf;  
    fprintf('\n Run: %g',i); %Run count
end

%Metrics for KH for each Selected Features Run
KH.Acc    = Acc; 
KH.FMe    = FMe; 
KH.Recall = Recall;
KH.Prec   = Prec;
KH.FPR    = FPR;
KH.FNR    = FNR;
KH.TNR    = TNR;
KH.Sf     = Sf; %Selected features 0=no / 1=yes (Each row = 1 Run)

%% Grey Wolf Optimizer(GWO)

clear FS

% Parameter of GWO
opts.b = 1;

% Perform feature selection 

for i = 1:Runs
    FS     = FeatureSelection('gwo',feat,label,opts);
    % Define index of selected features
    sf_idx = FS.sf_idx;
   % Metrics using KNN from the selected features 
    [Acc(i),FMe(i),Recall(i),Prec(i),FPR(i),FNR(i),TNR(i)] = KNN(feat,label,opts)
    Sf(i,:) = FS.sf;  
    fprintf('\n Run: %g',i); %Run count
end

%Metrics for GWO for each Selected Features Run
GWO.Acc    = Acc; 
GWO.FMe    = FMe; 
GWO.Recall = Recall;
GWO.Prec   = Prec;
GWO.FPR    = FPR;
GWO.FNR    = FNR;
GWO.TNR    = TNR;
GWO.Sf     = Sf; %Selected features 0=no / 1=yes (Each row = 1 Run)

%% Cuckoo Search Algorithm (CSA)

clear FS

% Parameter of CSA
opts.b = 1;

% Perform feature selection 

for i = 1:Runs
    FS     = FeatureSelection('csa',feat,label,opts);
    % Define index of selected features
    sf_idx = FS.sf_idx;
    % Metrics using KNN from the selected features 
    [Acc(i),FMe(i),Recall(i),Prec(i),FPR(i),FNR(i),TNR(i)] = KNN(feat,label,opts)
    Sf(i,:) = FS.sf;  
    fprintf('\n Run: %g',i); %Run count
end

%Metrics for CS for each Selected Features Run
CS.Acc    = Acc; 
CS.FMe    = FMe; 
CS.Recall = Recall;
CS.Prec   = Prec;
CS.FPR    = FPR;
CS.FNR    = FNR;
CS.TNR    = TNR;
CS.Sf     = Sf; %Selected features 0=no / 1=yes (Each row = 1 Run)
