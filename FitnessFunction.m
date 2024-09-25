% Fitness Function KNN (9/12/2020)

function cost = FitnessFunction2(feat,label,X,opts)
% Default of [alpha; beta]
ws = [0.975; 0.475; 0.025];

if isfield(opts,'ws'), ws = opts.ws; end

% Check if any feature exist
if sum(X == 1) == 0
  cost = 100;
else
  % Error rate
  [dr,far,acc]    = wrapper_DecTree(feat(:,X == 1),label,opts);
  % Number of selected features
  num_feat = sum(X == 1);
  % Total number of features
  max_feat = length(X); 
  % Set alpha & beta
  alpha    = ws(1); 
  beta     = ws(2);
  gama     = ws(3);
  % Cost function 
  %cost     = 100*(alpha * (1-dr) + far * beta *  + gama * (num_feat / max_feat)); 
  cost     = 100*(2*alpha * (1-acc) + gama * (num_feat / max_feat)); 
end
end


%---Call Functions-----------------------------------------------------
function [DR,Far,Acc] = wrapper_DecTree(sFeat,label,opts)

if isfield(opts,'Model'), Model = opts.Model; end
if isfield(opts,'k'), k = opts.k; end

    
    % Define training & validation sets    
    trainIdx = Model.training;    testIdx = Model.test;
    xtrain   = sFeat(trainIdx,:);  ytrain  = label(trainIdx);
    xvalid   = sFeat(testIdx,:);   yvalid  = label(testIdx);
    % Training model
    My_Model = fitcknn(xtrain,ytrain,'NumNeighbors',k);
    % Prediction
    pred     = predict(My_Model,xvalid);
    
    TP = sum(pred & yvalid);
    TN = sum(~pred & ~yvalid);
    FP = sum(~pred & yvalid);
    FN = sum(pred & ~yvalid);
        
    %False Alarm Rate
    Far  = FP / (FP + TN);
    
    %Detection Rate
    DR   = TP / (TP + FN);

    Acc  = (TP + TN) / (TP + TN + FP + FN);
end












