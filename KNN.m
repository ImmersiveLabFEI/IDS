% K-nearest Neighbor (9/12/2020)

function [Acc,FMe,Recall,Prec,FPR,FNR,TNR] = KNN(feat, label,opts)

if isfield(opts,'k'), k = opts.k; end
if isfield(opts,'Modelf'), Model = opts.Modelf; end

HO2 = cvpartition(label,'k',10);
Model = HO2;

% Define training & validation sets

    for i = 1:Model.NumTestSets
        
    trainIdx = Model.training(i);    testIdx = Model.test(i);
    xtrain   = feat(trainIdx,:);  ytrain  = label(trainIdx);
    xvalid   = feat(testIdx,:);   yvalid  = label(testIdx);
    % Training model
    My_Model = fitcknn(xtrain,ytrain,'NumNeighbors',k); 
    
    % Prediction
    pred     = predict(My_Model,xvalid);
    
    TP = sum(pred & yvalid);
    TN = sum(~pred & ~yvalid);
    FP = sum(~pred & yvalid);
    FN = sum(pred & ~yvalid);

    %Metrics
    recall(i) = TP / (TP + FN);
    fnr(i) = FN / (TP + FN);
    tnr(i) = TN / (TN + FP);
    fpr(i) = FP / (FP + TN);
    prec(i) = TP / (TP + FP);
    acc(i) = (TP + TN) / (TP + TN + FP + FN);
    fme(i) = 2.*prec.*recall / (prec+recall);

    end

Acc = mean(acc);
Recall = mean(recall);
FNR = mean(fnr);
TNR = mean(tnr);
FPR = mean(fpr);
Prec = mean(prec);
FMe = mean(fme);


end


