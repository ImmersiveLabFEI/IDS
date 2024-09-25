%Normalization code for the UNSW-NB15 dataset

clc
close all
clear all

feat = readtable('Datasets/UNSW-NB15/UNSW_NB15_training-set.csv'); %change the file

categ = table2cell(feat(:,45));
feat = table2cell(feat(:,2:43));

% Transform colums from text to number
for i = 2:4
    dif = unique(feat(:,i));
    index = size(dif,1);
    for j = 1:index 
        subs = find(strcmp(dif(j,1),feat(:,i)));
        for k = 1:length(subs)
            feat{subs(k),i} = num2str(j);
        end
    end
end

% Transforma a categ em matriz

feat1 = cell2mat(feat(:,1));
feataux = cellfun(@str2num,feat(:,[2,3,4]));
feat2 = cell2mat(feat(:,5:end));

features = [feat1,feataux,feat2];

categoria = cell2mat(categ);
