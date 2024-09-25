%Normalization code for the NSL-KDD dataset
clc
close all
clear all

feat = importdata('Datasets/NSL-KDD/KDDTest+.txt'); %Change the file
cat = feat.textdata(:,42);
feat = feat.textdata(:,1:41);

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

feat = cellfun(@str2num,feat);

% Transforma a categ em matriz

Index = strfind(cat, 'normal');
idx = cellfun(@isempty,Index); 
Index(idx) = {0};

categ = cell2mat(Index);

features = feat;
categoria = categ;