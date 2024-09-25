% Wrapper Feature Selection Toolbox by Jingwei Too - 9/12/2020

function model = FeatureSelection(type,feat,label,opts)
switch type
  case 'kh'   ; fun = @KrillHerdOptimization;
  case 'alo'  ; fun = @AntLionOptimizer;    
  case 'woa'  ; fun = @WhaleOptimizationAlgorithm;
  case 'gwo'  ; fun = @GreyWolfOptimizer;
  case 'csa'  ; fun = @CuckooSearchAlgorithm;
end
tic;
model = fun(feat,label,opts); 
% Computational time
t = toc;

model.t = t;
fprintf('\n Processing Time (s): %f % \n',t); fprintf('\n');
end


