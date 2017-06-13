function values = weakclass( numb,dataset,labels,w,test )
%Inputs
%numb->switch between classifiers
%dataset->training data
% labels->natural class of dataset
% w->weights
% test->test data
% Output
% Values->predicted classes
if numb==11
    mdl=fitcknn(dataset,labels,'NumNeighbors',2,'Distance','cityblock','Weights',w);
    values=predict(mdl,test);
elseif numb==12
     mdl=fitcknn(dataset,labels,'NumNeighbors',2,'Distance','chebychev','Weights',w);
    values=predict(mdl,test);
elseif numb==13
     mdl=fitcknn(dataset,labels,'NumNeighbors',2,'Distance','correlation','Weights',w);
   values=predict(mdl,test);
elseif numb==14
     mdl=fitcknn(dataset,labels,'NumNeighbors',2,'Distance','cosine','Weights',w);
    values=predict(mdl,test);
elseif numb==15
     mdl=fitcknn(dataset,labels,'NumNeighbors',2,'Distance','euclidean','Weights',w);
    values=predict(mdl,test);
elseif numb==16
     mdl=fitcknn(dataset,labels,'NumNeighbors',2,'Distance','hamming','Weights',w);
    values=predict(mdl,test);
elseif numb==17
     mdl=fitcknn(dataset,labels,'NumNeighbors',2,'Distance','jaccard','Weights',w);
   values=predict(mdl,test);
elseif numb==18
     mdl=fitcknn(dataset,labels,'NumNeighbors',2,'Distance','mahalanobis','Weights',w);
    values=predict(mdl,test);
elseif numb==19
     mdl=fitcknn(dataset,labels,'NumNeighbors',2,'Distance','minkowski','Weights',w);
    values=predict(mdl,test);
elseif numb==20
     mdl=fitcknn(dataset,labels,'NumNeighbors',2,'Distance','spearman','Weights',w);
    values=predict(mdl,test);
elseif numb==1
     mdl=fitcnb(dataset,labels,'ScoreTransform','invlogit','Weights',w);
    values=predict(mdl,test);
elseif numb==2
     mdl=fitcnb(dataset,labels,'ScoreTransform','none','Weights',w);
    values=predict(mdl,test);
elseif numb==3
     mdl=fitcnb(dataset,labels,'ScoreTransform','symmetricismax','Weights',w);
    values=predict(mdl,test);
elseif numb==4
     mdl=fitcnb(dataset,labels,'Prior','uniform','Weights',w);
    values=predict(mdl,test);
elseif numb==5
     mdl=fitcnb(dataset,labels,'ScoreTransform','sign','Weights',w);
    values=predict(mdl,test);
elseif numb==6
     mdl=fitcnb(dataset,labels,'ScoreTransform','doublelogit','Weights',w);
    values=predict(mdl,test);
elseif numb==7
     mdl=fitcnb(dataset,labels,'ScoreTransform','ismax','Weights',w);
   values=predict(mdl,test);
elseif numb==8
     mdl=fitcnb(dataset,labels,'ScoreTransform','logit','Weights',w);
    values=predict(mdl,test);
elseif numb==9
     mdl=fitcnb(dataset,labels,'ScoreTransform','symmetric','Weights',w);
    values=predict(mdl,test);
elseif numb==10
     mdl=fitcnb(dataset,labels,'ScoreTransform','symmetriclogit','Weights',w);
    values=predict(mdl,test);
    
elseif numb==21
    mdl=fitcdiscr(dataset,labels,'DiscrimType','quadratic','Weights',w);
    values=predict(mdl,test);
elseif numb==22
    mdl=fitcdiscr(dataset,labels,'DiscrimType','linear','Weights',w);
    values=predict(mdl,test);
elseif numb==23
    mdl=fitcdiscr(dataset,labels,'DiscrimType','diagLinear','Weights',w);
    values=predict(mdl,test);
elseif numb==24
    mdl=fitcdiscr(dataset,labels,'DiscrimType','diagQuadratic','Weights',w);
    values=predict(mdl,test);
elseif numb==25
    mdl=fitcdiscr(dataset,labels,'DiscrimType','pseudoLinear','Weights',w);
    values=predict(mdl,test);
elseif numb==26
    mdl=fitcdiscr(dataset,labels,'DiscrimType','pseudoQuadratic','Weights',w);
    values=predict(mdl,test);
elseif numb==27
    mdl=fitcdiscr(dataset,labels,'ScoreTransform','logit','Weights',w);
    values=predict(mdl,test);
elseif numb==28
    mdl=fitcdiscr(dataset,labels,'ScoreTransform','doublelogit','Weights',w);
    values=predict(mdl,test);
elseif numb==29
    mdl=fitcdiscr(dataset,labels,'ScoreTransform','ismax','Weights',w);
    values=predict(mdl,test);
else
    mdl=fitcdiscr(dataset,labels,'ScoreTransform','sign','Weights',w);
    values=predict(mdl,test);
end
end

