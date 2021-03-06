function finallabels = adaboostV2( dataset,instanceclass,testdata,subset )

%Inputs
% dataset->training data
% instanceclass->natural class of dataset
% testdata->test data
%subset->maximum size of subset
% Output
% finallabels->predicted classes


%initilaze weights
weights=ones(1040,1)./1040;
k=1;
while k~=31
    y=zeros(1040,1);
    %select subset according to weight values.
    [ndata,idk]=datasample(dataset,subset,'Weights',weights,'Replace',false);
    nclass=instanceclass(idk);
    nweights=weights(idk);
    %Fit classifier to the training set using weights wi
    s=find(weakclass(k,ndata,nclass,nweights,ndata)~=nclass);
    accuracy=(subset-length(s))/subset;
     %if accuracy<0.5, dont use this weak classifier
    if(accuracy<0.5)
        alfa(k)=0;
        k=k+1;
        continue;
    %if accuracy=1, set trust 1
    elseif accuracy==1
            alfa(k)=1;
            k=k+1;
        continue;
    end
    %Compute error
    y(s)=1;
    err=sum(weights.*y)/sum(weights);
    %Compute trust
    alfa(k)=log((1-err)/err);
    %Update and normalize weights
    weights=weights.*exp(alfa(k).*y);
    weights=weights./sum(weights);
     k=k+1;
end


alfa=alfa';
C1=zeros(length(testdata),1);
C0=zeros(length(testdata),1);

w2=ones(1040,1)./1040;
% compute C0-> y=0 C1-> y=1
for i=1:30,

C1=C1+weakclass(i,dataset,instanceclass,w2,testdata).*alfa(i);
temp=zeros(length(testdata),1);
temp(find(weakclass(i,dataset,instanceclass,w2,testdata)==0))=1;
C0=C0+temp.*alfa(i);
end
%select maximum y
finallabels=zeros(length(testdata),1);
finallabels(find(C1>C0))=1;


end

