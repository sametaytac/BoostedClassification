fileID = fopen('train_data.prj','r');
%take traindata
for i=1:1040,
    ab=strsplit(fgetl(fileID),',');
Big(i,:)=[1;cellfun(@str2num,ab).'];
end
fclose(fileID);
dataset=Big(:,3:28);
instanceclass=Big(:,30);
nsdataset=dataset;
%feature scaling with range(0,1) to traindata 
for i=1:26,
dataset(:,i)=dataset(:,i)./ max(dataset(:,i));
end

%take testdata
fileID2 = fopen('test_data.prj','r');
for i=1:168,
    ab=strsplit(fgetl(fileID2),',');
Big2(i,:)=[1;cellfun(@str2num,ab).'];
end
fclose(fileID2);
testdata=Big2(:,3:28);
label2=Big2(:,29);
nstestdata=testdata;
%feature scaling with range(0,1) to testdata 
for i=1:26,
testdata(:,i)=testdata(:,i)./ max(testdata(:,i));
end

%Experience 1:try adaboostV1 for find accurancy on trainingdata
C0=adaboost(dataset,instanceclass,dataset);
accuracyexp1=length(find(instanceclass==C0))/1040;
%plot roc
figure
plotroc(instanceclass',C0');
%plot data instances
drawtrue=nsdataset(find(C0==instanceclass),:);
drawfalse=nsdataset(find(C0~=instanceclass),:);

figure
scatter(drawtrue(:,24),drawtrue(:,13),'b');
hold on;
scatter(drawfalse(:,24),drawfalse(:,13),'r');
hold on;


%Experience 2:try adaboostV1 for find accurancy on testdata
C1=adaboost(dataset,instanceclass,testdata);
accuracyexp2=length(find(label2==C1))/168;
%plot roc
figure
plotroc(label2',C1');


%Experience 3:try adaboostV2 for find accurancy on trainingdata with max subspace=500.
C2=adaboostV2(dataset,instanceclass,dataset,500);
accuracyexp3=length(find(instanceclass==C2))/1040;
%plot roc
figure
plotroc(instanceclass',C2');


%Experience 4:try adaboostV2 for find accurancy on testdata with max subspace=500.
C3=adaboostV2(dataset,instanceclass,testdata,500);
accuracyexp4=length(find(label2==C3))/168;

%plot data instances
drawtrue2=nstestdata(find(C3==label2),:);
drawfalse2=nstestdata(find(C3~=label2),:);

figure
scatter(drawtrue2(:,18),drawtrue2(:,21),'b');
hold on;
scatter(drawfalse2(:,18),drawfalse2(:,21),'r');
hold on;

%plot roc
figure
plotroc(label2',C3');




%Experience 5:try adaboostV2 for find accurancy on trainingdata with max subspace=700.

C4=adaboostV2(dataset,instanceclass,dataset,700);
accuracyexp5=length(find(instanceclass==C4))/1040;

%plot roc
figure
plotroc(instanceclass',C4');



%Experience 6:try adaboostV2 for find accurancy on testdata with max subspace=700.

C5=adaboostV2(dataset,instanceclass,testdata,700);
accuracyexp6=length(find(label2==C5))/168;

%plot roc
figure
plotroc(label2',C5');



%%%pca start;
%find coeff and explained
[coeff,~,~,~,explained] = pca(dataset);
exp=0;
iter=1;
%find how many feature needed to explain more than %80
while exp<80
exp=exp+explained(iter);
iter=iter+1;
end
%adjust new features
pcatrain=dataset*coeff(1:iter-1,:)';
pcatest=testdata*coeff(1:iter-1,:)';
%Experience 7:adaboostV1 with pca for find accurancy on testdata
C6=adaboost(pcatrain,instanceclass,pcatest);
accuracyexp7=length(find(label2==C6))/168;
%plot roc
figure
plotroc(label2',C6');

%Experience 8:adaboostV2 with pca for find accurancy on testdata with supspace size=500
C7=adaboostV2(pcatrain,instanceclass,pcatest,500);
accuracyexp8=length(find(label2==C7))/168;

%plot roc
figure
plotroc(label2',C7');


% I select probabilistic pca for dimensionality reduction technique.
%find coeffs
coeffp = ppca(dataset,10);
%adjust new features
ppcatrain=dataset*coeffp;
ppcatest=testdata*coeffp;
%Experience 9:adaboostV1 with ppca for find accurancy on testdata.
C8=adaboost(ppcatrain,instanceclass,ppcatest);
accuracyexp9=length(find(label2==C8))/168;

%plot roc
figure
plotroc(label2',C8');



%Experience 10:adaboostV2 with ppca for find accurancy on testdata with supspace size=500.
C9=adaboostV2(ppcatrain,instanceclass,ppcatest,500);
accuracyexp10=length(find(label2==C9))/168;

%plot roc
figure
plotroc(label2',C9');



