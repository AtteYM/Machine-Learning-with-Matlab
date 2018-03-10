function Fisher_goodness_of_features

load('male_and_female.mat') %load data for classifier

%Calculate FDR for all features
FDR_H = fisher_linear(height_male_cm, height_female_cm);
FDR_S = fisher_linear(shoe_male_EU_size, shoe_female_EU_size);
FDR_W = fisher_linear(weight_male_kg, weight_female_kg);

%Remove one feature with smallest Fisher Discriminant Ratio.
if (FDR_S > FDR_H) & (FDR_W > FDR_H)
    conclusion = "Height had smallest Fisher Discriminant Ratio and"...
        + " will be removed.";
    feature1 = 1;
    feature2 = 2;
elseif (FDR_W > FDR_S) & (FDR_H > FDR_S)
    conclusion = "Shoe size had smallest Fisher Discriminant Ratio and"...
        + " will be removed.";
    feature1 = 2;
    feature2 = 3;
elseif (FDR_S > FDR_W) & (FDR_H > FDR_W)
    conclusion = "Weight had smallest Fisher Discriminant Ratio and"...
        + " will be removed.";
    feature1 = 1;
    feature2 = 3;
end

disp(conclusion); %display which feature was removed
test_FDR(feature1, feature2) %test fitcnb with only 2 features

end

%Fisher discriminant ratio (FDR)
function FDR=fisher_linear(feature1,feature2) %two sets of features

mean1=mean(feature1);   %compute mean of feature1
mean2=mean(feature2);   %compute mean of feature2
var1=var(feature1);     %compute variance of feature1
var2=var(feature2);     %compute variance of feature2

FDR=(mean1-mean2)^2/(var1+var2); %compute FDR
end

%function that makes naive Bayes model using fitncb function using only
%70% of original data and leaving 30% for testing
function fitcnb_malli = test_FDR(feature1, feature2)
load('male_and_female.mat') %load data for classifier

%make 2*10000 array from data and turn it to 10000*2 array
%choose only 2 features based on Fisher Discriminant Ratio
if (feature1==1) & (feature2==2)    
    P = [shoe_female_EU_size shoe_male_EU_size;...
    weight_female_kg weight_male_kg];
elseif (feature1==2) & (feature2==3)
    P = [height_female_cm height_male_cm ;...
    weight_female_kg weight_male_kg];
elseif (feature1==1) & (feature2==3)
    P = [height_female_cm height_male_cm ;...
    shoe_female_EU_size shoe_male_EU_size];
else
    Error = "couldn't find removable feature with FDR"
end
P = P';

%create variables containing gender and empty array to store that data
gender1 = 'female';
gender2 = 'male';
genders = {};

%create array with 5000 'female' first and 5000 'male' in the end
for i = 1:10000
    if i<5001
        genders = [genders gender1];
    else
        genders = [genders gender2];
    end
end

genders = genders'; %Turn array to 10000*1


X = P;
Y = genders;

%Create model with 70% of data and holdout 30% for testing
CVMdl = fitcnb(X,Y,'Holdout',0.30,'ClassNames',{'female','male'});
CMdl = CVMdl.Trained{1};          % Extract trained, compact classifier
testIdx = test(CVMdl.Partition); % Extract the test indices
XTest = X(testIdx,:);
YTest = Y(testIdx);

%Create random sample of 20 testsamples
idx = randsample(sum(testIdx),20);

%Store testsamples in testidata
testidata = XTest(idx,:);

%make predictions using model created with fitcnb
% if ';' is deleted function would also display propabilites (Cost)
[label,Posterior,Cost] = predict(CMdl,XTest);


%create table for comparison of the original values and predictions made
%with fitcnb
comparison_table = table(YTest(idx),label(idx),'VariableNames',...
    {'TrueLabel','PredictedLabel'})

%check if all 20 testsamples are classified correctly
if isempty(setdiff(YTest(idx),label(idx)))
    test_result = "All classification results match with 2 features ."
else
    test_result = "Not all reslults match. Can't reduce features."
end
end