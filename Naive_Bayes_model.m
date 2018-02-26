%function that makes naive Bayes model using fitncb function using only
%70% of original data and leaving 30% for testing
function fitcnb_malli
load('male_and_female.mat') %load data for classifier

%make 3*10000 array from data and turn it to 10000*3 array
P = [height_female_cm height_male_cm ;...
    shoe_female_EU_size shoe_male_EU_size ;...
    weight_female_kg weight_male_kg];
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

%Store testsamples in testidata and create empty henkilot array
testidata = XTest(idx,:);
henkilot ={};

%for every testsample use code from Naive_bayes_classifier to predict
%gender using vertailukoodi (modified code from Naive_bayes_classifier.m)
%and store it in henkilot array
for i = 1:20
    xt=testidata(i,1);
    yt=testidata(i,2);
    zt=testidata(i,3);
    henkilot = [henkilot Naive_bayes_classifier(xt,yt,zt)];
end

%plotting data to see the patterns
figure(3)
plot3(height_male_cm,weight_male_kg,shoe_male_EU_size,'x');
hold on;
plot3(height_female_cm,weight_female_kg,shoe_female_EU_size,'rx');
hold off;

%turn henkilot 1*20 array in to 20*1
henkilot = henkilot';

%make predictions using model created with fitcnb
% if ';' is deleted function would also display propabilites (Cost)
[label,Posterior,Cost] = predict(CMdl,XTest);


%create table for comparison of the original values, predictions made
%with fitcnb and predictions made with Naive_bayes_classifier code named
%as "vertailukoodi"
table(YTest(idx),label(idx),henkilot,'VariableNames',{'TrueLabel',...
    'PredictedLabel','vertailukoodi'})
end

%function made by modifying code from Naive_bayes_classifier.m
%instead of giving input() we give the values when function is called
%also output is modified to give only predicted gender
%figure is removed as unusefull for this function
function henkilo = Naive_bayes_classifier(tuntematon_height,...
    tuntematon_shoe,tuntematon_weight)
load('male_and_female.mat')

mean_shoe_male=mean(shoe_male_EU_size);
mean_shoe_female=mean(shoe_female_EU_size);
var_shoe_male=var(shoe_male_EU_size);
var_shoe_female=var(shoe_female_EU_size);

mean_weight_male=mean(weight_male_kg);
mean_weight_female=mean(weight_female_kg);
var_weight_male=var(weight_male_kg);
var_weight_female=var(weight_female_kg);

mean_height_male=mean(height_male_cm);
mean_height_female=mean(height_female_cm);
var_height_male=var(height_male_cm);
var_height_female=var(height_female_cm);

p_male=0.5; %Let us assume that 50% of the population are males, is it?
p_female=1-p_male;

p_male_height=gauss(tuntematon_height,mean_height_male,var_height_male); %is gauss distribution assumption right for height, weight and shoe size distribution?
p_male_weight=gauss(tuntematon_weight,mean_weight_male,var_weight_male);
p_male_shoe=gauss(tuntematon_shoe,mean_shoe_male,var_shoe_male);

p_female_height=gauss(tuntematon_height,mean_height_female,var_height_female);
p_female_weight=gauss(tuntematon_weight,mean_weight_female,var_weight_female);
p_female_shoe=gauss(tuntematon_shoe,mean_shoe_female,var_shoe_female);

denominator=p_male*p_male_height*p_male_weight*p_male_shoe+...
    p_female*p_female_height*p_female_weight*p_female_shoe;

prob_male=(p_male*p_male_height*p_male_weight*p_male_shoe)/denominator;
prob_female=(p_female*p_female_height*p_female_weight*p_female_shoe)/denominator;

if prob_male<prob_female
    henkilo = 'female';
else
    henkilo = 'male';
end

end

function t=gauss(x,mu,var_x) %cost or probability function
t=(1/sqrt(2*pi*var_x))*exp((-(x-mu).^2)/(2*var_x)); %envelope of Gaussian distribution
end