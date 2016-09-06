##Set working directory ####
setwd("/Volumes/Prasant3TB/04. Academic/02. Data Science/Kaggle")

## Load Data ####
train <- read.csv("/Volumes/Prasant3TB/04. Academic/02. Data Science/Kaggle/train.csv")
test <- read.csv("/Volumes/Prasant3TB/04. Academic/02. Data Science/Kaggle/test.csv")
weather <- read.csv("/Volumes/Prasant3TB/04. Academic/02. Data Science/Kaggle/weather.csv")
spray <- read.csv("/Volumes/Prasant3TB/04. Academic/02. Data Science/Kaggle/spray.csv")
sampleSubmission <- read.csv("/Volumes/Prasant3TB/04. Academic/02. Data Science/Kaggle/sampleSubmission.csv")

## Explore Data ####
head(sampleSubmission)
head(train)
ncol(train)
names(train)
names(test)
levels(train$Species)
summary(train)
summary(test)

## Load Packages ####
library(data.table)
library(rpart)
library(randomForest)
library(ranger)
library(party)
library(mefa)
library(caret)
library(pROC)
library(e1071)
library(neuralnet)


## Processing Weather Data ####
# http://www.accuweather.com/en/weather-news/bite-me-weathers-impact-on-mos/34268
weather[(weather == " ")] <- NA
weather[(weather == "M")] <- NA
weather[(weather == "-")] <- NA
weather[(weather == "T")] <- NA
weather[(weather == " T")] <- NA
weather[(weather == "  T")] <- NA

weather$Water1 = NULL
weather$Depth = NULL
weather$SnowFall = NULL
weather$Sunrise = NULL
weather$Sunset = NULL
weather$Depart = NULL

# https://stackoverflow.com/questions/23615063/calculating-great-circle-distance-matrix
# https://en.wikipedia.org/wiki/Haversine_formula

# Mapping all weather stations ####

train$Station <- ifelse((((train$Latitude-41.995)^2 + (train$Longitude + 87.933)^2) < 
                           ((train$Latitude-41.786)^2 + (train$Longitude + 87.752)^2)),1,2)

test$Station <- ifelse((((test$Latitude-41.995)^2 + (test$Longitude + 87.933)^2) < 
                          ((test$Latitude-41.786)^2 + (test$Longitude + 87.752)^2)),1,2)

w1 = weather[weather$Station ==1,]
w2 = weather[weather$Station ==2,]

W1 <- rbind(w1[2,],w1)
W1 <- fill.na(W1) 
W1 <- W1[-1,]
rownames(W1) <- NULL

W2 <- rbind(w2[2,],w2)
W2 <- fill.na(W2) 
W2 <- W2[-1,]
rownames(W2) <- NULL

Weather <- rbind(W1,W2)

for(i in c(3:9,11:16)){
  Weather[,i] <- as.numeric(Weather[,i])
}
Weather[,10] <- factor(Weather[,10])

## Merge training and testing datasets with Weather data ####
train <- merge.data.frame(train,Weather)
test <- merge.data.frame(test,Weather)

SpeciesX<-c(as.character(train$Species),as.character(test$Species))
SpeciesX[SpeciesX=="UNSPECIFIED CULEX"]<-"CULEX ERRATICUS"
Species<-factor(SpeciesX,levels=unique(SpeciesX))

train = data.table(train)
test = data.table(test)
train[,Species:=factor(SpeciesX[1:nrow(train)],levels=unique(SpeciesX))]
test[,Species:=factor(SpeciesX[(nrow(train)+1):length(SpeciesX)],levels=unique(SpeciesX))]


summary(train)
nrow(train) #10506
nrow(test) #116293

## Create NA Colums in test dataset ####
test$WnvPresent <- NA
test$NumMosquitos <- NA
test$Id <- NULL

## Merge Training and Testing datasets to normalise levels ####
merged_list <- rbind(train,test)
head(merged_list)
merged_list$Date <- as.factor(merged_list$Date)
merged_list$Station <- as.factor(merged_list$Station)
merged_list$Address <- as.factor(merged_list$Address)
merged_list$Species <- as.factor(merged_list$Species)
merged_list$Street <- as.factor(merged_list$Street)
merged_list$Trap <- as.factor(merged_list$Trap)
merged_list$AddressNumberAndStreet <- as.factor(merged_list$AddressNumberAndStreet)
merged_list$CodeSum <- as.factor(merged_list$CodeSum)
nrow(merged_list)

## Re-split into training and testing data sets ####
train.new <- merged_list[1:10506,]
test.new <- merged_list[10507:126799,]

summary(train.new)
summary(test.new)


# Begin Modelling ####
set.seed(1337)
fit <- randomForest(as.factor(WnvPresent) ~ Station + Species + Tmax + Tmin + Tavg + DewPoint +
                      WetBulb + Cool + PrecipTotal + AvgSpeed,
                    data=train.new,
                    importance = TRUE)
fit$confusion
fit$ntree
fit$mtry
fit
varImp(fit, scale = TRUE)
varImpPlot(fit,type=2) #mean decrease in node impurity 

rmpred <- predict(fit, newdata = test.new)
rmpred
summary(rmpred)
confusionMatrix(fit, test.new$WnvPresent)

# ctrl <- trainControl(method = "repeatedcv", repeats = 10, summaryFunction = twoClassSummary)
# fit2 <- train(as.factor(WnvPresent) ~ Station + Species + Tmax + Tmin + Tavg + DewPoint +
#                 WetBulb + Cool + PrecipTotal + AvgSpeed,
#               data = train.new,
#               preProc = c("center", "scale"),
#               trControl = ctrl,
#               tuneLength = 15,
#               method = 'cforest')
# fit2


# ctrl <- trainControl(method = 'repeatedcv', number = 10)
# nsv <- nearZeroVar(train.new, saveMetrics = TRUE)
# nsv
# rmfit <- train(as.factor(WnvPresent) ~ Station + Species + Tmax + Tmin + Tavg + DewPoint + 
#                   WetBulb + Cool + PrecipTotal + AvgSpeed,
#                 data = train.new,
#                 method = 'ranger',
#                trControl = ctrl)
# var_imp <- varImp(rmfit, scale = TRUE)
# var_imp 



# result.roc <- roc(test.new$WnvPresent,) # Draw ROC curve.
# plot(result.roc, print.thres="best", print.thres.best.method="closest.topleft")

# result.coords <- coords(result.roc, "best", best.method="closest.topleft", ret=c("threshold", "accuracy"))
# print(result.coords)#to get threshold and accuracy
# 



ctrl <- trainControl(method = 'repeatedcv', classProbs = TRUE, number = 10)
gamboost <- train(WnvPresent ~ Station + Species + Tmax + Tmin + Tavg + DewPoint + 
                    WetBulb + Cool + PrecipTotal + AvgSpeed,
                  data = train.new,
                  method ='gamboost',
                  trControl = ctrl)

glmfit <-glm(WnvPresent ~ Station + Species + Tmax + Tmin + Tavg + DewPoint +
               WetBulb + Cool + PrecipTotal + AvgSpeed,
             data=train.new, 
             family = "binomial")

glmfit

glmsubmit<-predict(glmfit, newdata = test.new, type = "response")
summary(glmsubmit)
# Look at variable importance
varImpPlot(rmfit)






summary(merged_list)
neur_net <-neuralnet(WnvPresent ~ NumMosquitos + Tmax + Tmin + Tavg + DewPoint + 
               WetBulb + Cool + PrecipTotal + AvgSpeed,
             data = train.new,
             stepmax = 1e+05, 
             rep = 10,
             hidden=10, threshold=0.01,
             algorithm = "rprop+")
neur_net

neurpred <- compute(neur_net, test.new)


prediction <- predict(rmfit, test.new)
summary(prediction)

## Create Output File ####
submit <- data.frame(Id = seq.int(nrow(sampleSubmission)), WnvPresent = prediction)
write.csv(submit, file = "Submit-gbm1.csv", row.names = FALSE)
