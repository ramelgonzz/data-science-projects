#modified from https://www.kaggle.com/code/jagofc/tps-may-2022-lgbm-baseline-in-r
#Install and load libraries
pacman::p_load(tidyverse, tidymodels, lightgbm)

#Load training Data
train <- read_csv('train.csv', show_col_types=F)

alphabet <- c("A","B","C","D","E","F","G","H","I","J",
           "K","L","M","N","O","P","Q","R","S","T",
           "U","V","W","X","Y","Z")

alpha <- "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

f_27_names <- sapply(0:10, function(i) {paste0("l_", i)})

featurise <- function(df, remove=T) {
    df %>% 
        separate(f_27, sep="", into=f_27_names, remove=remove) %>%
        select(-l_0) %>%
        rowwise %>%
        mutate(across(contains("l_"), 
                      function(col) which(alphabet == col)[[1]])) %>%
        ungroup()
}

data_train <- featurise()(train, remove=T)
split <- data_train %>% initial_split(prop=4/5)
data_train <- training(split)
data_valid <- testing(split)

#use recipe for preprocessing data
rec <- recipe(target ~ ., data=data_train) %>%
    update_role(id, new_role="id")

rec <- rec %>% 
    step_center(all_numeric_predictors()) %>%
    step_scale(all_numeric_predictors())

trained_rec <- prep(rec, training=data_train)

data_train <- bake(trained_rec, new_data=data_train)# %>% slice_sample(prop=0.5))
data_valid <- bake(trained_rec, new_data=data_valid)# %>% slice_sample(prop=0.5))

#training using lightgbm

feat_cols <- data_train %>% select(-id, -target) %>% colnames
feat_cols

ltrain <- lgb.Dataset(
    data = data_train %>% select(all_of(feat_cols)) %>% as.matrix(),
    label = data_train$target
)

lvalid <- lgb.Dataset(
    data = data_valid %>% select(all_of(feat_cols)) %>% as.matrix(),
    label = data_valid$target
)
lgb.params = list(
    objective="binary",
    learning_rate = 0.08
)

train_an_lgb <- function() {
    lgb.train(params=lgb.params,
              data=ltrain,
              nrounds = 5000,
              maximize=F,
              verbose=1,
              valids=list("lvalid"=lvalid),
              early_stopping_rounds = 10)
}
lgb_model <- train_an_lgb()

prob_to_pred <- function(prob) as.integer(prob > 0.5)
valid_pred_lgb <- predict(lgb_model, data_valid %>% select(all_of(feat_cols)) %>% as.matrix())

data_valid %>%
    mutate(prob = valid_pred_lgb) %>%
    mutate(pred = factor(prob_to_pred(prob), levels = c(1, 0))) %>%
    mutate(target = factor(target, levels = c(1, 0))) %>%
    metrics(truth=target, estimate=pred, prob)

#read test data

test <- read_csv('test.csv',show_col_types=F)

test_data <- test %>%
    featurise() %>%
    bake(trained_rec, new_data=.)

test_prob <- predict(lgb_model, test_data %>% select(all_of(feat_cols)) %>% as.matrix())

test_data <- test_data %>%
    mutate(target = test_prob) %>%
    mutate(pred = prob_to_pred(target))