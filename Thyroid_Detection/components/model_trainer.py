from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, recall_score
from Thyroid_Detection.entity import artifact_entity, config_entity
from Thyroid_Detection.exception import ThyroidException
from Thyroid_Detection.logger import logging
from typing import Optional
import os, sys
from Thyroid_Detection import utils


class ModelTrainer:

    def __init__(self, model_trainer_config: config_entity.ModelTrainerConfig,
                 data_transformation_artifact: artifact_entity.DataTransformationArtifact):
        try:
            logging.info(f"{'>>'*20} Model Trainer {'<<'*20}")
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact

        except Exception as e:
            raise ThyroidException(e, sys)

    def train_model(self, x, y):
        try:
            # Define the parameter grid for hyperparameter tuning
            param_grid = {
                'criterion': ['gini', 'entropy'],
                'max_depth': [None, 10, 20, 30, 40, 50],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['auto', 'sqrt', 'log2']
            }

            # Initialize the Decision Tree Classifier
            clf = DecisionTreeClassifier()

            # Initialize Grid Search Cross-Validation
            grid_search = GridSearchCV(clf, param_grid, cv=5, scoring='recall_macro', verbose=1)

            # Train the model using Grid Search CV
            grid_search.fit(x, y)

            # Get the best model from Grid Search CV
            best_model = grid_search.best_estimator_

            return best_model
        except Exception as e:
            raise ThyroidException(e, sys)

    def initiate_model_training(self) -> artifact_entity.ModelTrainerArtifact:
        try:
            logging.info("Loading train and test numpy array for model training")
            train_arr = utils.load_num_array(file_path=self.data_transformation_artifact.transformed_train_path)
            test_arr = utils.load_num_array(file_path=self.data_transformation_artifact.transformed_test_path)

            logging.info(f"Splitting input and target feature from both train and test arrays")
            x_train, y_train = train_arr[:, :-1], train_arr[:, -1]
            x_test, y_test = test_arr[:, :-1], test_arr[:, -1]

            logging.info(f"Train the model using Grid Search Cross-Validation for hyperparameter tuning")
            model = self.train_model(x=x_train, y=y_train)

            logging.info(f"Calculating f1 train score")
            yhat_train = model.predict(x_train)
            f1_train_score = f1_score(y_true=y_train, y_pred=yhat_train, average='macro')

            logging.info(f"Calculating recall train score")
            recall_train_score = recall_score(y_true=y_train, y_pred=yhat_train, average='macro')

            logging.info(f"Calculating f1 test score")
            yhat_test = model.predict(x_test)
            f1_test_score = f1_score(y_true=y_test, y_pred=yhat_test, average='macro')

            logging.info(f"Calculating recall test score")
            recall_test_score = recall_score(y_true=y_test, y_pred=yhat_test, average='macro')

            logging.info(f"Train score: {recall_train_score} and test score: {recall_test_score}")

            # Check for overfitting or underfitting or expected score
            logging.info(f"Checking if our model is underfitting or not")
            if recall_test_score < self.model_trainer_config.expected_score:
                raise Exception(f"Model is not good as it is not able to give "
                                f"expected accuracy: {self.model_trainer_config.expected_score}: "
                                f"model actual score: {recall_test_score}")

            logging.info(f"Checking if our model is overfitting or not")
            diff = abs(recall_train_score - recall_test_score)
            if diff > self.model_trainer_config.overfitting_threshold:
                raise Exception(f"Train and test score diff: {diff} is more than overfitting threshold "
                                f"{self.model_trainer_config.overfitting_threshold}")

            # Save the trained model
            logging.info(f"Saving model object")
            utils.save_object(file_path=self.model_trainer_config.model_path, obj=model)

            # Prepare artifact
            logging.info(f"Prepare the artifact")
            model_trainer_artifact = artifact_entity.ModelTrainerArtifact(
                model_path=self.model_trainer_config.model_path,
                recall_train_score=recall_train_score,
                recall_test_score=recall_test_score,
                f1_train_score=f1_train_score,
                f1_test_score=f1_test_score)
            logging.info(f"Model trainer artifact: {model_trainer_artifact}")
            return model_trainer_artifact
        except Exception as e:
            raise ThyroidException(e, sys)
