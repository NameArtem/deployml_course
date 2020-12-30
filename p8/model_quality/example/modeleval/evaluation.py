import numpy as np
import pandas as pd
import os
from IPython.display import display, HTML
import matplotlib.pyplot as plt
plt.style.use("ggplot")
from sklearn.metrics import *

class BaseEvaluator(object):
    """
    Базовый оценщик 
    Для всех основных моделей: sklearn, *boost
    """
    def __init__(self):
        pass
    
    
    def ensure_path(self, file_path):
        directory = os.path.dirname(file_path)
        if not os.path.exists(directory):
            os.makedirs(directory)


class RegressionEvaluator(BaseEvaluator):
    """
    Оценщик регрессии
    """
    def __init__(self):
        super(RegressionEvaluator, self).__init__()
    
    
    def get_rmsle(self, predicted, real):
        """Calculate Root Mean Squared Logarithmic Error (RMSLE)
        
        Parameters
        ----------
        predicted : ndarray
	    The predicted response.

        real : ndarray
            The real response.
        
        Returns
        -------
        rmsle : float 
        """
        from sklearn.metrics import mean_squared_log_error
        
        return np.sqrt(mean_squared_log_error(np.absolute(real), np.absolute(predicted)))
        
        #return (sum/len(predicted))**0.5
    
    
    def get_metrics(self, model, eval_X, eval_y):
        """Calculate common metrics for regression problem.
        
        Parameters
        ----------
        model : sklearn/lightgbm/xgboost/catboost regression model object
            The model for evaluation.

        eval_X : ndarray or pd.DataFrame
            The test data's features.

        eval_y : ndarray
            The test data's response.
        """
        self.model = model
        self.eval_y = eval_y
        if isinstance(eval_X, pd.DataFrame):
            eval_X = eval_X.values
        
        self.eval_X = eval_X
        self.y_pred = model.predict(self.eval_X)
        self.res = self.y_pred - self.eval_y
        # metrics
        self.mse = mean_squared_error(eval_y, self.y_pred)
        self.mae = mean_absolute_error(eval_y, self.y_pred)
        self.rmse = np.sqrt(self.mse)
        self.rmsle = self.get_rmsle(self.y_pred, eval_y)
        self.r2 = r2_score(eval_y, self.y_pred)
    
    
    def res_fit_plot(self):
        """Residual vs Fitted value plot"""
        plt.scatter(self.y_pred, self.res, c='red')
        plt.title("Residuals vs Fitted Values plot")
        plt.xlabel("Fitted Values")
        plt.ylabel("Residuals")
        return plt
    
    def evaluate(self, model, eval_X, eval_y, plot=True, save=False, save_folder="result"):
        """Make prediction and generate evaluation report.

        Parameters
        ----------
        model : sklearn/lightgbm/xgboost/catboost regression model object
            The model for evaluation.

        eval_X : ndarray or pd.DataFrame
            The test data's features.

        eval_y : ndarray
            The test data's response.

        plot : bool (default=True)
            If "Flase", return only common metrics. If "True", return the 
            residual vs fitted values plot as well.

        save : bool (default=False)
            Save the result to path if True.

        save_folder : string (default="result")
            The folder path to save the result, default is the result folder in the current directory.
        """
        self.get_metrics(model, eval_X, eval_y)
        print("Evaluation Result")
        print("---Common Metrics---")
        print("The mse is %0.4f" % self.mse)
        print("The mae is %0.4f" % self.mae)
        print("The rmse is %0.4f" % self.rmse)
        print("The rmsle is %0.4f" % self.rmsle)
        print("The r-square is %0.4f" % self.r2)
        if plot:
            fig = self.res_fit_plot()
        if save:
            super(RegressionEvaluator, self).ensure_path(save_folder)
            plot_path = save_folder + 'residual_vs_fittedval.png'
            if fig:
                fig.savefig(
                    plot_path,
                    bbox_inches='tight'
                )
            #Write result into a txt
            output = '\n'.join([
                '--Model Evaluation--',
                '\tmse: {mse:.4f}',
                '\tmae: {mae:.4f}',
                '\trmse: {rmse:.4f}',
                '\trmsle: {rmsle:.4f}',
                '\tr-square: {r2:.4f}',
                '\n'
            ]).format(
                mse = self.mse,
                mae = self.mae,
                rmse = self.rmse,
                rmsle = self.rmsle,
                r2 = self.r2
            )
            result_path = save_folder + 'reg_output.txt'
            with open(result_path, 'w+') as f:
                f.write(output)
            
    
    def find_best_model(self, models, eval_X=None, eval_y=None, objective="mse"):
        """Find the best model with the specified objective metric.

        Parameters
        ----------
        models : list 
            List of sklearn/lightgbm/xgboost/catboost regression model.
            
        eval_X : ndarray or pd.DataFrame
            The test data's features.

        eval_y : ndarray
            The test data's response.
        
        objective: string (default="mse")
            The objective metric.
            
        Returns
        -------
        A single model object.       
        """
        result = np.array([])
        for model in models:
            y_pred = model.predict(eval_X)
            if objective == "mse":
                result = np.append(result, mean_squared_error(eval_y, y_pred))
            if objective == "mae":
                result = np.append(result, mean_absolute_error(eval_y, y_pred))
            if objective == "rmse":
                result = np.append(result, np.sqrt(mean_squared_error(eval_y, y_pred)))
            if objective == "rmsle":
                result = np.append(result, self.rmsle(y_pred, eval_y))
            if objective == "r2":
                result = np.append(result, r2_score(eval_y, y_pred))
        if objective == "r2":
            print("The model with maximum r-square (%s) is the %s th model" % (result.max(), result.argmax()+1))
            return models[result.argmax()]
        else:
            print("The model with minimum %s (%s) is the %s th model" % (objective, result.min(), result.argmin()+1))
            return models[result.argmin()]

class BinaryEvaluator(BaseEvaluator):
    """
    Оценщик для бинарной классификации
    """
    def __init__(self):
        super(BinaryEvaluator, self).__init__()
        
        
    def get_metrics(self, model, eval_X, eval_y, threshold=0.5):
        """Calculate common metrics for binary classification problem"
        
        Parameters
        ----------
        model : sklearn/lightgbm/xgboost/catboost classification model object
            The model for evaluation.

        eval_X : ndarray or pd.DataFrame
            The test data's features.

        eval_y : ndarray
            The test data's labels.
        
        threshold : int or float 
            The threshold to determine the predicted label
        """
        self.model = model
        self.eval_y = eval_y
        if isinstance(eval_X, pd.DataFrame):
            eval_X = eval_X.values
            #X_cols = eval_X.columns
        self.eval_X = eval_X
        y_probs = model.predict_proba(eval_X)[:, 1]
        self.y_probs = y_probs
        if threshold == 0.5:
            pred = model.predict(eval_X)
        else:
            pred = np.where(y_probs>threshold, 1, 0)       
        accuracy = accuracy_score(pred, eval_y)
        recall_1 = recall_score(eval_y, pred, pos_label=1)
        precision_1 = precision_score(eval_y, pred, pos_label=1)
        recall_0 = recall_score(eval_y, pred, pos_label=0)
        precision_0 = precision_score(eval_y, pred, pos_label=0)
        f1 = f1_score(eval_y, pred, pos_label=1)
        roc_auc = roc_auc_score(eval_y, y_probs)
        y_true = pd.Series(eval_y)
        y_pred = pd.Series(pred)
        confusion = pd.crosstab(y_true, y_pred, rownames=['True'], colnames=['Predicted'], margins=True)
        return accuracy, recall_1, precision_1, recall_0, precision_0, f1, roc_auc, confusion
        
        
    def evaluate(self, model, eval_X, eval_y, threshold=0.5, plot=True, save=False, save_folder="result"):
        """Make prediction and generate evaluation report based on specified threshold.

        Parameters
        ----------
        model : sklearn/lightgbm/xgboost/catboost classification model object
            The model for evaluation.

        eval_X : ndarray or pd.DataFrame
            The test data's features.

        eval_y : ndarray
            The test data's labels.

        plot : bool (default=True)
            If "Flase", return only common metrics. If "True", return the plots including
            ROC curve, Precision_Recall vs threshold, class probability distribution and
            feature importance as well.

        save : bool, optional (default=False)
            Whether to save the result or not.

        save_folder : string (default="result")
            The folder path to save the result, default is the result folder in the current directory.
        """
        accuracy, recall_1, precision_1, recall_0, precision_0, f1, roc_auc, confusion = self.get_metrics(model, eval_X, eval_y, threshold)
        print("Evaluation result of Threshold=={thres}".format(thres=threshold))
        print("---Common Metrics---")
        print("The accuracy is %0.4f" % accuracy)
        print("The recall for 1 is %0.4f" % recall_1)
        print("The precision for 1 is %0.4f" % precision_1)
        print("The recall for 0 is %0.4f" % recall_0)
        print("The precision for 0 is %0.4f" % precision_0)
        print("The F1-score is %0.4f" % f1)
        print("The ROC-AUC is %0.4f" % roc_auc)
        print("\n---Confusion Matrix---")
        print(confusion)
        if plot:
            # AUC
            fpr, tpr, auc_thresholds = roc_curve(eval_y, self.y_probs)
            precisions, recalls, thresholds = precision_recall_curve(eval_y, self.y_probs)

            # ROC
            fig = plt.figure(figsize=(10, 27))
            plt.subplots_adjust(hspace=0.25)
            ax1 = fig.add_subplot(411)
            ax1.set_title('ROC Curve')
            ax1.plot(fpr, tpr, linewidth=2)
            ax1.plot([0, 1], [0, 1], 'k--')
            ax1.axis([-0.005, 1, 0, 1.005])
            ax1.set_xticks(np.arange(0, 1, 0.05))
            ax1.set_xlabel('False Positive Rate')
            ax1.set_ylabel('True Positive Rate (Recall)')

            # Recall_Precision VS Decision Threshold Plot
            ax2 = fig.add_subplot(412)
            ax2.set_title('Precision and Recall vs Decision Threshold')
            ax2.plot(thresholds, precisions[:-1], 'b--', label='Precision')
            ax2.plot(thresholds, recalls[:-1], 'g-', label='Recall')
            ax2.set_ylabel('Score')
            ax2.set_xlabel('Decision Threshold')
            ax2.legend(loc='best')

            # Class Probability Distribution
            ax3 = fig.add_subplot(413)
            ax3.set_title('Class Probability Distribution')
            ax3.set_ylabel('Density')
            ax3.set_xlabel('Predicted Probability')
            ax3.hist(self.y_probs[eval_y == 1], bins=40,
                           density=True, alpha=0.5)
            ax3.hist(self.y_probs[eval_y == 0], bins=40,
                              density=True, alpha=0.5)
            
            # Feature importance
            model_list = ["DecisionTree","RandomForest", "XGB", "LGBM"]
            if any(mod_name in str(type(model)) for mod_name in model_list): 
                ax4 = fig.add_subplot(414)
                ax4.set_title('Feature Importance')
                feature_importance = model.feature_importances_
                try:
                    X_cols
                except:
                    pd.Series(feature_importance, index=range(eval_X.shape[1])).nlargest(eval_X.shape[1]).plot(kind='barh')
                    ax4.set_ylabel('Column Index')
                else:
                    pd.Series(feature_importance, index=X_cols).nlargest(eval_X.shape[1]).plot(kind='barh', color = range(eval_X.shape[1]))
        if save:
            super(BinaryEvaluator, self).ensure_path(save_folder)
            plot_path = save_folder + 'multiple_metrics_plots.png'
            if fig:
                fig.savefig(
                    plot_path,
                    bbox_inches='tight'
                )
            #Write result into a txt
            output = '\n'.join([
                '--Model Evaluation--',
                '\tAccuracy: {accuracy:.4f}',
                '\tRecall for 1: {recall_1:.4f}',
                '\tPrecision for 1: {precision_1:.4f}',
                '\tRecall for 0: {recall_0:.4f}',
                '\tPrecision for 0: {precision_0:.4f}',
                '\tF1 score: {f1:.4f}',
                '\tROC-AUC: {roc_auc:.4f}',
                '\n',
                '--Confusion Matrix--',
                '{confusion}'
            ]).format(
                accuracy = accuracy,
                recall_1 = recall_1,
                precision_1 = precision_1,
                recall_0 = recall_0,
                precision_0 = precision_0,
                f1 = f1,
                roc_auc = roc_auc,
                confusion = confusion
            )
            result_path = save_folder + 'clf_output.txt'
            with open(result_path, 'w+') as f:
                f.write(output)
            

    def ThresGridSearch(self, model, eval_X, eval_y, thres_list=None, objective=None):
        """Show the result of common metrics of given thresholds grid.

        Parameters
        ----------
        thres_list : list (default="None")
            If no threshold list is given, it uses default grid:
            [0.3, 0.4, 0.5, 0.6, 0.7], else, uses the given threshold
            grid list.
 
        objective : list (default=None)
            Sort the result by given metrics result.
            Possible choices are 'accuracy', 'recall_1',
            precision_1', 'recall_0', 'precision_0', 'f1'.
        """
        accuracy_list = []
        recall_1_list = []
        precision_1_list = []
        recall_0_list = []
        precision_0_list = []
        f1_list = []
        roc_auc_list = []
        confusion_list = []
        if thres_list == None:
            thres_list = np.arange(0.3, 0.7, 0.1)
        for thres in thres_list:
            accuracy, recall_1, precision_1, recall_0, precision_0, f1, roc_auc, confusion = self.get_metrics(model, eval_X, eval_y, thres)
            accuracy_list.append(accuracy)
            recall_1_list.append(recall_1)
            precision_1_list.append(precision_1)
            recall_0_list.append(recall_0)
            precision_0_list.append(precision_0)
            f1_list.append(f1)
            roc_auc_list.append(roc_auc)
            confusion_list.append(confusion)
        data = np.array([thres_list, accuracy_list, recall_1_list, precision_1_list, recall_0_list, precision_0_list, f1_list, roc_auc_list]).T
        self.df = pd.DataFrame(data, columns=["Threshold", "accuracy", "recall_1", "precision_1", "recall_0", "precision_0", "f1","roc_auc"])
        if objective:
            self.df = self.df.sort_values(by=objective)
        display(self.df)
    
    
    def find_best_model(self, models, eval_X=None, eval_y=None, objective="accuracy", threshold=0.5):
        """Find the best model with the specified objective metrics on given 
        threshold.

        Parameters
        ----------
        model : sklearn/lightgbm/xgboost/catboost classification model object
            The model for evaluation.

        eval_X : ndarray or pd.DataFrame
            The test data's features.

        eval_y : ndarray
            The test data's labels.
            
        objective: string (default="accuracy")
            The search goal     
            
        threshold: float (default=0.5)
        
        Returns
        -------
        A single model object.
        """
        result = np.array([])
        for model in models:
            accuracy, recall_1, precision_1, recall_0, precision_0, f1, _, _ = self.get_metrics(model, eval_X, eval_y, threshold)
            if objective == "accuracy":
                result = np.append(result, accuracy)
            if objective == "recall_1":
                result = np.append(result, recall_1)
            if objective == "precision_1":
                result = np.append(result, precision_1)
            if objective == "recall_0":
                result = np.append(result, recall_0)
            if objective == "precision_0":
                result = np.append(result, precision_0)
            if objective == "f1":
                result = np.append(result, f1)
        print("The model with maximum %s (%s) is the %s th model" % (objective, result[result.argmax()],result.argmax()+1))
        return models[result.argmax()]