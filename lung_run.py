import warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import confusion_matrix
demo = pd.read_csv(r'C:\Users\saich\OneDrive\Desktop\lungcancerfile.csv')
y = demo['LUNG_CANCER']
X = demo.drop(['LUNG_CANCER'], axis = 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 3)
from sklearn.tree import DecisionTreeClassifier

plt.rcParams['font.size'] = 20
decision_tree = DecisionTreeClassifier(random_state=5)

# Train the classifier
decision_tree.fit(X_train, y_train)

# Predict the target variable on the test set
y_pred = decision_tree.predict(X_test)

# Evaluate the classifier
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy with decision tree:", accuracy)

accuracy_train = accuracy_score(y_train, decision_tree.predict(X_train))
print("Training Accuracy for decision tree:", accuracy_train)


# Precision
precision = precision_score(y_test, y_pred)
print("Precision for decision tree:", precision)

# Recall
recall = recall_score(y_test, y_pred)
print("Recall for decision tree:", recall)

# F1 Score
f1 = f1_score(y_test, y_pred)
print("F1 Score for decision tree:", f1)

# F2 Score (You can adjust beta for different emphasis on precision or recall)
beta = 2
f2 = (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall)
print("F2 Score for decision tree:", f2)

# Predict the probabilities of the positive class
y_probs = decision_tree.predict_proba(X_test)[:, 1]

roc_auc = roc_auc_score(y_test, y_probs)
print("ROC AUC Score for DT:", roc_auc)

# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_probs)

# Plot the ROC curve
plt.figure(figsize=(8,6))
plt.xlabel('False Positive Rate', fontsize=25)
plt.ylabel('True Positive Rate', fontsize=25)
plt.title('ROC Curve of DT')
plt.legend(loc="lower right", fontsize=25)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0]) 
plt.ylim([0.0, 1.05])
plt.legend(loc="lower right", fontsize="20")
plt.plot(fpr, tpr, color='green',label='ROC Curve (area = %0.2f)' % roc_auc)
plt.legend(loc="lower right", fontsize="20")
plt.show()
















import matplotlib.pyplot as plt
import numpy
from sklearn import metrics
plt.rcParams['figure.dpi'] = 300
actual = numpy.random.binomial(1,.9,size = 1000)
predicted = numpy.random.binomial(1,.9,size = 1000)

confusion_matrix = metrics.confusion_matrix(y_test, y_pred)

cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])
cm_display.plot()
plt.xlabel("Predicted labels", fontsize=25)
plt.ylabel("True labels", fontsize=25)
plt.title("DT", fontsize=25)
plt.show()


test_accuracy = accuracy_score(y_test, y_pred)
test_error = 1 - test_accuracy

print("Testing Error for Decision Tree:", test_error)





















from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(random_state=5)

clf.fit(X_train, y_train)

# Predict on the testing set
y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy with Random Forest:", accuracy)

accuracy_train = accuracy_score(y_train, clf.predict(X_train))
print("Training Accuracy for Random Forest:", accuracy_train)


# Precision
precision = precision_score(y_test, y_pred)
print("Precision for Random Forest:", precision)

# Recall
recall = recall_score(y_test, y_pred)
print("Recall for Random Forest:", recall)

# F1 Score
f1 = f1_score(y_test, y_pred)
print("F1 Score for Random Forest:", f1)

# F2 Score (You can adjust beta for different emphasis on precision or recall)
beta = 2
f2 = (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall)
print("F2 Score for Random Forest:", f2)

# Predict the probabilities of the positive class
y_probs = clf.predict_proba(X_test)[:, 1]

# ROC AUC Score
roc_auc = roc_auc_score(y_test, y_probs)
print("ROC AUC Score for Random Forest:", roc_auc)

# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_probs)

# Plot the ROC curve
plt.figure(figsize=(8,6))
plt.xlabel('False Positive Rate', fontsize=25)
plt.ylabel('True Positive Rate', fontsize=25)
plt.title('ROC Curve of RF')
plt.legend(loc="lower right", fontsize=25)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0]) 
plt.ylim([0.0, 1.05])
plt.legend(loc="lower right", fontsize="20")
plt.plot(fpr, tpr, color='green',label='ROC Curve (area = %0.2f)' % roc_auc)
plt.legend(loc="lower right", fontsize="20")
plt.show()


actual = numpy.random.binomial(1,.9,size = 1000)
predicted = numpy.random.binomial(1,.9,size = 1000)

confusion_matrix = metrics.confusion_matrix(y_test, y_pred)

cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])
cm_display.plot()
plt.xlabel("Predicted labels", fontsize=25)
plt.ylabel("True labels", fontsize=25)
plt.title("RF", fontsize=25)
plt.show()


test_accuracy = accuracy_score(y_test, y_pred)
test_error = 1 - test_accuracy

print("Testing Error for Random Forest:", test_error)





from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(random_state=5)
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy with LogisticRegression:", accuracy)

accuracy_train = accuracy_score(y_train,lr.predict(X_train))
print("Training Accuracy for LogisticRegression:", accuracy_train)


# Precision
precision = precision_score(y_test, y_pred)
print("Precision for LogisticRegression:", precision)

# Recall
recall = recall_score(y_test, y_pred)
print("Recall for LogisticRegression:", recall)

# F1 Score
f1 = f1_score(y_test, y_pred)
print("F1 Score for LogisticRegression:", f1)

# F2 Score (You can adjust beta for different emphasis on precision or recall)
beta = 2
f2 = (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall)
print("F2 Score for LogisticRegression:", f2)

# Predict the probabilities of the positive class
y_probs = lr.predict_proba(X_test)[:, 1]

# ROC AUC Score
roc_auc = roc_auc_score(y_test, y_probs)
print("ROC AUC Score for LogisticRegression:", roc_auc)

# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_probs)

# Plot the ROC curve
plt.figure(figsize=(8,6))
plt.xlabel('False Positive Rate', fontsize=25)
plt.ylabel('True Positive Rate', fontsize=25)
plt.title('ROC Curve of LR')
plt.legend(loc="lower right", fontsize=25)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0]) 
plt.ylim([0.0, 1.05])
plt.legend(loc="lower right", fontsize="20")
plt.plot(fpr, tpr, color='green',label='ROC Curve (area = %0.2f)' % roc_auc)
plt.legend(loc="lower right", fontsize="20")
plt.show()


actual = numpy.random.binomial(1,.9,size = 1000)
predicted = numpy.random.binomial(1,.9,size = 1000)

confusion_matrix = metrics.confusion_matrix(y_test, y_pred)

cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])
cm_display.plot()
plt.xlabel("Predicted labels", fontsize=25)
plt.ylabel("True labels", fontsize=25)
plt.title("LR", fontsize=25)
plt.show()


test_accuracy = accuracy_score(y_test, y_pred)
test_error = 1 - test_accuracy

print("Testing Error for LogisticRegression:", test_error)












from sklearn.svm import SVC

clf = SVC(random_state=1,probability=True)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy with SVM:", accuracy)

accuracy_train = accuracy_score(y_train,clf.predict(X_train))
print("Training Accuracy for SVM:", accuracy_train)


# Precision
precision = precision_score(y_test, y_pred)
print("Precision for SVM:", precision)

# Recall
recall = recall_score(y_test, y_pred)
print("Recall for SVM:", recall)

# F1 Score
f1 = f1_score(y_test, y_pred)
print("F1 Score for SVM:", f1)

# F2 Score (You can adjust beta for different emphasis on precision or recall)
beta = 2
f2 = (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall)
print("F2 Score for SVM:", f2)

# Predict the probabilities of the positive class
y_probs = clf.predict_proba(X_test)[:, 1]

# ROC AUC Score
roc_auc = roc_auc_score(y_test, y_probs)
print("ROC AUC Score for SVM:", roc_auc)

# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_probs)

# Plot the ROC curve
plt.figure(figsize=(8,6))
plt.xlabel('False Positive Rate', fontsize=25)
plt.ylabel('True Positive Rate', fontsize=25)
plt.title('ROC Curve of SVM')
plt.legend(loc="lower right", fontsize=25)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0]) 
plt.ylim([0.0, 1.05])
plt.legend(loc="lower right", fontsize="20")
plt.plot(fpr, tpr, color='green',label='ROC Curve (area = %0.2f)' % roc_auc)
plt.legend(loc="lower right", fontsize="20")
plt.show()

actual = numpy.random.binomial(1,.9,size = 1000)
predicted = numpy.random.binomial(1,.9,size = 1000)

confusion_matrix = metrics.confusion_matrix(y_test, y_pred)

cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])
cm_display.plot()
plt.xlabel("Predicted labels", fontsize=25)
plt.ylabel("True labels", fontsize=25)
plt.title("SVM", fontsize=25)
plt.show()


test_accuracy = accuracy_score(y_test, y_pred)
test_error = 1 - test_accuracy

print("Testing Error for SVM:", test_error)





from sklearn.neighbors import KNeighborsClassifier

clf = KNeighborsClassifier(n_neighbors=3)
clf.fit(X_train, y_train)

# Predict on the testing set
y_pred = clf.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy with KNN:", accuracy)

accuracy_train = accuracy_score(y_train,clf.predict(X_train))
print("Training Accuracy for KNN:", accuracy_train)


# Precision
precision = precision_score(y_test, y_pred)
print("Precision for KNN:", precision)

# Recall
recall = recall_score(y_test, y_pred)
print("Recall for KNN:", recall)

# F1 Score
f1 = f1_score(y_test, y_pred)
print("F1 Score for KNN:", f1)

# F2 Score (You can adjust beta for different emphasis on precision or recall)
beta = 2
f2 = (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall)
print("F2 Score for KNN:", f2)

# Predict the probabilities of the positive class
y_probs = clf.predict_proba(X_test)[:, 1]

# ROC AUC Score
roc_auc = roc_auc_score(y_test, y_probs)
print("ROC AUC Score for KNN:", roc_auc)

# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_probs)

# Plot the ROC curve
plt.figure(figsize=(8,6))
plt.xlabel('False Positive Rate', fontsize=25)
plt.ylabel('True Positive Rate', fontsize=25)
plt.title('ROC Curve of KNN')
plt.legend(loc="lower right", fontsize=25)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0]) 
plt.ylim([0.0, 1.05])
plt.legend(loc="lower right", fontsize="20")
plt.plot(fpr, tpr, color='green',label='ROC Curve (area = %0.2f)' % roc_auc)
plt.legend(loc="lower right", fontsize="20")
plt.show()


actual = numpy.random.binomial(1,.9,size = 1000)
predicted = numpy.random.binomial(1,.9,size = 1000)

confusion_matrix = metrics.confusion_matrix(y_test, y_pred)

cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])
cm_display.plot()
plt.xlabel("Predicted labels", fontsize=25)
plt.ylabel("True labels", fontsize=25)
plt.title("KNN", fontsize=25)
plt.show()


test_accuracy = accuracy_score(y_test, y_pred)
test_error = 1 - test_accuracy

print("Testing Error for KNN:", test_error)


clf = RandomForestClassifier(random_state=5)
from sklearn.ensemble import BaggingClassifier
# Initialize the Bagging classifier with Decision Tree as the base estimator
bagging_clf = BaggingClassifier(estimator=clf, n_estimators=10)

# Fit the Bagging classifier on the training data
bagging_clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = bagging_clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy with Bagging:", accuracy)

accuracy_train = accuracy_score(y_train,bagging_clf.predict(X_train))
print("Training Accuracy for Bagging:", accuracy_train)


# Precision
precision = precision_score(y_test, y_pred)
print("Precision for Bagging:", precision)

# Recall
recall = recall_score(y_test, y_pred)
print("Recall for Bagging:", recall)

# F1 Score
f1 = f1_score(y_test, y_pred)
print("F1 Score for Bagging:", f1)

# F2 Score (You can adjust beta for different emphasis on precision or recall)
beta = 2
f2 = (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall)
print("F2 Score for Bagging:", f2)

# Predict the probabilities of the positive class
y_probs = bagging_clf.predict_proba(X_test)[:, 1]

# ROC AUC Score
roc_auc = roc_auc_score(y_test, y_probs)
print("ROC AUC Score for Bagging:", roc_auc)

# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_probs)

# Plot the ROC curve
plt.figure(figsize=(8,6))
plt.xlabel('False Positive Rate', fontsize=25)
plt.ylabel('True Positive Rate', fontsize=25)
plt.title('ROC Curve of Bagging')
plt.legend(loc="lower right", fontsize=25)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0]) 
plt.ylim([0.0, 1.05])
plt.legend(loc="lower right", fontsize="20")
plt.plot(fpr, tpr, color='green',label='ROC Curve (area = %0.2f)' % roc_auc)
plt.legend(loc="lower right", fontsize="20")
plt.show()


actual = numpy.random.binomial(1,.9,size = 1000)
predicted = numpy.random.binomial(1,.9,size = 1000)

confusion_matrix = metrics.confusion_matrix(y_test, y_pred)

cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])
cm_display.plot()
plt.xlabel("Predicted labels", fontsize=25)
plt.ylabel("True labels", fontsize=25)
plt.title("Bagging", fontsize=25)
plt.show()


test_accuracy = accuracy_score(y_test, y_pred)
test_error = 1 - test_accuracy

print("Testing Error for Bagging:", test_error)





clf= DecisionTreeClassifier(random_state=5)
from sklearn.ensemble import AdaBoostClassifier
ada_boost_clf = AdaBoostClassifier(estimator=clf, random_state=3)
ada_boost_clf.fit(X_train, y_train)
y_pred = ada_boost_clf.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy with AdaBoost:", accuracy)

accuracy_train = accuracy_score(y_train,ada_boost_clf.predict(X_train))
print("Training Accuracy for AdaBoost:", accuracy_train)


# Precision
precision = precision_score(y_test, y_pred)
print("Precision for AdaBoost:", precision)

# Recall
recall = recall_score(y_test, y_pred)
print("Recall for AdaBoost:", recall)

# F1 Score
f1 = f1_score(y_test, y_pred)
print("F1 Score for AdaBoost:", f1)

# F2 Score (You can adjust beta for different emphasis on precision or recall)
beta = 2
f2 = (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall)
print("F2 Score for AdaBoost:", f2)

# Predict the probabilities of the positive class
y_probs = ada_boost_clf.predict_proba(X_test)[:, 1]

# ROC AUC Score
roc_auc = roc_auc_score(y_test, y_probs)
print("ROC AUC Score for AdaBoost:", roc_auc)

# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_probs)

# Plot the ROC curve
plt.figure(figsize=(8,6))
plt.xlabel('False Positive Rate', fontsize=25)
plt.ylabel('True Positive Rate', fontsize=25)
plt.title('ROC Curve of AdaBoost')
plt.legend(loc="lower right", fontsize=25)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0]) 
plt.ylim([0.0, 1.05])
plt.legend(loc="lower right", fontsize="20")
plt.plot(fpr, tpr, color='green',label='ROC Curve (area = %0.2f)' % roc_auc)
plt.legend(loc="lower right", fontsize="20")
plt.show()


actual = numpy.random.binomial(1,.9,size = 1000)
predicted = numpy.random.binomial(1,.9,size = 1000)

confusion_matrix = metrics.confusion_matrix(y_test, y_pred)

cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])
cm_display.plot()
plt.xlabel("Predicted labels", fontsize=25)
plt.ylabel("True labels", fontsize=25)
plt.title("AdaBoost", fontsize=25)
plt.show()


test_accuracy = accuracy_score(y_test, y_pred)
test_error = 1 - test_accuracy

print("Testing Error for AdaBoost:", test_error)




from sklearn.ensemble import GradientBoostingClassifier
gb_clf = GradientBoostingClassifier(learning_rate=0.5,min_samples_leaf=5,random_state=3,max_depth=5,n_estimators=720)
gb_clf.fit(X_train, y_train)
y_pred = gb_clf.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy with GBoost:", accuracy)

accuracy_train = accuracy_score(y_train,gb_clf.predict(X_train))
print("Training Accuracy for GBoost:", accuracy_train)


# Precision
precision = precision_score(y_test, y_pred)
print("Precision for GBoost:", precision)

# Recall
recall = recall_score(y_test, y_pred)
print("Recall for GBoost:", recall)

# F1 Score
f1 = f1_score(y_test, y_pred)
print("F1 Score for GBoost:", f1)

# F2 Score (You can adjust beta for different emphasis on precision or recall)
beta = 2
f2 = (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall)
print("F2 Score for GBoost:", f2)

# Predict the probabilities of the positive class
y_probs = gb_clf.predict_proba(X_test)[:, 1]

# ROC AUC Score
roc_auc = roc_auc_score(y_test, y_probs)
print("ROC AUC Score for GBoost:", roc_auc)

# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_probs)

# Plot the ROC curve
plt.figure(figsize=(8,6))
plt.xlabel('False Positive Rate', fontsize=25)
plt.ylabel('True Positive Rate', fontsize=25)
plt.title('ROC Curve of GBoost')
plt.legend(loc="lower right", fontsize=25)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0]) 
plt.ylim([0.0, 1.05])
plt.legend(loc="lower right", fontsize="20")
plt.plot(fpr, tpr, color='green',label='ROC Curve (area = %0.2f)' % roc_auc)
plt.legend(loc="lower right", fontsize="20")
plt.show()


actual = numpy.random.binomial(1,.9,size = 1000)
predicted = numpy.random.binomial(1,.9,size = 1000)

confusion_matrix = metrics.confusion_matrix(y_test, y_pred)

cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])
cm_display.plot()
plt.xlabel("Predicted labels", fontsize=25)
plt.ylabel("True labels", fontsize=25)
plt.title("GBoost", fontsize=25)
plt.show()


test_accuracy = accuracy_score(y_test, y_pred)
test_error = 1 - test_accuracy

print("Testing Error for GBoost:", test_error)


plotdata=pd.DataFrame({
     
    "Training Accuracy":[ 93.07,95.84,   99.72 ,  97.22,  99.72,   99.72, 99.16,  99.72],
    "Testing Accuracy":[88.26, 90.50,   92.73,    93.29,    93.29,     94.97, 95.53,  96.08]
    },
    
index = ['LR','KNN', 'DT','SVM', 'AdaBoost','GBoost', 'Bagging','RF'])
plotdata.plot(kind="bar",figsize=(10,10))
plt.grid(True) 
plt.xticks(rotation=45) 
plt.legend(loc='lower right')
plt.xlabel('Classification Models', fontsize=25)
plt.ylabel("Acuuracy (in %)", fontsize=25)
