# cd C:\\Users\\ebo\\Dropbox\\cityU\\PhD\\bug predict\\improving-performance-cross\\mahakil\\imbalanced-learn\\examples\\over-sampling
import pandas as pd
import math
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from imblearn.over_sampling import ADASYN, RandomOverSampler, SMOTE
from imblearn.under_sampling import TomekLinks, RandomUnderSampler, OneSidedSelection
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from mahakil import *
from nn_filter import nnfilter
import csv

def mergeTrain (trd):
    trdat = pd.DataFrame()
    for file in trd:
     file_path = 'train/' + file
     dataset = pd.read_csv(file_path)
     trdat = trdat.append(dataset)
    return trdat

def display(tsdat_lab, y_pred,func_name):
    function_name.append(func_name)
    CM = confusion_matrix(tsdat_lab, y_pred)
    tn = CM[0][0]
    fn = CM[1][0]
    tp = CM[1][1]
    fp = CM[0][1]
    pff = fp / (tn + fp)
    pf2 = tn / (tn + fp)
    # recall or sensitivity
    pd = tp / (tp + fn)
    # G-MEAN
    gmean = math.sqrt(pd * pf2)
    # balance
    #numt = ((1 - pd) * 2) + ((0 - pff) * 2)
    #bal = 1 - (math.sqrt(numt) / math.sqrt(2))
    #precision = metrics.precision_score(tsdat_lab, y_pred)
    #precision = round(precision * 100, 2)
    #print('precision', precision)
    #prec.append(precision)
    recall = metrics.recall_score(tsdat_lab, y_pred)
    recall = round(recall * 100, 2)
    print('recall', recall )
    reca.append(recall)
    # f1 = metrics.f1_score(tsdat_lab, y_pred)
    # f1 =round(f1 * 100, 2)
    # print('f1_score', f1)
    # f1_value.append(f1)
    auc = metrics.roc_auc_score(tsdat_lab, y_pred)
    auc = round(auc * 100, 2)
    print('auc', auc)
    auc_value.append(auc)
    pff = round(pff * 100, 2)
    print('pf', pff)
    pf_value.append(pff)
    gmean = round(gmean * 100, 2)
    print('gmean',gmean)
    gmean_value.append(gmean)
    # bal = round(bal * 100, 2)
    # print('bal', bal)
    # bal_value.append(bal)
    print('\n')

def finform(function_name, reca, auc_value, pf_value,gmean_value, trname, tsname):
    #function_name.append('model')
    function_name.append( 'train')
    function_name.append( 'test')
   # prec.append(mod_name)
   #  prec.append(trname)
   #  prec.append(tsname)
    #reca.append(mod_name)
    reca.append(trname)
    reca.append(tsname)
   # f1_value.append(mod_name)
   #  f1_value.append(trname)
   #  f1_value.append(tsname)
    #auc_value.append(mod_name)
    auc_value.append(trname)
    auc_value.append(tsname)
    pf_value.append(trname)
    pf_value.append(tsname)
    gmean_value.append(trname)
    gmean_value.append(tsname)
    # bal_value.append(trname)
    # bal_value.append(tsname)
    print('\n')

def pipeline (X_resampled, y_resampled,tsdat,tsdat_lab, datsamp):
    # NB
    func_name = datsamp + '_NB'
    clf = GaussianNB()
    clf.fit(X_resampled, y_resampled)
    y_pred = clf.predict(tsdat)
    display(tsdat_lab, y_pred, func_name)
    # RF
    # Instantiate model with 1000 decision trees
    func_name = datsamp + '_RF'
    rf = RandomForestClassifier(n_estimators=1000, random_state=42)
    rf.fit(X_resampled, y_resampled)
    rf_pred = rf.predict(tsdat)
    display(tsdat_lab, rf_pred, func_name)
    # SVM
    func_name = datsamp + '_SVM'
    svclassifier = SVC(kernel='linear')
    svclassifier.fit(X_resampled, y_resampled)
    sy_pred = svclassifier.predict(tsdat)
    display(tsdat_lab, sy_pred, func_name)
    # KNN
    func_name = datsamp + '_KNN'
    neighclas = KNeighborsClassifier(n_neighbors=3)
    neighclas.fit(X_resampled, y_resampled)
    kn_pred = neighclas.predict(tsdat)
    display(tsdat_lab, kn_pred, func_name)
    # NNET  clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
    func_name = datsamp + '_NNET'
    mlp = MLPClassifier(hidden_layer_sizes=(30, 30, 30))
    mlp.fit(X_resampled, y_resampled)
    nnet_pred = mlp.predict(tsdat)
    display(tsdat_lab, nnet_pred, func_name)
    # XGBOOST
    func_name = datsamp + '_XGB'
    xgb = XGBClassifier()
    xgb.fit(X_resampled, y_resampled)
    xgb_pred = xgb.predict(tsdat)
    display(tsdat_lab, xgb_pred, func_name)
#'prop-1.csv', 'prop-2.csv', 'prop-3.csv','arc.csv','pdftranslator.csv','skarbonka.csv','redaktor.csv','workflow.csv','zuzel.csv','systemdata.csv',
           # 'kalkulator.csv', 'serapion.csv', 'nieruchomosci.csv',  'berek.csv',
trfiles=['ant-1.4.csv', 'ant-1.5.csv', 'ant-1.6.csv',  'ant-1.3.csv',  'pbeans1.csv','pbeans2.csv',
         'synapse-1.0.csv',  'xalan-2.4.csv','camel-1.0.csv', 'camel-1.4.csv', 'camel-1.6.csv','xerces-1.2.csv',
         'xerces-1.3.csv',  'ivy-1.1.csv','ivy-1.4.csv', 'ivy-2.0.csv','jedit-4.3.csv', 'jedit-4.2.csv', 'jedit-4.0.csv', 'jedit-4.1.csv']


testfiles=[ 'serapion.csv']
#file_path = "tests/arc.csv"
for file in testfiles:
 outp='results/' + file
 file_path = 'tests/' + file
 tsname = file
 tsdat = pd.read_csv(file_path)
 tsdat_lab = tsdat.bug
 tsdat = tsdat.drop('bug', axis=1)
    # declare parameters for output
 function_name = ['evalmet']
 prec = ['Precision']
 reca = ['recall']
 f1_value = ['f1_score']
 auc_value = ['auc']
 pf_value = ['pf']
 gmean_value = ['gmean']
 bal_value = ['bal']
    #import train data
 trainset = mergeTrain(trfiles)
 coname = list(trainset.columns.values)
    #find filtered data
 nfil = nnfilter()
 trdat = nfil.filter(trainset,tsdat)
 trdat.columns = trdat.columns[:0].tolist() + coname
 print(trdat)
 trname = 'burak'
 b = 1
    #extract trdat
 # for file in trfiles:
 #    file_path = 'train/' + file
 #    trname = file
 #    trdat = pd.read_csv(file_path)
 trdat_lab = trdat.bug
 trdat = trdat.drop('bug', axis=1)
 #numpy arrays don't work well with mahakil so use old format
 trdatm = trdat
 trdatm_lab = trdat_lab
 #convert data into numpy arrays that speed up the computations:
 tsdat = np.array(tsdat)
 tsdat_lab =  np.array(tsdat_lab)
 trdat = np.array(trdat)
 trdat_lab =  np.array(trdat_lab)
    #Apply sampling method
    #no sampling
 pipeline(trdat, trdat_lab,tsdat, tsdat_lab, datsamp='NOS')
    #MAHAKIL
 mahak = MAHAKIL()
 X_resampled, y_resampled = mahak.fit_sample(trdatm, trdatm_lab)
 pipeline(X_resampled, y_resampled, tsdat, tsdat_lab, datsamp='MAHAKIL')
    # Apply adasyn
 ada = ADASYN()
 X_resampled, y_resampled = ada.fit_sample(trdat, trdat_lab)
 pipeline(X_resampled, y_resampled, tsdat, tsdat_lab, datsamp='ADASYN')
    #ROS
 ros = RandomOverSampler()
 X_resampled, y_resampled = ros.fit_sample(trdat, trdat_lab)
 pipeline(X_resampled, y_resampled, tsdat, tsdat_lab, datsamp='ROS')
    #SMOTE
 kind = ['regular', 'borderline1']
 sm = [SMOTE(kind=k) for k in kind]
 X_resampled = []
 y_resampled = []
 for method in sm:
        X_res, y_res = method.fit_sample(trdat, trdat_lab)
        X_resampled.append(X_res)
        y_resampled.append(y_res)
 for i in range(len(kind)):
        print('Smote_' + kind[i])
        OS_name = 'Smote_' + kind[i]
        pipeline(X_resampled[i], y_resampled[i], tsdat, tsdat_lab, datsamp=OS_name)
 # remove Tomek links
 tl = TomekLinks(return_indices=True)
 X_resampled, y_resampled, idx_resampled = tl.fit_sample(trdat, trdat_lab)
 pipeline(X_resampled, y_resampled, tsdat, tsdat_lab, datsamp='Tomek')
 # Apply the random under-sampling
 rus = RandomUnderSampler(return_indices=True)
 X_resampled, y_resampled, idx_resampled = rus.fit_sample(trdat, trdat_lab)
 pipeline(X_resampled, y_resampled, tsdat, tsdat_lab, datsamp='RUS')
 # Apply One-Sided Selection
 oss = OneSidedSelection(return_indices=True)
 X_resampled, y_resampled, idx_resampled = oss.fit_sample(trdat, trdat_lab)
 #execute training models and test
 pipeline(X_resampled, y_resampled, tsdat, tsdat_lab, datsamp='OSS')
 out = open(outp, 'a', newline='')
 csv_write = csv.writer(out, dialect='excel')
 #compute performance measures
 finform(function_name,  reca, auc_value, pf_value, gmean_value, trname, tsname)
 #save results in excel sheet
 if (b == 1):
    csv_write.writerow(function_name)
 #csv_write.writerow(prec)
 csv_write.writerow(reca)
 #csv_write.writerow(f1_value)
 csv_write.writerow(auc_value)
 csv_write.writerow(pf_value)
 csv_write.writerow(gmean_value)
 #csv_write.writerow(bal_value)
 csv_write.writerow('\n')
 b=b+1



#print(X_resampled)
#print(y_resampled)
#print(len(X_resampled))
#print(len(y_resampled))

