import pandas as pd 
import numpy as np 
import pickle

path1 = 'D:/Data/Bad_Loan_Project/Model/RFA_XGBoost_model.pickle'
path2 = 'D:/Data/Bad_Loan_Project/Model/test.pickle' 

model = pickle.load(open(path1,'rb'))
test_model = pickle.load(open(path2,'rb')) 

class XGBoost():
    def bad_loan_predict(self,loan_amnt,term,int_rate,emp_length,home_ownership,annual_inc,purpose,addr_state,dti,delinq_2yrs,revol_util,total_acc,longest_credit_length,verification_status):
        z = np.zeros(len(test_model.columns))
        z[0] = loan_amnt
        z[1] = term
        z[2] = int_rate
        z[3] = emp_length
        z[4] = home_ownership
        z[5] = annual_inc
        z[6] = purpose
        z[7] = addr_state
        z[8] = dti
        z[9] = delinq_2yrs
        z[10] = revol_util
        z[11] = total_acc
        z[12] = longest_credit_length
        z[13] = verification_status
        Y = (pd.DataFrame(z)).T
                                        
        return model.predict(Y)[0]
if __name__ == "__main__":
    xgb = XGBoost()
    loan = xgb.bad_loan_predict(23475,1,20.49,7.0,1,11.034890,11,27,14.22,0.0,70.7,36.0,11.0,1)
    print('The Prediction Is : ',loan)
