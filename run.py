from flask import Flask, request 
import testing
app = Flask(__name__)

@app.route('/get_predict_loan', methods = ['GET', 'POST'])

def get_predict_loan():
    if request.method == 'POST':
        loan_amount = float(request.form['loan_amnt'])
        term = int(request.form['term'])
        int_rate = float(request.form['int_rate'])
        emp_length = float(request.form['emp_length'])
        home_ownership = int(request.form['home_ownership'])
        annual_inc = float(request.form['annual_inc']) 
        purpose = int(request.form['purpose'])
        addr_state = int(request.form['addr_state'])
        dti = float(request.form['dti'])
        delinq_2yrs = float(request.form['delinq_2yrs'])
        revol_util = float(request.form['revol_util'])
        total_acc = float(request.form['total_acc'])
        longest_credit_length = float(request.form['longest_credit_length'])
        verification_status = int(request.form['verification_status'])

        prediction = testing.XGBoost().bad_loan_predict(loan_amount,term,int_rate,emp_length,home_ownership,
        annual_inc,purpose,addr_state,dti,delinq_2yrs,revol_util,total_acc,longest_credit_length,
        verification_status)
        return 'Prediction of loan is : {}'.format(prediction) 

if __name__ == '__main__':
    print('... Starting Python Flask Server For Loan Prediction ....')
    app.run()

