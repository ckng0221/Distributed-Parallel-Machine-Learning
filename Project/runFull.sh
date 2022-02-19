echo "Doing data preprocessing..."
python3.9 DAG/ML/1_preprocessing.py 
echo "Doing knn modelling..."
python3.9 DAG/ML/2a_model_knn.py 
echo "Doing random forest modelling..."
python3.9 DAG/ML/2b_model_rf.py 
echo "Doing svc modelling..."
python3.9 DAG/ML/2c_model_svc.py 
echo "Evaluating results..."
python3.9 DAG/ML/3_res_eval.py 
cat DAG/ML/result_tab/result_table.csv 