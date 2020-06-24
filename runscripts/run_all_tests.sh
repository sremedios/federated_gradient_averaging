echo "Running local A..."
time python test.py models/weights/ A local 0
echo "Running local B..."
time python test.py models/weights/ B local 0
echo "Running local both..."
time python test.py models/weights/ both local 0
echo "Running federated A..."
time python test.py models/weights/ A federated 0
echo "Running federated B..."
time python test.py models/weights/ B federated 0
echo "Running weightavg A..."
time python test.py models/weights/ A weightavg 0
echo "Running weightavg B..."
time python test.py models/weights/ B weightavg 0
echo "Running cyclic A..."
time python test.py models/weights/ A cyclic 0
echo "Running cyclic B..."
time python test.py models/weights/ B cyclic 0
