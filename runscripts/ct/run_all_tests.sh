echo "Running local A..."
time python test.py HAM10000 A local 0
echo "Running local B..."
time python test.py HAM10000 B local 0
echo "Running local both..."
time python test.py HAM10000 both local 0
echo "Running federated A..."
time python test.py HAM10000 A federated 0
echo "Running federated B..."
time python test.py HAM10000 B federated 0
echo "Running weightavg A..."
time python test.py HAM10000 A weightavg 0
echo "Running weightavg B..."
time python test.py HAM10000 B weightavg 0
echo "Running cyclic A..."
time python test.py HAM10000 A cyclic 0
echo "Running cyclic B..."
time python test.py HAM10000 B cyclic 0
