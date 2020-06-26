echo "Running local A..."
time python test.py MNIST A local 0
echo "Running local B..."
time python test.py MNIST B local 0
echo "Running local both..."
time python test.py MNIST both local 0
echo "Running federated A..."
time python test.py MNIST A federated 0
echo "Running federated B..."
time python test.py MNIST B federated 0
echo "Running weightavg A..."
time python test.py MNIST A weightavg 0
echo "Running weightavg B..."
time python test.py MNIST B weightavg 0
echo "Running cyclic A..."
time python test.py MNIST A cyclic 0
echo "Running cyclic B..."
time python test.py MNIST B cyclic 0
