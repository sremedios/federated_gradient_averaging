echo "Running local A..."
time ./run_test_local.sh A
echo "Running local B..."
time ./run_test_local.sh B
echo "Running local both..."
time ./run_test_local.sh both
echo "Running federated A..."
time ./run_test_fl.sh A
echo "Running federated B..."
time ./run_test_fl.sh B
echo "Running modelavg A..."
time ./run_test_modelavg.sh A
echo "Running modelavg B..."
time ./run_test_modelavg.sh B
echo "Running cyclic A..."
time ./run_test_cyclic.sh A
echo "Running cyclic B..."
time ./run_test_cyclic.sh B
