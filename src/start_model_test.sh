cd energy_model
tmux new-session -d -s model 'python3 bayesian_regressor.py'
cd ..
sleep 2
tmux new-session -d -s energy 'python3 energy-experiment.py'
