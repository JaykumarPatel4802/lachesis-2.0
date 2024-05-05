cd energat_daemon
sudo rm -r __pycache__
# sudo python3.10 __main__.py --basepower
tmux new-session -d -s energy_daemon 'sudo python3.10 __main__.py'
cd ..
sleep 3
tmux new-session -d -s util_daemon './util-daemon'
tmux new-session -d -s agg_daemon 'python3.10 aggregator-daemon.py'
# sudo tmux new-session -d -s server_energy 'sudo python3 server_power.py'