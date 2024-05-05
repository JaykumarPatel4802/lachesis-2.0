sudo energat -basepower
sleep 3
tmux new-session -d -s util_daemon './util-daemon'
tmux new-session -d -s agg_daemon 'python3.10 aggregator-daemon-new.py'