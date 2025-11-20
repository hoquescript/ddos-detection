launch:
	@echo "Starting Flask server..."
	@pkill -f server.py || true
	python3 demo/server.py & \
	sleep 2 && \
	open http://127.0.0.1:3000


train:
	@echo "Training model..."
	python3 ddos_experiment.py

server: 
	@echo "Starting Flask server..."
	python3 demo/server.py