#!/bin/bash

# Initialize model files from Drive if needed
echo "Checking for model files..."
python -c "from drive_utils import ensure_model_files; ensure_model_files()" || exit 1

# Start Gunicorn for the Flask API
gunicorn -w 1 -b 0.0.0.0:8000 api:app &

# Start the Kafka consumer
python kafka_consumer.py &

# Wait for any process to exit
wait -n

# Exit with status of process that exited first
exit $? 