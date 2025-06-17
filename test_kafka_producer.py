from kafka_producer import send_kafka_message

# Sample recommendation request
message = {
    "user_id": "6821cb8fb5ce7f4dd4d2ad73"
}

# Send to recommendation_requests topic
send_kafka_message("recommendation_requests", message)