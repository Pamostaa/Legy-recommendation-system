import redis

r = redis.Redis(host='localhost', port=6379, db=0)

# Test set
r.set('test_key', 'Hello, Redis!', ex=60)  # expires in 60 seconds

# Test get
value = r.get('test_key')
print("Value from Redis:", value.decode() if value else "None")

