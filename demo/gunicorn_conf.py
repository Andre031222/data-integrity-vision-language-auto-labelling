# Gunicorn config for the paqocha demo.
bind = "127.0.0.1:9060"
workers = 2                 # CPU inference -- keep low; each worker holds a copy of the model
threads = 2
timeout = 120               # first request loads the model (cold start)
graceful_timeout = 30
keepalive = 5
accesslog = "-"
errorlog = "-"
loglevel = "info"
