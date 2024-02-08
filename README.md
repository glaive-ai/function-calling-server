# function-calling-server

Run python3 server.py, then server is callable using client.py

## Run it in Docker

<pre>
docker build . -t function-calling-server
nvidia-docker run --rm -p 8000:8000 function-calling-server
</pre>
