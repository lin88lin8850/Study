ps -ef | grep "python3 async_llm_server.py" | awk '{print $2}' | xargs kill -9
rm -f *.log