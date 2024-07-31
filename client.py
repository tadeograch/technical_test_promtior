from langserve import RemoteRunnable
import sys

if len(sys.argv) < 2:
    print("missing input")
elif len(sys.argv) > 2:
    print("only one input accepted")
else:
    remote_chain = RemoteRunnable("http://localhost:8000/promtior_chatbot/")

    remote_chain.invoke({
        "input": "{}".format(str(sys.argv[1])),
        "chat_history": []  # Providing an empty list as this is the first call
    })