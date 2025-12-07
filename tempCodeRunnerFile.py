
#  QUERY
# ---------------------------
user_query = input("Write Query Here: ")

response = qa_chain.invoke({"query": user_query})

print("\nRESULT:\n", response["result"])
print