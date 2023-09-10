# hyphen-challenges


### Challenge 1: Chat Conversations -  Indexing & Querying
### Objectives:

- [ ]  Index provided example chats in appropriate index (eg. vector store)
- [ ]  Construct a knowledge graph from the chat messages. Use NLP tasks like information extraction (NER etc.) to extract triplets and build the knowledge graph
- [ ]  New data should be merged with the existing data in the knowledge graph (eg. first index `chat-1.json` , then index `chat-2.json` etc. and see how the KG evolves)
- [ ]  A simple interface or API for querying. The query will be a string (which can be the context of a chat conversation) and the response should be the relevant knowledge that can be fed back to an AI agent to inform a reply to the user.